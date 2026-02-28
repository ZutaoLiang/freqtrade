from typing import Optional
import pandas_ta as pta
import pandas as pd
# pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_helper import stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Order, Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib

from datetime import datetime, timezone, timedelta
import logging
logger = logging.getLogger(__name__)


class LongShortV1(IStrategy):
    
    timeframe = '15m'
    
    minimal_roi = {"0": 100}
    can_short = True
    process_only_new_candles = True
    informative_prefix = "market_"
    
    is_portfolio_exit = False
    portfolio_exit_reason = ''
    portfolio_profit_dict = {}

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.base_pair = self.get_config("base_pair", "ETH/USDT:USDT")
        self.base_pair_is_strong = self.get_config("base_pair_is_strong", False)
        self.portfolio_take_profit_amount = self.get_config("portfolio_take_profit_amount", 5)
        self.portfolio_stoploss_amount = self.get_config("portfolio_stoploss_amount", -5)
        self.portfolio_exit_only = self.get_config("portfolio_exit_only", False)
        
        self.stake_amount = self.get_config("stake_amount", 6)
        self.trade_leverage = self.get_config("trade_leverage", 3)
        
        self.trailing_stop = self.get_config("trailing_stop", True)
        if not self.trailing_stop:
            self.custom_trailing_stop = self.get_config("custom_trailing_stop", False)
        else:
            self.trailing_stop_positive = self.get_config("base_trailing_stop", 0.12) * self.trade_leverage
            self.trailing_stop_positive_offset = self.get_config("base_trailing_stop_offset", 0.3) * self.trade_leverage
            self.trailing_only_offset_is_reached = self.get_config("trailing_only_offset_is_reached", True)

        self.base_stop_loss = self.get_config("base_stop_loss", 0.07)
        self.stoploss = - float(self.base_stop_loss * self.trade_leverage)
        self.use_custom_stoploss = self.get_config("use_custom_stoploss", False)
        self.atr_stop_loss_multiplier = self.get_config("atr_stop_loss_multiplier", 0)
        
        self.use_ha_candles = self.get_config("use_ha_candles", False)
        
        self.trend_length = self.get_config("trend_length", 3)
 
        self.ma_short_length = self.get_config("ma_short_length", 0)
        self.ma_mid_length = self.get_config("ma_mid_length", 0)
        self.ma_long_length = self.get_config("ma_long_length", 0)
        
        self.crossover_lookback_length = self.get_config("crossover_lookback_length", 8)

        self.startup_candle_count = int(max(self.ma_mid_length, self.ma_long_length) * 1)
        
        self.atr_period = self.get_config("atr_period", 21)
        
        self.fee = self.get_config("fee", 0.0005)
 
        self.long_time_low_profit_minutes = self.get_config("long_time_low_profit_minutes", 0)
        self.long_time_low_profit_max = self.get_config("long_time_low_profit_max", 0.05)
        self.long_time_low_profit_lower_bound = self.get_config("long_time_low_profit_lower_bound", 0.003)
        self.long_time_low_profit_upper_bound = self.get_config("long_time_low_profit_upper_bound", 0.02)
        
        self.long_time_stoploss_minutes = self.get_config("long_time_stoploss_minutes", 0)
        self.long_time_stoploss_profit = self.get_config("long_time_stoploss_profit", 0.03)
        
        self.cooldown_candles = self.get_config("cooldown_candles", 1)
        self.stoploss_guard_lookback_period_candles = self.get_config("stoploss_guard_lookback_period_candles", 0)  # 0 is disabled
        self.stoploss_guard_trade_limit = self.get_config("stoploss_guard_trade_limit", 4)
        self.stoploss_guard_stop_duration_candles = self.get_config("stoploss_guard_stop_duration_candles", 2)
        self.max_drawdown_lookback_period = self.get_config("max_drawdown_lookback_period", 0)  # 0 is disabled
        self.max_drawdown_stop_duration = self.get_config("max_drawdown_stop_duration", 60)
        self.max_allowed_drawdown = self.get_config("max_allowed_drawdown", 0.3)
     
    def get_config(self, key: str, default):
        return self.config.get(key, default)

    def informative_pairs(self):
        informative_pairs = [(self.base_pair, self.timeframe)]
        return informative_pairs

    def calculate_ha(self, df: DataFrame) -> DataFrame:
        if self.use_ha_candles:
            df_ref = qtpylib.heikinashi(df)
        else:
            df_ref = df
        
        df['ha_open'] = df_ref['open']
        df['ha_high'] = df_ref['high']
        df['ha_low'] = df_ref['low']
        df['ha_close'] = df_ref['close']
        return df
 
    def calc_ma(self, close, length: int):
        # ma = pta.ema(close=close, length=length, talib=False)
        ma = pta.ema(close=close, length=length, talib=False)
        return ma.ffill() if ma is not None else ma

    def rename_informative(self, informative: DataFrame, columns: list):
        return informative[['date'] + columns].add_prefix(self.informative_prefix) \
                .rename(columns={f'{self.informative_prefix}date': 'date'})

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            dataframe = self.calculate_ha(dataframe)
            
            informative = self.dp.get_pair_dataframe(pair=self.base_pair, timeframe=self.timeframe)
            informative = self.calculate_ha(informative)
            informative_rename = self.rename_informative(informative, ['ha_open', 'ha_high', 'ha_low', 'ha_close'])
            dataframe = pd.merge(dataframe, informative_rename, on='date', how='left')
            
            # ma
            if self.ma_short_length > 0:
                dataframe['ma_short'] = self.calc_ma(close=dataframe['ha_close'], length=self.ma_short_length)
                informative['ma_short'] = self.calc_ma(close=informative['ha_close'], length=self.ma_short_length)
                dataframe = pd.merge(dataframe, self.rename_informative(informative, ['ma_short']), on='date', how='left')
            
            if self.ma_mid_length > 0:
                dataframe['ma_mid'] = self.calc_ma(close=dataframe['ha_close'], length=self.ma_mid_length)
                informative['ma_mid'] = self.calc_ma(close=informative['ha_close'], length=self.ma_mid_length)
                dataframe = pd.merge(dataframe, self.rename_informative(informative, ['ma_mid']), on='date', how='left')

            if self.ma_long_length > 0:
                dataframe['ma_long'] = self.calc_ma(close=dataframe['ha_close'], length=self.ma_long_length)
                informative['ma_long'] = self.calc_ma(close=informative['ha_close'], length=self.ma_long_length)
                dataframe = pd.merge(dataframe, self.rename_informative(informative, ['ma_long']), on='date', how='left')
            
            # atr
            dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_period)
            dataframe['natr'] = pta.natr(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], length=self.atr_period, talib=False, scalar=1.0)

            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_indicators: {e}")
            return dataframe
        
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        if dataframe.empty:
            return dataframe
        
        try:
            informative_up_mask = \
                    (dataframe[f'{self.informative_prefix}ha_close'] >= dataframe[f'{self.informative_prefix}ma_short']) \
                    & (dataframe[f'{self.informative_prefix}ha_close'] >= dataframe[f'{self.informative_prefix}ma_mid']) \
                    & (dataframe[f'{self.informative_prefix}ma_short'] > dataframe[f'{self.informative_prefix}ma_mid']) \
                    & (self.indicator_up_n_periods_mask(dataframe, f'{self.informative_prefix}ma_short', self.trend_length)) \
                    & (self.indicator_up_n_periods_mask(dataframe, f'{self.informative_prefix}ma_mid', self.trend_length))
            informative_down_mask = \
                    (dataframe[f'{self.informative_prefix}ha_close'] <= dataframe[f'{self.informative_prefix}ma_short']) \
                    & (dataframe[f'{self.informative_prefix}ha_close'] <= dataframe[f'{self.informative_prefix}ma_mid']) \
                    & (dataframe[f'{self.informative_prefix}ma_short'] < dataframe[f'{self.informative_prefix}ma_mid']) \
                    & (self.indicator_down_n_periods_mask(dataframe, f'{self.informative_prefix}ma_short', self.trend_length)) \
                    & (self.indicator_down_n_periods_mask(dataframe, f'{self.informative_prefix}ma_mid', self.trend_length))
                    
            if self.base_pair_is_strong:
                if metadata['pair'] == self.base_pair:
                    dataframe.loc[informative_up_mask, ['enter_long', 'enter_tag']] = (1, 'entry_long')
                    dataframe.loc[informative_down_mask, ['enter_short', 'enter_tag']] = (1, 'entry_short')
                else:
                    dataframe.loc[informative_up_mask, ['enter_short', 'enter_tag']] = (1, 'entry_short')
                    dataframe.loc[informative_down_mask, ['enter_long', 'enter_tag']] = (1, 'entry_long')
            else:
                if metadata['pair'] == self.base_pair:
                    dataframe.loc[informative_up_mask, ['enter_short', 'enter_tag']] = (1, 'entry_short')
                    dataframe.loc[informative_down_mask, ['enter_long', 'enter_tag']] = (1, 'entry_long')
                else:
                    dataframe.loc[informative_up_mask, ['enter_long', 'enter_tag']] = (1, 'entry_long')
                    dataframe.loc[informative_down_mask, ['enter_short', 'enter_tag']] = (1, 'entry_short')
                          
            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_entry_trend: {e}")
            return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe
        
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        if self.portfolio_exit_only:
            return dataframe
        
        try:
            informative_up_mask = (dataframe[f'{self.informative_prefix}ma_short'] > dataframe[f'{self.informative_prefix}ma_mid'])
            informative_down_mask = (dataframe[f'{self.informative_prefix}ma_short'] < dataframe[f'{self.informative_prefix}ma_mid'])
            
            if self.base_pair_is_strong:
                if metadata['pair'] == self.base_pair:
                    dataframe.loc[informative_down_mask, ['exit_long', 'exit_tag']] = (1, 'exit_ma')
                    dataframe.loc[informative_up_mask, ['exit_short', 'exit_tag']] = (1, 'exit_ma')
                else:
                    dataframe.loc[informative_down_mask, ['exit_short', 'exit_tag']] = (1, 'exit_ma')
                    dataframe.loc[informative_up_mask, ['exit_long', 'exit_tag']] = (1, 'exit_ma')
            else:
                if metadata['pair'] == self.base_pair:
                    dataframe.loc[informative_down_mask, ['exit_short', 'exit_tag']] = (1, 'exit_ma')
                    dataframe.loc[informative_up_mask, ['exit_long', 'exit_tag']] = (1, 'exit_ma')
                else:
                    dataframe.loc[informative_down_mask, ['exit_long', 'exit_tag']] = (1, 'exit_ma')
                    dataframe.loc[informative_up_mask, ['exit_short', 'exit_tag']] = (1, 'exit_ma')
                
            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_exit_trend: {e}")
            return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, after_fill: bool,
                       **kwargs) -> Optional[float]:
        if not self.use_custom_stoploss:
            return None
        
        if self.atr_stop_loss_multiplier <= 0:
            return None
        
        leverage = trade.leverage
        is_short = trade.is_short
        open_rate = trade.open_rate
        _current_profit = current_profit / leverage

        if after_fill:
            filled_orders = trade.select_filled_orders()
            count_of_orders = len(filled_orders)
            if count_of_orders == 0:
                return None
            
            # last_filled_price = filled_orders[-1].average
            last_candle = self.get_last_candle(trade)
            atr = last_candle['atr']
            natr = last_candle['natr']
            
            if is_short:
                stop_rate_atr = open_rate + (self.atr_stop_loss_multiplier * atr) 
                stop_rate_abs = open_rate * (1 + self.base_stop_loss)
                stop_rate = min(stop_rate_atr, stop_rate_abs)
            else:
                stop_rate_atr = open_rate - (self.atr_stop_loss_multiplier * atr)
                stop_rate_abs = open_rate * (1 - self.base_stop_loss)
                stop_rate = max(stop_rate_atr, stop_rate_abs)
            
            if count_of_orders == 1:
                logger.info(f'Set {trade.pair} after fill #{count_of_orders} stoploss rate to:{stop_rate:.6f}'
                            f'(stop_rate_atr:{stop_rate_atr:.6f}, stop_rate_abs:{stop_rate_abs:.6f}), '
                            f'[new_open_rate:{open_rate:.6f}](stop/open dist:{abs(stop_rate/open_rate-1):.2%}, atr:{atr:.6f}, natr:{natr:.2%}), '
                            f'current_rate:{current_rate:.6f}, '
                            f'current_profit:{current_profit:.2%}(without leverage:{_current_profit:.2%}) at {current_time}')
            else:
                logger.info(f'Set {trade.pair} after fill #{count_of_orders} stoploss rate to:{stop_rate:.6f}'
                            f'(stop_rate_atr:{stop_rate_atr:.6f}, stop_rate_abs:{stop_rate_abs:.6f}), '
                            f'[new_open_rate:{open_rate:.6f}](stop/open dist:{abs(stop_rate/open_rate-1):.2%}, atr:{atr:.6f}, natr:{natr:.2%}), '
                            f'current_rate:{current_rate:.6f}, '
                            f'current_profit:{current_profit:.2%}(without leverage:{_current_profit:.2%}) at {current_time}')
            return stoploss_from_absolute(stop_rate, current_rate, is_short, leverage)
        
        if self.custom_trailing_stop:
            if _current_profit > self.get_config("base_trailing_stop_offset", 0.3):
                return self.get_config("base_trailing_stop", 0.12) * leverage
 
        if self.long_time_stoploss_minutes > 0:
            open_minutes = round((current_time - trade.open_date_utc).total_seconds() / 60, 1)
            if open_minutes > self.long_time_stoploss_minutes and _current_profit > (self.long_time_stoploss_profit + 0.005):
                    return stoploss_from_open(self.long_time_stoploss_profit * leverage, current_profit, is_short, leverage)

        return None

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        if self.is_portfolio_exit:
            return self.portfolio_exit_reason
        
        profit_abs = trade.calculate_profit(rate=current_rate).profit_abs
        self.portfolio_profit_dict[pair] = profit_abs
        
        open_rate = trade.open_rate
        leverage = trade.leverage
        _current_profit = current_profit / leverage
        
        if self.long_time_low_profit_minutes > 0:
            if trade.is_short:
                max_profit = (open_rate - trade.min_rate) / open_rate - 2 * self.fee
            else:
                max_profit = (trade.max_rate - open_rate) / open_rate - 2 * self.fee
                
            open_minutes = round((current_time - trade.open_date_utc).total_seconds() / 60, 1)
            if open_minutes > self.long_time_low_profit_minutes:
                if max_profit < self.long_time_low_profit_max and self.long_time_low_profit_lower_bound < _current_profit < self.long_time_low_profit_upper_bound:
                    return "longtime_low_profit"
        
        return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        if self.is_portfolio_exit:
            return False 

        return True
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        self.is_portfolio_exit = False
        self.portfolio_exit_reason = ''
        
        open_trades = Trade.get_open_trades()
        open_pairs = [t.pair for t in open_trades]
        
        keys_to_remove = [p for p in self.portfolio_profit_dict.keys() if p not in open_pairs]
        for p in keys_to_remove:
            del self.portfolio_profit_dict[p]
            
        if not open_trades:
            return
        
        total_profit_abs = sum(self.portfolio_profit_dict.values())
        if total_profit_abs >= self.portfolio_take_profit_amount:
            self.is_portfolio_exit = True
            self.portfolio_exit_reason = 'portfolio_take_profit'
            logger.info(f'Total portfolio profit reached take profit:{total_profit_abs:.2f} >= {self.portfolio_take_profit_amount:.2f}, exiting portfolio')
        elif total_profit_abs <= self.portfolio_stoploss_amount:
            self.is_portfolio_exit = True
            self.portfolio_exit_reason = 'portfolio_stoploss'
            logger.info(f'Total portfolio profit reached stoploss:{total_profit_abs:.2f} <= {self.portfolio_stoploss_amount:.2f}, exiting portfolio')
            
    def get_last_candle(self, trade: Trade):
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        return last_candle
 
    def indicator_up_n_periods_mask(self, dataframe: DataFrame, indicator: str, days: int):
        indicator_up_mask = (dataframe[f'{indicator}'] > dataframe[f'{indicator}'].shift(1))
        for i in range(2, days):
            indicator_up_mask = indicator_up_mask & (dataframe[f'{indicator}'].shift(i-1) > dataframe[f'{indicator}'].shift(i))
        return indicator_up_mask
    
    def indicator_down_n_periods_mask(self, dataframe: DataFrame, indicator: str, days: int):
        indicator_down_mask = (dataframe[f'{indicator}'] < dataframe[f'{indicator}'].shift(1))
        for i in range(2, days):
            indicator_down_mask = indicator_down_mask & (dataframe[f'{indicator}'].shift(i-1) < dataframe[f'{indicator}'].shift(i))
        return indicator_down_mask

    def leverage(
        self, 
        pair: str, 
        current_time: datetime, 
        current_rate: float, 
        proposed_leverage: float, 
        max_leverage: float, 
        entry_tag: Optional[str], 
        side: str, 
        **kwargs
    ) -> float:
        return self.trade_leverage

    @property
    def protections(self): # type: ignore
        protections = []
        
        if self.cooldown_candles > 0:
            protections.append(
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": self.cooldown_candles,
                }
            )
        
        if self.stoploss_guard_lookback_period_candles > 0:
            protections.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": self.stoploss_guard_lookback_period_candles,
                    "trade_limit": self.stoploss_guard_trade_limit,
                    "stop_duration_candles": self.stoploss_guard_stop_duration_candles,
                    "only_per_pair": False
                }
            )
        
        if self.max_drawdown_lookback_period > 0:
            protections.append(
                {
                    "method": "MaxDrawdown",
                    "lookback_period": self.max_drawdown_lookback_period,
                    "stop_duration": self.max_drawdown_stop_duration,
                    "max_allowed_drawdown": self.max_allowed_drawdown,
                }
            )
        
        return protections

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        if pair == self.base_pair:
            pairs = self.dp.current_whitelist()
            stake_amount = proposed_stake * (len(pairs) - 1)
        else:
            stake_amount = proposed_stake
        
        return stake_amount