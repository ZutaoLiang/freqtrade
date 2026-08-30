from typing import Optional
import pandas_ta as pta
import numpy as np
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
import os
import json
from freqtrade.enums import RunMode
logger = logging.getLogger(__name__)
# TODO: remove this
logging.getLogger('freqtrade.leverage.liquidation_price').setLevel(logging.WARNING)


class LongShortV3BetaW(IStrategy):
    
    timeframe = '15m'
    
    minimal_roi = {"0": 100}
    can_short = True
    process_only_new_candles = True
    informative_prefix = "market_"
    
    is_portfolio_exit = False
    portfolio_exit_reason = ''
    portfolio_profit_dict = {}
    portfolio_max_profit = -100000
    portfolio_min_profit = 100000

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
        main_pairs_str = self.get_config("main_pairs", "ETH/USDT:USDT").split(",")
        self.main_pairs = [s.strip() for s in main_pairs_str]
        
        self.portfolio_take_profit_amount = self.get_config("portfolio_take_profit_amount", 5)
        self.take_profit_decay_factor = self.get_config("take_profit_decay_factor", 48)
        self.portfolio_stoploss_amount = self.get_config("portfolio_stoploss_amount", -5)
        self.portfolio_exit_only = self.get_config("portfolio_exit_only", False)
        self.enable_dynamic_stake = self.get_config("enable_dynamic_stake", False)
        self.beta_window_days = self.get_config("beta_window_days", 90)
        self.beta_clip_lo = self.get_config("beta_clip_lo", 0.5)
        self.beta_clip_hi = self.get_config("beta_clip_hi", 3.0)
        
        self.portfolio_trailing_stop_activation = self.get_config("portfolio_trailing_stop_activation", 0)
        self.portfolio_trailing_stop_drawback_ratio = self.get_config("portfolio_trailing_stop_drawback_ratio", 0.3)
        
        self.total_stake_amount = self.get_config("total_stake_amount", 6)
        self.min_notional = self.get_config("min_notional", 20.5)
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

        self.startup_candle_count = int(max(self.ma_mid_length, self.ma_long_length) * 1)
        
        self.atr_period = self.get_config("atr_period", 21)
        
        self.fee = self.get_config("fee", 0.0005)        
        
        self.analyse_portfolio_spread = self.get_config("analyse_portfolio_spread", False)
        self.spread_avg_window = self.get_config("spread_avg_window", 15)
        self.spread_zscore_threshold = self.get_config("spread_zscore_threshold", 0.5)

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
        
        self.portfolio_cooldown_minutes = self.get_config("portfolio_cooldown_minutes", 8)
        self.next_entry_time_file = f"next_entry_time_{self.__class__.__name__}.json"

    def get_config(self, key: str, default):
        return self.config.get(key, default)

    def _get_next_entry_time(self) -> datetime | None:
        if self.config.get("runmode") in (RunMode.BACKTEST, RunMode.HYPEROPT):
            return None
            
        try:
            if os.path.exists(self.next_entry_time_file):
                with open(self.next_entry_time_file, 'r') as f:
                    data = json.load(f)
                    if 'next_entry_time' in data:
                        return datetime.fromisoformat(data['next_entry_time'])
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to read next entry time file: {e}")
        return None

    def _set_next_entry_time(self, next_time: datetime) -> None:
        if self.config.get("runmode") in (RunMode.BACKTEST, RunMode.HYPEROPT):
            return
            
        try:
            with open(self.next_entry_time_file, 'w') as f:
                json.dump({'next_entry_time': next_time.isoformat()}, f)
            logger.info(f"Set next entry time to {next_time} (file: {self.next_entry_time_file})")
        except Exception as e:
            logger.error(f"Failed to write next entry time file: {e}")

    def _is_entry_time_valid(self, current_time: datetime) -> bool:
        if self.config.get("runmode") in (RunMode.BACKTEST, RunMode.HYPEROPT):
            return True
            
        next_entry_time = self._get_next_entry_time()
        if next_entry_time is None:
            return True
            
        two_cooldown_later = current_time + timedelta(minutes=2 * self.portfolio_cooldown_minutes)
        if two_cooldown_later < next_entry_time:
            logger.info(f"Detected backtesting mode: current_time={current_time}, "
                        f"two_cooldown_later={two_cooldown_later}, "
                        f"stored_next_entry_time={next_entry_time}")
            return True
            
        return current_time >= next_entry_time

    def informative_pairs(self):
        informative_pairs = [(self.main_pairs[0], self.timeframe)]
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
            
            informative = self.dp.get_pair_dataframe(pair=self.main_pairs[0], timeframe=self.timeframe)
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

            # rolling hedge-ratio beta: OLS slope of log(alt) on log(main), hourly closes, trailing window.
            # Computed here because get_analyzed_dataframe only returns ~1000 candles in backtest.
            try:
                base = self.dp.get_pair_dataframe(pair=self.main_pairs[0], timeframe=self.timeframe)
                a = dataframe[['date', 'close']].set_index('date')['close'].resample('1h').last()
                e = base[['date', 'close']].set_index('date')['close'].resample('1h').last()
                j = pd.concat([a.rename('a'), e.rename('e')], axis=1, join='inner').dropna()
                le, la = np.log(j['e']), np.log(j['a'])
                w = int(self.beta_window_days * 24)
                m_e = le.rolling(w, min_periods=240).mean()
                m_a = la.rolling(w, min_periods=240).mean()
                cov = (le * la).rolling(w, min_periods=240).mean() - m_e * m_a
                var = (le * le).rolling(w, min_periods=240).mean() - m_e ** 2
                beta_h = (cov / var).rename('roll_beta').reset_index()
                beta_h['date'] = beta_h['date'] + pd.Timedelta(hours=1)  # causal: hour-h estimate usable from hour h+1
                dataframe = pd.merge_asof(dataframe.sort_values('date'), beta_h[['date', 'roll_beta']], on='date', direction='backward')
            except Exception as e:
                logger.error(f"Error computing roll_beta: {e}")

            if self.analyse_portfolio_spread:
                base_df = self.dp.get_pair_dataframe(pair=self.main_pairs[0], timeframe=self.timeframe)
                base_df['base_log'] = np.log(base_df['close'])
                
                spread_result = pd.DataFrame({'date': dataframe['date']})
                spread_result = pd.merge(spread_result, base_df[['date', 'base_log']], on='date', how='left')
                
                alt_log_values = []
                pairs = self.get_pairs()
                for pair in pairs:
                    if pair in self.main_pairs:
                        continue
                    alt_df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
                    merged_df = pd.merge(spread_result[['date']], alt_df[['date', 'close']], on='date', how='left')
                    alt_log_values.append(np.log(merged_df['close']))
                
                avg_alt_log = pd.concat(alt_log_values, axis=1).mean(axis=1, skipna=True)
                spread_result['spread'] = spread_result['base_log'] - avg_alt_log
                spread_result['spread'] = spread_result['spread'].ffill()
                
                dataframe = pd.merge(dataframe, spread_result[['date', 'spread']], on='date', how='left')
                
                dataframe['spread_avg'] = dataframe['spread'].rolling(window=self.spread_avg_window).mean()
                dataframe['spread_std'] = dataframe['spread'].rolling(window=self.spread_avg_window).std()
                dataframe['spread_zscore'] = (dataframe['spread'] - dataframe['spread_avg']) / dataframe['spread_std']

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

            if self.analyse_portfolio_spread:
                informative_up_mask = informative_up_mask & (dataframe['spread_zscore'] > -self.spread_zscore_threshold)
                informative_down_mask = informative_down_mask & (dataframe['spread_zscore'] < self.spread_zscore_threshold)
                    
            if metadata['pair'] in self.main_pairs:
                dataframe.loc[informative_up_mask, ['enter_long', 'enter_tag']] = (1, 'entry_long')
                dataframe.loc[informative_down_mask, ['enter_short', 'enter_tag']] = (1, 'entry_short')
            else:
                dataframe.loc[informative_up_mask, ['enter_short', 'enter_tag']] = (1, 'entry_short')
                dataframe.loc[informative_down_mask, ['enter_long', 'enter_tag']] = (1, 'entry_long')

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
            
            if metadata['pair'] in self.main_pairs:
                dataframe.loc[informative_down_mask, ['exit_long', 'exit_tag']] = (1, 'exit_ma')
                dataframe.loc[informative_up_mask, ['exit_short', 'exit_tag']] = (1, 'exit_ma')
            else:
                dataframe.loc[informative_down_mask, ['exit_short', 'exit_tag']] = (1, 'exit_ma')
                dataframe.loc[informative_up_mask, ['exit_long', 'exit_tag']] = (1, 'exit_ma')
                
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
            
        if not self._is_entry_time_valid(current_time):
            logger.info(f"Entry blocked: current time {current_time} is before next allowed entry time")
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
            self.portfolio_max_profit = -100000
            return
        
        total_profit_abs = sum(self.portfolio_profit_dict.values())
        if total_profit_abs > self.portfolio_max_profit:
            self.portfolio_max_profit = total_profit_abs
        
        if total_profit_abs < self.portfolio_min_profit:
            self.portfolio_min_profit = total_profit_abs
            logger.warning(f'Total portfolio profit reached min profit:{total_profit_abs:.2f} at {current_time}')
            
        if total_profit_abs <= self.portfolio_stoploss_amount:
            self.is_portfolio_exit = True
            self.portfolio_exit_reason = 'portfolio_stoploss'
            logger.info(f'Total portfolio profit reached stoploss:{total_profit_abs:.2f} <= {self.portfolio_stoploss_amount:.2f}, exiting portfolio at {current_time}')
            next_entry_time = current_time + timedelta(minutes=self.portfolio_cooldown_minutes)
            self._set_next_entry_time(next_entry_time)
            return
        
        trade_0 = open_trades[0]
        open_hours = round((current_time - trade_0.open_date_utc).total_seconds() / 3600, 1)
        time_decay = round(open_hours / self.take_profit_decay_factor, 1)
        if time_decay < 1:
            time_decay = 1
        take_profit = self.portfolio_take_profit_amount / time_decay
        
        if total_profit_abs >= take_profit:
            self.is_portfolio_exit = True
            self.portfolio_exit_reason = 'portfolio_take_profit'
            logger.info(f'Total portfolio profit reached take profit:{total_profit_abs:.2f} >= {take_profit:.2f}(time decay factor:{time_decay:.1f}), exiting portfolio at {current_time}')
            next_entry_time = current_time + timedelta(minutes=self.portfolio_cooldown_minutes)
            self._set_next_entry_time(next_entry_time)
            return
            
        if self.portfolio_trailing_stop_activation > 0 and self.portfolio_max_profit >= self.portfolio_trailing_stop_activation:
            trailing_stop_threshold = self.portfolio_max_profit * (1 - self.portfolio_trailing_stop_drawback_ratio)
            if total_profit_abs <= trailing_stop_threshold:
                self.is_portfolio_exit = True
                self.portfolio_exit_reason = 'portfolio_trailing_stop'
                logger.info(f'Total portfolio profit reached trailing stop, profit:{total_profit_abs:.2f} <= {trailing_stop_threshold:.2f}, '
                            f'max_profit:{self.portfolio_max_profit:.2f}, trailing stop ratio:{self.portfolio_trailing_stop_drawback_ratio:.2%}, exiting portfolio at {current_time}')
                next_entry_time = current_time + timedelta(minutes=self.portfolio_cooldown_minutes)
                self._set_next_entry_time(next_entry_time)
                return

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
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: float | None, max_stake: float, 
        leverage: float, entry_tag: str | None, side: str, 
        **kwargs,
    ) -> float:
        total_stake_amount_per_side = self.total_stake_amount / 2
        
        if pair in self.main_pairs:
            if len(self.main_pairs) == 1:
                stake_amount = total_stake_amount_per_side
            else:
                if pair == self.main_pairs[0]:
                    stake_amount = (total_stake_amount_per_side / 2)
                else:
                    stake_amount = (total_stake_amount_per_side / 2) / (len(self.main_pairs) - 1)
            # logger.info(f"Fixed stake for main pair {pair}: {stake_amount:.2f} at {current_time}")
        else:
            if self.enable_dynamic_stake:
                stake_amount = self.calc_stake_based_on_beta(pair, current_time, current_rate, proposed_stake, min_stake, max_stake, leverage, entry_tag, side, **kwargs)
            else:
                stake_amount = total_stake_amount_per_side / (len(self.get_pairs()) - len(self.main_pairs))
                # logger.info(f"Fixed stake for pair {pair}: {stake_amount:.2f} at {current_time}")
        
        return stake_amount

    def _last_roll_beta(self, pair: str) -> float | None:
        """Read roll_beta (computed in populate_indicators) from the latest candle.

        Uses the analyzed dataframe's own column because get_analyzed_dataframe
        only returns ~1000 candles in backtest — too short to recompute a 90d β.
        """
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is not None and not df.empty and 'roll_beta' in df.columns:
                v = float(df['roll_beta'].iloc[-1])
                return v if np.isfinite(v) else None
        except Exception as exc:  # pragma: no cover - defensive for live data gaps
            logger.warning(f"roll_beta read failed for {pair}: {exc}")
        return None

    def calc_stake_based_on_beta(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: float | None, max_stake: float,
        leverage: float, entry_tag: str | None, side: str,
        **kwargs,
    ) -> float:
        """Alt legs sized proportional to 1/beta: equal hedge sensitivity per alt.

        Pairs with beta < 0.3 (or unestimable) get zero weight; if no pair has
        usable beta, fall back to equal split.
        """
        pairs = self.get_pairs()
        alt_pairs = [p for p in pairs if p not in self.main_pairs]
        total_alt_stake = self.total_stake_amount / 2

        betas = {}
        for alt in alt_pairs:
            betas[alt] = self._last_roll_beta(alt)
        if os.environ.get('LS3_BETA_DEBUG'):
            logger.warning(f"BETADEBUG wd={self.beta_window_days} clip={self.beta_clip_lo}-{self.beta_clip_hi} betas={ {a: (round(b,3) if b is not None else None) for a, b in betas.items()} }")
        # usable beta -> inv-weight; unestimable/degenerate -> neutral weight 1.0
        inv = {}
        for a, b in betas.items():
            inv[a] = 1.0 / min(max(b, self.beta_clip_lo), self.beta_clip_hi) if (b is not None and b >= 0.3) else 1.0
        s = sum(inv.values())
        alt_stake = total_alt_stake * inv[pair] / s

        if max_stake is not None:
            alt_stake = min(alt_stake, max_stake)
        if min_stake is not None:
            alt_stake = max(alt_stake, min_stake)
        if alt_stake * self.trade_leverage < self.min_notional:
            alt_stake = self.min_notional / self.trade_leverage
        return alt_stake

    def calc_stake_based_on_atr(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: float | None, max_stake: float, 
        leverage: float, entry_tag: str | None, side: str, 
        **kwargs,
    ) -> float:
        pairs = self.get_pairs()
        alt_pairs = [p for p in pairs if p not in self.main_pairs]
        num_alts = len(alt_pairs)
        
        total_alt_stake = self.total_stake_amount / 2
        
        raw_weights = {}
        total_raw_weight = 0.0
        
        for alt in alt_pairs:
            df, _ = self.dp.get_analyzed_dataframe(alt, self.timeframe)
            if df is not None and not df.empty:
                last_candle = df.iloc[-1].squeeze()
                natr_val = last_candle.get('natr', 0) 
                if pd.notna(natr_val) and natr_val > 0:
                    weight = 1.0 / natr_val
                else:
                    weight = 0.0
            else:
                weight = 0.0
                
            raw_weights[alt] = weight
            total_raw_weight += weight
            
        if total_raw_weight == 0:
            return total_alt_stake / num_alts
        
        if pair in raw_weights:
            normalized_weight = raw_weights[pair] / total_raw_weight
            
            alt_stake = total_alt_stake * normalized_weight
            
            if max_stake is not None:
                alt_stake = min(alt_stake, max_stake)
            if min_stake is not None:
                alt_stake = max(alt_stake, min_stake)
                
            if alt_stake * self.trade_leverage < self.min_notional:
                alt_stake = self.min_notional / self.trade_leverage
                
            # logger.info(f"Dynamic stake for {pair}: {alt_stake:.2f} (weight: {normalized_weight:.2%}) at {current_time}")
            return alt_stake
        else:
            return 0.0
    
    def get_pairs(self):
        return self.config['exchange']['pair_whitelist']
