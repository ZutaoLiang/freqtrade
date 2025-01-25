import numpy as np
import pandas_ta as pta
import pandas as pd
from pandas import DataFrame
from functools import reduce

from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

from freqtrade.strategy.interface import IStrategy, Trade, Order
from freqtrade.strategy.strategy_helper import stoploss_from_absolute, stoploss_from_open
from freqtrade.strategy import IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib

import logging
logger = logging.getLogger(__name__)


class ReversalScalpingV1Strategy(IStrategy):
    
    minimal_roi = {"0": 100}
    can_short = True
    is_long = True
    
    timeframe = '3m'
    trade_leverage = 5
    
    base_stoploss_pct = 0.03
    stoploss = -base_stoploss_pct * trade_leverage
    trailing_stop = False
    use_custom_stoploss = True

    period = 10
    
    ema_length = period
    ema_mid_length = 9 * period
    ema_long_length = 36 * period
    ema_trend = 6
    ema_mid_trend = ema_trend
    ema_long_trend = ema_trend * 3
    ema_dist_ratio = 1.02
    
    breakout_period = 4
    
    adx_length = period
    adx_threshold = 40
    rsi_length = period
    rsi_long_threshold = 55
    rsi_short_threshold = 30
    atr_length = int(1.5 * period)
    
    startup_candle_count = int(ema_long_length)
    
    stake_ratio = 0.25
    exit_loss_ratio = -0.25
    
    position_adjustment_enable = False
    
    MAX_PROFIT_ABS = 'max_profit_abs'
    HIGH_PROFIT = 'high_profit'

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return self.trade_leverage
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        leverage = trade.leverage
        is_short = trade.is_short
        entry_stake = self.calc_entry_stake_without_leverage()
        entry_stake_with_leverage = entry_stake * leverage
        stake_amount = trade.amount * trade.open_rate
        
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        diff_pct = 0.002
        if is_short:
            in_trend = last_candle['ha_close'] < last_candle['ha_open'] and current_rate < last_candle['ha_close'] * (1+diff_pct)
        else:
            in_trend = last_candle['ha_close'] > last_candle['ha_open'] and current_rate > last_candle['ha_close'] * (1-diff_pct)
 
        low_stake_threshold_array = [2.5 * entry_stake_with_leverage]
        holding_minutes_array = [120]
        for index, (low_stake_threshold, holding_minutes) in enumerate(zip(low_stake_threshold_array, holding_minutes_array)):
            is_low_stake = stake_amount < low_stake_threshold
            if is_low_stake:
                if (current_time - timedelta(minutes=holding_minutes)) > trade.open_date_utc and 0.005 * leverage < current_profit < 0.015 * leverage \
                    and not in_trend:
                    exit_reason = f'Long time low profit-{index+1}'
                    logger.info(f'{exit_reason} for pair:{trade.pair}, current_rate:{current_rate:.6f}, open_rate:{trade.open_rate:.6f}, current_profit:{current_profit:.2%}, stake_amount:{stake_amount:.4f} at {current_time}')
                    return exit_reason
                
        open_profit_abs = current_profit / leverage * stake_amount
        realized_profit_abs = trade.realized_profit if trade.realized_profit else 0
        total_profit_abs = realized_profit_abs + open_profit_abs
        
        max_profit_abs = trade.get_custom_data(self.MAX_PROFIT_ABS)
        if max_profit_abs is None:
            max_profit_abs = total_profit_abs
            trade.set_custom_data(self.MAX_PROFIT_ABS, max_profit_abs)
        elif total_profit_abs > max_profit_abs:
            logger.info(f'New max profit for {trade.pair}, from {max_profit_abs:.4f} to {total_profit_abs:.4f}, current_rate:{current_rate:.6f} at {current_time}')
            max_profit_abs = total_profit_abs
            trade.set_custom_data(self.MAX_PROFIT_ABS, max_profit_abs)
        
        logger.info(f'{trade.pair} total profit:{total_profit_abs:.4f}(open:{open_profit_abs:.4f}, close:{realized_profit_abs:.4f}), current_rate:{current_rate:.6f}, open_rate:{trade.open_rate:.6f}, current_profit:{current_profit:.2%}, stake_amount:{stake_amount:.4f} at {current_time}')

        market_value_threshold_array = [entry_stake * 0.5, entry_stake * 0.3]
        draw_back_ratio_array = [0.7, 0.5]

        for index, (market_value_threshold, draw_back_ratio) in enumerate(zip(market_value_threshold_array, draw_back_ratio_array)):
            reach_profit = max_profit_abs > market_value_threshold
            if index == 0:
                trade.set_custom_data(self.HIGH_PROFIT, reach_profit)
                
            reach_drawback = reach_profit and total_profit_abs < max_profit_abs * draw_back_ratio
            logger.info(f'Checking {trade.pair} drawback result:{reach_drawback} on threshold #{index+1}:{market_value_threshold:.4f}(market_value) and total_profit_abs:{total_profit_abs:.4f} vs {max_profit_abs*draw_back_ratio:.4f}(max_profit_abs:{max_profit_abs:.4f}*ratio:{draw_back_ratio:.2%}) at {current_time}')
            if reach_drawback:
                exit_reason = f'Profit drawback-{index+1}'
                logger.info(f'{exit_reason} for {pair}: total profit {total_profit_abs:.4f} < (max_profit_abs {max_profit_abs:.4f} * {draw_back_ratio:.2%}), current_rate:{current_rate:.4f} at {current_time}')
                return exit_reason

        profit_drawdown_threshold = entry_stake_with_leverage * self.exit_loss_ratio
        reach_max_loss = total_profit_abs < profit_drawdown_threshold
        logger.info(f'Checking {trade.pair} max loss result:{reach_max_loss}, total_profit_abs:{total_profit_abs:.4f}, threshold:{profit_drawdown_threshold:.4f}=(entry_stake_with_leverage:{entry_stake_with_leverage:.4f}*exit_loss_ratio:{self.exit_loss_ratio:.2%}), current_profit:{current_profit:.2%} at {current_time}')
        if reach_max_loss:
            exit_reason = 'Max loss'
            logger.info(f'{exit_reason} for {pair}:{total_profit_abs:.4f} < {profit_drawdown_threshold:.4f}, current_rate:{current_rate:.6f} at {current_time}')
            return exit_reason
        
        return False

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        leverage = trade.leverage
        is_short = trade.is_short
        _current_profit = current_profit / leverage

        if after_fill:
            filled_orders = trade.select_filled_orders()
            count_of_orders = len(filled_orders)
            if count_of_orders <= 1:
                return self.stoploss
            
            logger.info(f'Should not happen when adjust_postion is disabled')
            # 存在加仓时的处理
            if _current_profit < 0.015:
                relative_factor = 0.005
            elif _current_profit < 0.03:
                relative_factor = 0.0075
            elif _current_profit < 0.06:
                relative_factor = 0.01
            elif _current_profit < 0.06:
                relative_factor = 0.02
            elif _current_profit < 0.06:
                relative_factor = 0.03
            else:
                return 0.04 * leverage
                
            open_rate = trade.open_rate
            if is_short:
                relative_factor *= -1
                
            stop_rate = trade.open_rate*(1+relative_factor)
            
            logger.info(f'Setting {trade.pair} #{count_of_orders} after fill stoploss rate to:{stop_rate:.6f}, open_rate:{open_rate:.6f}(stop/open ratio:{abs(stop_rate/open_rate-1):.2%}), current_rate:{current_rate:.6f}, current_profit:{current_profit:.2%}(without leverage:{current_profit/leverage:.2%}) at {current_time}')
            
            return stoploss_from_absolute(stop_rate, current_rate, is_short, leverage)
        
        if _current_profit < 0.015:
            return None
        
        if _current_profit < 0.025:
            relative_factor = 0.0075
        elif _current_profit < 0.035:
            relative_factor = 0.015
        elif _current_profit > 0.05:
            return 0.2 * leverage   # 当前利润较高，按照最高利润价跌20%来止损
        else:
            relative_factor = _current_profit * 0.5
            
        open_rate = trade.open_rate
        if is_short:
            relative_factor *= -1
            
        stop_rate = trade.open_rate*(1+relative_factor)
        
        logger.info(f'Setting {trade.pair} stoploss rate to:{stop_rate:.6f}, open_rate:{open_rate:.6f}(stop/open ratio:{abs(stop_rate/open_rate-1):.2%}), current_rate:{current_rate:.6f}, current_profit:{current_profit:.2%}(without leverage:{current_profit/leverage:.2%}) at {current_time}')
        
        return stoploss_from_absolute(stop_rate, current_rate, is_short, leverage)

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float, min_stake: float, max_stake: float, 
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        # Note: 这个函数需要返回的是不带杠杆的金额，具体代码参考freqtradebot.execute_entry()
        if not self.position_adjustment_enable:
            return None
        
        return None

    def calc_entry_stake_without_leverage(self) -> float:
        return self.wallets.get_total_stake_amount() * self.stake_ratio / self.max_open_trades
        
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        # Note: 这个函数需要返回的是不带杠杆的金额，具体代码参考freqtradebot.execute_entry()
        stake_amount = min(max(min_stake, proposed_stake * self.stake_ratio), max_stake)
        logger.info(f'Stake amount for {pair}={stake_amount:.5f} with leverage:{leverage}(after leverage={stake_amount*leverage:.5f}), proposed:{proposed_stake:.5f}, min_stake:{min_stake:.5f}, max_stake:{max_stake:.5f}, rate:{current_rate:.5f} at {current_time}')
        return stake_amount

    def heikinashi(self, dataframe: DataFrame) -> DataFrame:
        # ha = qtpylib.heikinashi(dataframe)
        ha = dataframe
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']
        return dataframe
       
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.heikinashi(dataframe)
        
        dataframe['ema'] = pta.ema(close=dataframe['ha_close'], length=self.ema_length, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['ha_close'], length=self.ema_mid_length, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ha_close'], length=self.ema_long_length, talib=False)
        dataframe['recent_high'] = dataframe['ha_close'].rolling(window=self.breakout_period).max()
        dataframe['recent_low'] = dataframe['ha_close'].rolling(window=self.breakout_period).min()
        dataframe['adx'] = pta.adx(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.adx_length)[f'ADX_{self.adx_length}']
        dataframe['rsi'] = pta.rsi(dataframe['ha_close'], length=self.rsi_length, talib=False)
        dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_length)
         
        return dataframe
        
    def ema_up_n_days_mask(self, dataframe: DataFrame, ema: str, days: int):
        ema_up_mask = (dataframe[f'{ema}'] > dataframe[f'{ema}'].shift(1))
        for i in range(2, days):
            ema_up_mask = ema_up_mask & (dataframe[f'{ema}'].shift(i-1) > dataframe[f'{ema}'].shift(i))
        return ema_up_mask
    
    def ema_down_n_days_mask(self, dataframe: DataFrame, ema: str, days: int):
        ema_down_mask = (dataframe[f'{ema}'] < dataframe[f'{ema}'].shift(1))
        for i in range(2, days):
            ema_down_mask = ema_down_mask & (dataframe[f'{ema}'].shift(i-1) < dataframe[f'{ema}'].shift(i))
        return ema_down_mask
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['general_trend'] = 0
        dataframe['strong_trend'] = 0
        dataframe['reversal'] = 0
        
        if self.is_long:
            ema_down_mask = self.ema_down_n_days_mask(dataframe, 'ema', self.ema_trend)
            ema_mid_down_mask = self.ema_down_n_days_mask(dataframe, 'ema_mid', self.ema_mid_trend)
            ema_long_down_mask = self.ema_down_n_days_mask(dataframe, 'ema_long', self.ema_long_trend)
            
            general_trend_mask = (dataframe['ha_close'] * self.ema_dist_ratio < dataframe['ema_long']) \
                        & (dataframe['ha_close'] < dataframe['ema']) \
                        & (ema_long_down_mask) \
                        & (dataframe['ema_mid'] < dataframe['ema_long'])
            
            dataframe.loc[general_trend_mask, 'general_trend'] = 1
            
            dataframe.loc[(
                        general_trend_mask
                        & (ema_down_mask)
                        & (ema_mid_down_mask)
                        & (dataframe['ema'] < dataframe['ema_mid'])
                        & (dataframe['ha_close'] < dataframe['recent_low'].shift(1))
                        & (dataframe['adx'] < self.adx_threshold)
                        # (dataframe['rsi'] < self.rsi_short_threshold) & 
                    ), 'strong_trend'] = 1
            
            dataframe.loc[
                    (
                        (dataframe['strong_trend'] == 0)
                        & (dataframe['strong_trend'].shift(1) == 1)
                        & (dataframe['ha_close'] > dataframe['ha_open'])
                        & (dataframe['ha_close'] > dataframe['ha_close'].shift(1))
                    ),
                    ['enter_long', 'enter_tag']] = (1, 'entry')
        else:
            ema_up_mask = self.ema_up_n_days_mask(dataframe, 'ema', self.ema_trend)
            ema_mid_up_mask = self.ema_up_n_days_mask(dataframe, 'ema_mid', self.ema_mid_trend)
            ema_long_up_mask = self.ema_up_n_days_mask(dataframe, 'ema_long', self.ema_long_trend)
        
            general_trend_mask = (dataframe['ha_close'] > self.ema_dist_ratio * dataframe['ema_long']) \
                                    & (dataframe['ha_close'] > dataframe['ema']) \
                                    & (ema_long_up_mask) \
                                    & (dataframe['ema_mid'] > dataframe['ema_long'])
                                    
            dataframe.loc[general_trend_mask, 'general_trend'] = 1
            
            dataframe.loc[
                (
                    general_trend_mask
                    & (ema_up_mask)
                    & (ema_mid_up_mask)
                    & (dataframe['ema'] > dataframe['ema_mid'])
                    & (dataframe['ha_close'] > dataframe['recent_high'].shift(1))
                    & (dataframe['adx'] < self.adx_threshold)
                    # (dataframe['rsi'] > self.rsi_long_threshold) & 
                ), 'strong_trend'] = 1
            
            dataframe.loc[
                    (
                        (dataframe['strong_trend'] == 0)
                        & (dataframe['strong_trend'].shift(1) == 1)
                        & (dataframe['ha_close'] < dataframe['ha_open'])
                        & (dataframe['ha_close'] < dataframe['ha_close'].shift(1))
                    ),
                    ['enter_short', 'enter_tag']] = (1, 'entry')
            
        return dataframe
        
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        dataframe['reversal_signal'] = 0
        
        if self.is_long:
            ema_mid_down_mask = self.ema_down_n_days_mask(dataframe, 'ema_mid', self.ema_mid_trend)
            dataframe.loc[
                    (
                        (
                            (dataframe['ha_close'] < dataframe['ema_mid']) | (dataframe['ha_close'] < dataframe['ema_long'])
                        )
                        & (ema_mid_down_mask)
                        & (dataframe['ha_close'] < self.ema_dist_ratio * dataframe['ema_long'])
                    ), 'reversal_signal'] = 1
        else:
            ema_mid_up_mask = self.ema_up_n_days_mask(dataframe, 'ema_mid', self.ema_mid_trend)
            dataframe.loc[
                    (
                        (
                            (dataframe['ha_close'] > dataframe['ema_mid']) | (dataframe['ha_close'] > dataframe['ema_long'])
                        )
                        & (ema_mid_up_mask)
                        & (dataframe['ha_close'] > self.ema_dist_ratio * dataframe['ema_long'])
                    ), 'reversal_signal'] = 1
            
        return dataframe
