import math
import numpy as np
import pandas_ta as pta
import pandas as pd
from pandas import DataFrame
from functools import reduce

from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

from freqtrade.strategy.interface import IStrategy, Trade
from freqtrade.strategy.strategy_helper import stoploss_from_absolute, stoploss_from_open
from freqtrade.strategy import IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib

import logging
logger = logging.getLogger(__name__)


class StakePositionManager(IStrategy):
    """Base class for stake and position management"""
        
    # 常量
    MAX_PROFIT_ABS = 'max_profit_abs'
    HIGH_PROFIT = 'high_profit'
    
    # 官方变量，不常改
    minimal_roi = {"0": 100}
    trailing_stop = False
    use_custom_stoploss = True
    can_short = True
    position_adjustment_enable = False
    
    # 自定义变量，不常改
    enable_heikinashi = True

    # 自定义变量，可微调
    timeframe = '3m'
    trade_leverage = 10
    base_stoploss_pct = 0.09
    stoploss = -base_stoploss_pct * trade_leverage
    entry_stake_ratio = 0.2
    addition_stake_ratio = 1
    exit_loss_ratio = -0.2
    atr_length = 15
    atr_entry_stoploss_multiplier = 5               # 进场时基于open_rate的止损ATR倍数
    atr_entry_base_multiplier = 3.0                 # 进场时的价格ATR倍数，即当前价格超过成本价的这个ATR倍数进场
    atr_addition_base_multiplier = 2.5              # 加仓时的价格ATR倍数，即当前价格超过成本价的这个ATR倍数加仓
    atr_addition_stoploss_base_multiplier = 0.25    # 加仓后的价格ATR止损倍数，即加仓后不足成本价的这个ATR倍数止损


    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return self.trade_leverage
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        return False
    
    def atr_addition_multiplier(self, count_of_orders: int) -> float:
        return self.atr_addition_base_multiplier + (count_of_orders-1)

    def atr_addition_stoploss_multiplier(self, count_of_orders: int) -> float:
        return self.atr_addition_stoploss_base_multiplier + (count_of_orders-1) / 10

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, after_fill: bool,
                       **kwargs) -> Optional[float]:
        leverage = trade.leverage
        is_short = trade.is_short
        factor = -1 if is_short else 1
        open_rate = trade.open_rate
        _current_profit = current_profit / leverage

        if _current_profit < 0:
            return None

        if _current_profit > 0.30:
            return (_current_profit / 2) * leverage

        if _current_profit > 0.25:
            return stoploss_from_absolute(open_rate*(1+factor*0.15), current_rate, is_short, leverage)

        if _current_profit > 0.18:
            return stoploss_from_absolute(open_rate*(1+factor*0.08), current_rate, is_short, leverage)
        
        if _current_profit > 0.10:
            return stoploss_from_absolute(open_rate*(1+factor*0.05), current_rate, is_short, leverage)

        if _current_profit > 0.06:
            return stoploss_from_absolute(open_rate*(1+factor*0.01), current_rate, is_short, leverage)

        # if _current_profit > 0.02:
        #     return stoploss_from_absolute(open_rate*(1+factor*0.001), current_rate, is_short, leverage)
        
        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float, min_stake: float, max_stake: float, 
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        if not self.position_adjustment_enable:
            return None
        
        has_open_orders = any(order.status == "open" and not order.ft_is_open for order in trade.orders)
        if has_open_orders:
            logger.info(f'There are open orders for {trade.pair}, skip position adjustment.')
            return None
        
        filled_orders = trade.select_filled_orders()
        count_of_orders = len(filled_orders)
        if count_of_orders == 0:
            return None
        
        is_short = trade.is_short
        leverage = trade.leverage
        open_rate = trade.open_rate
        _current_profit = current_profit / leverage
        
        entry_side_orders = [order for order in filled_orders
                             if order.ft_order_side == trade.entry_side and ('entry' in order.ft_order_tag)]
        count_of_orders = len(entry_side_orders)

        addition_stake = self.get_entry_stake_without_leverage() * leverage * self.addition_stake_ratio
        min_stake_threshold = 0.75 * min_stake
        if addition_stake < min_stake_threshold:
            logger.info(f'Skip position addition for {trade.pair} while stake amount:{addition_stake:.5f} is smaller than threshold:{min_stake_threshold:.5f}(min_stake:{min_stake:.5f}) at {current_time}')
            return None

        addition_amount = round(addition_stake / current_rate)
        addition_stake = addition_amount * current_rate
        if min_stake_threshold < addition_stake < min_stake:
            logger.info(f'Adjusting {trade.pair} addition stake:{addition_stake:.5f} to around min_stake:{min_stake:.5f} at {current_time}')
            addition_amount = math.ceil(min_stake / current_rate)
            addition_stake = addition_amount * current_rate
        
        new_open_rate = (trade.amount * trade.open_rate + addition_stake) / (trade.amount + addition_amount)
        
        factor = -1 if is_short else 1
        new_open_profit = factor * (current_rate / new_open_rate - 1)

        last_candle = self.get_last_candle(trade)
        atr = last_candle['atr']

        addition_signal = False
        if is_short and last_candle['enter_short'] == 1 and new_open_profit > (0.1 + 0.05 * count_of_orders):
            addition_signal = True
        elif not is_short and last_candle['enter_long'] == 1 and new_open_profit > (0.1 + 0.05 * count_of_orders):
            addition_signal = True

        if addition_signal:
            logger.info(f'Initialize {trade.pair} addition stake to {addition_stake:.5f}(open rate:{open_rate:.6f}, [new_open_rate:{new_open_rate:.6f}], atr:{atr:.6f}, addition amount:{addition_amount:.2f}) '
                    f'at current_rate:{current_rate:.5f} with profit:{current_profit:.2%}({_current_profit:.2%}) at {current_time}')
            
            logger.info(f'Position addition for {trade.pair} with stake amount {addition_stake:.5f} triggered at addition signal, current_profit:{current_profit:.2f}, current_rate:{current_rate:.5f} at {current_time}')
            return (addition_stake / leverage, f'entry-addition')
        
        return None
        
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        stake_amount = min(proposed_stake * self.entry_stake_ratio, max_stake)
        logger.info(f'Stake amount for {pair}={stake_amount:.5f} with leverage:{leverage}(after leverage={stake_amount*leverage:.5f}), '
                    f'proposed:{proposed_stake:.5f}, min_stake:{min_stake:.5f}, max_stake:{max_stake:.5f}, current_rate:{current_rate:.5f} at {current_time}')
        return stake_amount

    def get_last_candle(self, trade: Trade):
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        return last_candle
        
    def get_entry_stake_without_leverage(self) -> float:
        return self.wallets.get_total_stake_amount() * self.entry_stake_ratio / self.max_open_trades
    
    def heikinashi(self, dataframe: DataFrame) -> DataFrame:
        if self.enable_heikinashi:
            ha = qtpylib.heikinashi(dataframe)
        else:
            ha = dataframe
            
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.heikinashi(dataframe)
        dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_length, talib=False)
        dataframe['addition_plus_atr'] = dataframe['ha_close'] + (self.atr_entry_base_multiplier * dataframe['atr'])
        dataframe['addition_minus_atr'] = dataframe['ha_close'] - (self.atr_entry_base_multiplier * dataframe['atr'])
        dataframe['stoploss_plus_atr'] = dataframe['ha_close'] + (self.atr_entry_stoploss_multiplier * dataframe['atr'])
        dataframe['stoploss_minus_atr'] = dataframe['ha_close'] - (self.atr_entry_stoploss_multiplier * dataframe['atr'])
        dataframe['atr_pct'] = 100 * dataframe['atr'] / dataframe['ha_close']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['addition'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
    
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
    

class Dream3V2ManualStrategy(StakePositionManager):
    """Trading strategy implementation"""
    
    is_long = False
    bidirectional = True
    
    # Strategy parameters
    period = 10
    ema_length = period
    ema_mid_length = 6 * period
    ema_long_length = 12 * period
    ema_trend = 4
    ema_mid_trend = ema_trend
    ema_long_trend = ema_trend
    
    breakout_period = 4
    
    rumi_multiplier = 150

    adx_length = period
    adx_threshold = 32
    rsi_length = period
    rsi_long_threshold = 55
    rsi_short_threshold = 30
    
    startup_candle_count = int(ema_long_length)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        
        dataframe['ema'] = pta.ema(close=dataframe['ha_close'], length=self.ema_length, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['ha_close'], length=self.ema_mid_length, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ha_close'], length=self.ema_long_length, talib=False)
        dataframe['ema_long_plus_atr'] = dataframe['ema_long'] + self.atr_addition_base_multiplier * dataframe['atr']
        dataframe['ema_long_minus_atr'] = dataframe['ema_long'] - self.atr_addition_base_multiplier * dataframe['atr']
        dataframe['recent_high'] = dataframe['ha_close'].rolling(window=self.breakout_period).max()
        dataframe['recent_low'] = dataframe['ha_close'].rolling(window=self.breakout_period).min()
        dataframe['adx'] = pta.adx(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.adx_length)[f'ADX_{self.adx_length}']
        dataframe['rsi'] = pta.rsi(dataframe['ha_close'], length=self.rsi_length, talib=False)
        
        dataframe['rumi_fast'] = pta.sma(dataframe['ha_close'], length=self.ema_length)
        dataframe['rumi_slow'] = pta.wma(dataframe['ha_close'], length=self.ema_mid_length)
        dataframe['rumi'] = pta.sma(dataframe['rumi_fast'] - dataframe['rumi_slow'], length=self.ema_length)

        dataframe['rumi_long_slow'] = pta.wma(dataframe['ha_close'], length=self.ema_long_length)
        dataframe['rumi_long'] = pta.sma(dataframe['rumi_fast'] - dataframe['rumi_long_slow'], length=self.ema_length)
        return dataframe
        

class Dream3V2Strategy(Dream3V2ManualStrategy):
        
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        
        if self.bidirectional or self.is_long:
            ema_up_mask = self.indicator_up_n_periods_mask(dataframe, 'ema', self.ema_trend)
            ema_mid_up_mask = self.indicator_up_n_periods_mask(dataframe, 'ema_mid', self.ema_mid_trend)
            # ema_long_up_mask = self.indicator_up_n_periods_mask(dataframe, 'ema_long', self.ema_long_trend)
            rumi_up_mask = self.indicator_up_n_periods_mask(dataframe, 'rumi', self.ema_trend)
        
            addition_trend_mask = (dataframe['ha_close'] > dataframe['ema_long_plus_atr']) \
                                    & (dataframe['ha_close'] > dataframe['ema']) \
                                    & (ema_mid_up_mask)
                                    
            dataframe.loc[addition_trend_mask, 'addition'] = 1
            
            dataframe.loc[
                    (
                        addition_trend_mask
                        & (ema_up_mask)
                        # & (ema_mid_up_mask)
                        & (rumi_up_mask)
                        & (dataframe['ema'] > dataframe['ema_mid'])
                        & (dataframe['ha_close'] > dataframe['recent_high'].shift(1))
                        & (dataframe['rumi'] * self.rumi_multiplier > dataframe['ha_close'])
                    ),
                    ['enter_long', 'enter_tag']] = (1, 'entry_long')
        
        if self.bidirectional or not self.is_long:
            ema_down_mask = self.indicator_down_n_periods_mask(dataframe, 'ema', self.ema_trend)
            ema_mid_down_mask = self.indicator_down_n_periods_mask(dataframe, 'ema_mid', self.ema_mid_trend)
            # ema_long_down_mask = self.indicator_down_n_periods_mask(dataframe, 'ema_long', self.ema_long_trend)
            rumi_down_mask = self.indicator_down_n_periods_mask(dataframe, 'rumi', self.ema_trend)
            
            addition_trend_mask = (dataframe['ha_close'] < dataframe['ema_long_minus_atr']) \
                        & (dataframe['ha_close'] < dataframe['ema']) \
                        & (ema_mid_down_mask)
            
            dataframe.loc[addition_trend_mask, 'addition'] = 1
            
            dataframe.loc[
                    (
                        addition_trend_mask 
                        & (ema_down_mask) 
                        # & (ema_mid_down_mask) 
                        & (rumi_down_mask)
                        & (dataframe['ema'] < dataframe['ema_mid']) 
                        & (dataframe['ha_close'] < dataframe['recent_low'].shift(1))
                        & (dataframe['rumi'] * -self.rumi_multiplier > dataframe['ha_close'])
                    ),
                    ['enter_short', 'enter_tag']] = (1, 'entry_short')
            
        return dataframe
        
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_exit_trend(dataframe, metadata)
        return dataframe
