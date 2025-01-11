from math import isnan
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


class BreakoutScalping1dStrategy(IStrategy):
    
    # common
    minimal_roi = {"0": 100}
    
    trade_leverage = 2

    timeframe = '1d'
    
    stoploss = -0.4 * trade_leverage
    trailing_stop = False
    use_custom_stoploss = True

    can_short = True
    
    atr_length = 14
    atr_multiplier = 1.5
    
    # strategy
    short_period = 14
    mid_period = 30
    long_period = 60

    adx_length = short_period
    adx_threshold = 30
    
    rsi_length = short_period
    rsi_long_threshold = 65
    rsi_short_threshold = 35
    
    ema_short_length = short_period
    ema_mid_length = mid_period
    ema_long_length = long_period
    ema_large_length = 120
    
    breakout_long_length = 60
    breakout_short_length = breakout_long_length
    
    large_up_ratio = 1.001
    
    startup_candle_count = max(breakout_long_length, breakout_short_length, ema_large_length)

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        # if after_fill:
        #     stop_loss_after_fill = 0.1 * self.trade_leverage
        #     logger.info(f'Set stop loss after fill:{stop_loss_after_fill}')
        #     return stoploss_from_open(stop_loss_after_fill, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        
        if current_profit > 0.4 * trade.leverage:
            return 0.2 * trade.leverage
        # elif current_profit > 0.3 * self.trade_leverage:
        #     return stoploss_from_open(0.1 * self.trade_leverage, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        # elif current_profit > 0.2 * self.trade_leverage:
        #     return stoploss_from_open(0.05 * self.trade_leverage, current_profit, is_short=trade.is_short, leverage=trade.leverage)
            # return stoploss_from_open(0.2 * self.trade_leverage * 0.6, current_profit, is_short=trade.is_short, leverage=trade.leverage)            
        # elif current_profit > 0.12 * self.trade_leverage:
        #     return stoploss_from_open(0.12 * self.trade_leverage * 0.5, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        # elif current_profit > 0.06 * self.trade_leverage:
            # return stoploss_from_open(0.06 * self.trade_leverage * 0.3, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        # elif current_profit > 0.1 * self.trade_leverage:
        #     return stoploss_from_open(0.1 * self.trade_leverage * 0.5, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        # elif current_profit > 0.01 * self.trade_leverage:
            # return stoploss_from_open(0.01 * self.trade_leverage * 0.1, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        # elif current_profit < -0.01 * self.trade_leverage:
        #     return stoploss_from_open(0.005 * self.trade_leverage * stepped_ratio, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        
        # 利润较低，例如不超过5%时，则如果持仓时间长，变成动态止损（始终根据当前价格来计算）
        # if (current_time - timedelta(days=5)) > trade.open_date_utc:
            # return stoploss_from_open(0.01 * self.trade_leverage, current_profit, is_short=trade.is_short, leverage=trade.leverage)
            # return 0.02 * self.trade_leverage
        
        return None
 
    def count_breakout(self, row_idx, prices, is_long: bool):
        if row_idx == 0:
            return 0
            
        current_price = prices.iloc[row_idx]
        count = 0
        
        for i in range(row_idx-1, -1, -1):
            if (is_long and current_price <= prices.iloc[i]) or (not is_long and current_price >= prices.iloc[i]):
                break
            count += 1
        
        return count
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['breakout_long'] = dataframe.index.map(
            lambda x: self.count_breakout(x, dataframe['close'], True)
        )
        
        dataframe['breakout_short'] = dataframe.index.map(
            lambda x: self.count_breakout(x, dataframe['close'], False)
        )
        
        dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_length)
        
        dataframe['ema_short'] = pta.ema(close=dataframe['close'], length=self.ema_short_length, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['close'], length=self.ema_mid_length, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['close'], length=self.ema_long_length, talib=False)
        dataframe['ema_large'] = pta.ema(close=dataframe['close'], length=self.ema_large_length, talib=False)
 
        adx = pta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=self.adx_length)
        dataframe['adx'] = adx[f'ADX_{self.adx_length}']
        
        dataframe['rsi'] = pta.rsi(dataframe['close'], length=self.rsi_length, talib=False)
        
        return dataframe
        
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        long_entry_conditions = []
        long_entry_conditions.append(dataframe['breakout_long'] >= self.breakout_long_length)
        long_entry_conditions.append(dataframe['close'] >= dataframe['ema_short'])
        long_entry_conditions.append(dataframe['ema_short'] >= dataframe['ema_mid'])
        long_entry_conditions.append(dataframe['ema_mid'] >= dataframe['ema_long'])
        long_entry_conditions.append(dataframe['ema_short'] >= dataframe['ema_short'].shift(1))
        long_entry_conditions.append(dataframe['ema_mid'] >= dataframe['ema_mid'].shift(1))
        long_entry_conditions.append(dataframe['ema_long'] >= dataframe['ema_long'].shift(1))
        
        long_entry_conditions.append(dataframe['ema_large'] >= dataframe['ema_large'].shift(1) * self.large_up_ratio)
        
        long_entry_conditions.append(dataframe['adx'] >= self.adx_threshold)
        long_entry_conditions.append(dataframe['rsi'] >= self.rsi_long_threshold)
        
        dataframe.loc[
        (
            (reduce(lambda x, y: x & y, long_entry_conditions))
        )
        , 'enter_long'] = 1
        
        short_entry_conditions = []
        short_entry_conditions.append(dataframe['breakout_short'] >= self.breakout_long_length)
        short_entry_conditions.append(dataframe['close'] <= dataframe['ema_short'])
        short_entry_conditions.append(dataframe['ema_short'] <= dataframe['ema_mid'])
        short_entry_conditions.append(dataframe['ema_mid'] <= dataframe['ema_long'])
        short_entry_conditions.append(dataframe['ema_short'] <= dataframe['ema_short'].shift(1))
        short_entry_conditions.append(dataframe['ema_mid'] <= dataframe['ema_mid'].shift(1))
        short_entry_conditions.append(dataframe['ema_long'] <= dataframe['ema_long'].shift(1))
        
        short_entry_conditions.append(dataframe['ema_large'] * self.large_up_ratio <= dataframe['ema_large'].shift(1))
        short_entry_conditions.append(dataframe['adx'] >= self.adx_threshold)
        short_entry_conditions.append(dataframe['rsi'] <= self.rsi_short_threshold)
        
        # dataframe.loc[
        # (
        #     (reduce(lambda x, y: x & y, short_entry_conditions))
        # )
        # , 'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        return dataframe

    @property
    def protections(self): # type: ignore
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.mid_period,
                "trade_limit": 2,
                "stop_duration_candles": 2,
                "only_per_pair": True,
                "only_per_side": False
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage
 