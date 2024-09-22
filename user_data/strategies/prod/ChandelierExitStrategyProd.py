from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import numpy as np
import pandas as pd
import pandas_ta as pta
from datetime import datetime
from typing import Optional, Tuple, Union

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (DecimalParameter, IntParameter)

class ChandelierExitStrategyProd(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.1
    
    # Stoploss:
    stoploss = -base_stop_loss * buy_leverage.value

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = base_stop_loss * buy_leverage.value
    trailing_stop_positive_offset = 0
    trailing_only_offset_is_reached = False

    can_short = True
 
    timeframe = '1h'
    
    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 3
        }
    ] # type: ignore
    
    base_period = 24
    short_period = base_period
    long_period = base_period * 20
    startup_candle_count = long_period

    # short_length = IntParameter(10, 50, default=short_period, space='buy', optimize=True)
    # short_mult = DecimalParameter(1.5, 2.5, default=1.5, space='buy', optimize=True, decimals=1)

    long_length = IntParameter(50, 500, default=long_period, space='buy', optimize=True)
    long_mult = DecimalParameter(1.5, 2.5, default=2, space='buy', optimize=True, decimals=1)
    
    ema_length = IntParameter(20, 200, default=long_period, space='buy')
    
    diff_percent = 0.03

    def heikinashi(self, dataframe: DataFrame) -> DataFrame:
        ha = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']
        return dataframe

    def chandelier_exit(self, dataframe: DataFrame, length: int, mult: float, suffix: str = '') -> DataFrame:
        high = dataframe['ha_high']
        low = dataframe['ha_low']
        close = dataframe['ha_close']
        atr = pta.atr(high=high, low=low, close=close, length=length)
        
        long_stop = high.rolling(window=length).max() - mult * atr # type: ignore
        short_stop = low.rolling(window=length).min() + mult * atr # type: ignore
        
        long_stop_prev = long_stop.shift(1)
        long_stop = np.where(close.shift(1) > long_stop_prev, np.maximum(long_stop, long_stop_prev), long_stop)
        
        short_stop_prev = short_stop.shift(1)
        short_stop = np.where(close.shift(1) < short_stop_prev, np.minimum(short_stop, short_stop_prev), short_stop)
        
        stop = np.where((close > close.shift(1)) & (close > long_stop_prev), long_stop,
                        np.where((close < close.shift(1)) & (close < short_stop_prev), short_stop, np.nan))
        
        dataframe[f'long_stop{suffix}'] = pd.Series(long_stop).ffill()  # Forward fill to propagate last non-NaN value
        dataframe[f'short_stop{suffix}'] = pd.Series(short_stop).ffill()  # Forward fill to propagate last non-NaN value
        dataframe[f'chandelier_exit{suffix}'] = pd.Series(stop).ffill()  # Forward fill to propagate last non-NaN value
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.heikinashi(dataframe)
        
        # dataframe = self.chandelier_exit(dataframe, length=self.short_length.value, mult=self.short_mult.value, suffix='_short')
        dataframe = self.chandelier_exit(dataframe, length=self.long_length.value, mult=self.long_mult.value, suffix='_long')
        
        dataframe['ema'] = pta.ema(close=dataframe['ha_close'], length=self.ema_length.value, talib=False)
        
        dataframe['diff_percent'] = (dataframe['chandelier_exit_long'] - dataframe['ema']).abs() / dataframe['ema']
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['chandelier_exit_long'] > dataframe['ema'])
                & (dataframe['diff_percent'] > self.diff_percent)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['chandelier_exit_long'] < dataframe['ema'])
                & (dataframe['diff_percent'] > self.diff_percent)
            ),
            'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['chandelier_exit_long'] <= dataframe['ema'])
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['chandelier_exit_long'] >= dataframe['ema'])
            ),
            'exit_short'] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.buy_leverage.value

