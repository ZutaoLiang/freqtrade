from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import numpy as np
import pandas as pd
import pandas_ta as pta
from datetime import datetime
from typing import Optional, Tuple, Union

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (DecimalParameter, IntParameter)

class EmaChandelierExitProdStrategy(IStrategy):
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
 
    timeframe = '1m'
    
    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 3
        }
    ] # type: ignore
    
    ema_period = 24 * 20

    long_length = IntParameter(50, 500, default=ema_period, space='buy', optimize=True)
    long_mult = DecimalParameter(1.5, 2.5, default=2, space='buy', optimize=True, decimals=1)
    ema_length = IntParameter(20, 200, default=ema_period, space='buy')
    
    ema_short_period = IntParameter(5, 100, default=240, space='buy')
    ema_mid_period = IntParameter(5, 100, default=720, space='buy')
    ema_long_period = IntParameter(5, 100, default=1440, space='buy')
    
    ema_dist_percent_entry = 0.07
    ema_dist_percent_exit = 0.02
    ema_days = 3
 
    diff_percent = 0.03

    startup_candle_count = ema_long_period.value

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.heikinashi(dataframe)
        
        dataframe = self.chandelier_exit(dataframe, length=self.long_length.value, mult=self.long_mult.value, suffix='_long')
        
        dataframe['ema'] = pta.ema(close=dataframe['ha_close'], length=self.ema_length.value, talib=False)
        
        dataframe['diff_percent'] = (dataframe['chandelier_exit_long'] - dataframe['ema']).abs() / dataframe['ema']

        dataframe['ema_short'] = pta.ema(close=dataframe['ha_close'], length=self.ema_short_period.value, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['ha_close'], length=self.ema_mid_period.value, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ha_close'], length=self.ema_long_period.value, talib=False)

        dataframe['dist_short'] = dataframe['ha_close'] - dataframe['ema_short']
        dataframe['dist_mid'] = dataframe['ha_close'] - dataframe['ema_mid']
        dataframe['dist_long'] = dataframe['ha_close'] - dataframe['ema_long']
        
        dataframe['dist_short_percent'] = dataframe['dist_short'] / dataframe['ha_close']
        dataframe['dist_mid_percent'] = dataframe['dist_mid'] / dataframe['ha_close']
        dataframe['dist_long_percent'] = dataframe['dist_long'] / dataframe['ha_close']
         
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_up_mask_short = self.ema_up_n_days_mask(dataframe, 'ema_short', self.ema_days)
        ema_up_mask_mid = self.ema_up_n_days_mask(dataframe, 'ema_mid', self.ema_days)
        ema_up_mask_long = self.ema_up_n_days_mask(dataframe, 'ema_long', self.ema_days)
        
        dataframe.loc[
            (
                (dataframe['dist_long_percent'] > self.ema_dist_percent_entry)
                & (dataframe['ema_short'] > dataframe['ema_mid'])
                & (dataframe['ema_mid'] > dataframe['ema_long'])
                & ema_up_mask_short
                & ema_up_mask_mid
                & ema_up_mask_long

                & (dataframe['chandelier_exit_long'] > dataframe['ema'])
                & (dataframe['diff_percent'] > self.diff_percent)
            ), 
            'enter_long'] = 1
        
        ema_down_mask_short = self.ema_down_n_days_mask(dataframe, 'ema_short', self.ema_days)
        ema_down_mask_mid = self.ema_down_n_days_mask(dataframe, 'ema_mid', self.ema_days)
        ema_down_mask_long = self.ema_down_n_days_mask(dataframe, 'ema_long', self.ema_days)

        dataframe.loc[
            (
                (dataframe['dist_long_percent'] < -self.ema_dist_percent_entry)
                & (dataframe['ema_short'] < dataframe['ema_mid'])
                & (dataframe['ema_mid'] < dataframe['ema_long'])
                & ema_down_mask_short
                & ema_down_mask_mid
                & ema_down_mask_long

                & (dataframe['chandelier_exit_long'] < dataframe['ema'])
                & (dataframe['diff_percent'] > self.diff_percent)
            ), 
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['chandelier_exit_long'] <= dataframe['ema'])
                | (dataframe['ha_close'] <= dataframe['ema_mid'] )
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['chandelier_exit_long'] >= dataframe['ema'])
                | (dataframe['ha_close'] >= dataframe['ema_mid'] )
            ),
            'exit_short'] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.buy_leverage.value

