from freqtrade.strategy.interface import IStrategy
import numpy as np
import pandas_ta as pta
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IntParameter, DecimalParameter, informative
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from typing import Optional, List


class MultiEmaProdStrategy(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.1
    stoploss = -base_stop_loss * buy_leverage.value

    trailing_stop = True
    trailing_stop_positive = 0.09 * buy_leverage.value
    trailing_stop_positive_offset = 0
    trailing_only_offset_is_reached = False

    can_short = True
 
    timeframe = '1m'
    
    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 2
        }
    ] # type: ignore
    
    ema_short_period = IntParameter(5, 100, default=240, space='buy')
    ema_mid_period = IntParameter(5, 100, default=720, space='buy')
    ema_long_period = IntParameter(5, 100, default=1440, space='buy')
    
    ema_dist_percent_entry = 0.08
    ema_dist_percent_exit = 0.02
    ema_days = 5
    
    startup_candle_count = ema_long_period.value
    
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
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['ema_short'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_short_period.value, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_mid_period.value, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_long_period.value, talib=False)

        dataframe['dist_short'] = dataframe['ohlc4'] - dataframe['ema_short']
        dataframe['dist_mid'] = dataframe['ohlc4'] - dataframe['ema_mid']
        dataframe['dist_long'] = dataframe['ohlc4'] - dataframe['ema_long']
        
        dataframe['dist_short_percent'] = (dataframe['ohlc4'] - dataframe['ema_short']) / dataframe['ohlc4']
        dataframe['dist_mid_percent'] = (dataframe['ohlc4'] - dataframe['ema_mid']) / dataframe['ohlc4']
        dataframe['dist_long_percent'] = (dataframe['ohlc4'] - dataframe['ema_long']) / dataframe['ohlc4']
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
            ), 
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_down_mask_short = self.ema_down_n_days_mask(dataframe, 'ema_short', self.ema_days)
        ema_down_mask_mid = self.ema_down_n_days_mask(dataframe, 'ema_mid', self.ema_days)
        ema_down_mask_long = self.ema_down_n_days_mask(dataframe, 'ema_long', self.ema_days)

        dataframe.loc[
            (
                (dataframe['ohlc4'] < dataframe['ema_mid'])
                | (dataframe['dist_long_percent'] < self.ema_dist_percent_exit)
                # | ema_down_mask_short
                | ema_down_mask_mid
                | ema_down_mask_long
            ), 
            'exit_long'] = 1

        ema_up_mask_short = self.ema_up_n_days_mask(dataframe, 'ema_short', self.ema_days)
        ema_up_mask_mid = self.ema_up_n_days_mask(dataframe, 'ema_mid', self.ema_days)
        ema_up_mask_long = self.ema_up_n_days_mask(dataframe, 'ema_long', self.ema_days)
        
        dataframe.loc[
            (
                (dataframe['ohlc4'] > dataframe['ema_mid'])
                | (dataframe['dist_long_percent'] > -self.ema_dist_percent_exit)
                # | ema_up_mask_short
                | ema_up_mask_mid
                | ema_up_mask_long
            ), 
            'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.buy_leverage.value
