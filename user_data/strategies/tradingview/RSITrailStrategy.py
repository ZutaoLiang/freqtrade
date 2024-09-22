import numpy as np
import pandas as pd
from freqtrade.strategy.interface import IStrategy

import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas_ta as pta
from freqtrade.strategy import IntParameter, DecimalParameter

class RSITrailStrategy(IStrategy):
    minimal_roi = {"0": 100}
    stoploss = -0.10
    timeframe = '1d'
    can_short = True

    base_period = 27
    rsi_length = IntParameter(7, 21, default=base_period, space='buy')
    # rsi_threshold = DecimalParameter(30.0, 70.0, default=50.0, space='buy')
    rsi_upper = IntParameter(50, 80, default=60, space='buy')
    rsi_lower = IntParameter(30, 50, default=40, space='buy')
    
    ema_length = IntParameter(20, 50, default=base_period, space='buy')
    atr_length = IntParameter(10, 30, default=base_period, space='buy')

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        
        dataframe['ma5'] = pta.sma(close=dataframe['ohlc4'], length=5)        
        dataframe['ma5_5'] = pta.sma(close=dataframe['ma5'], length=5)
        dataframe['ma10'] = pta.sma(close=dataframe['ohlc4'], length=10)

        dataframe['rsi'] = qtpylib.rsi(dataframe['ohlc4'], self.rsi_length.value)
        dataframe['atr'] = qtpylib.atr(dataframe, self.atr_length.value)
        dataframe['ema'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_length.value)
        dataframe['ema_5'] = pta.ema(close=dataframe['ma5'], length=self.ema_length.value)

        dataframe['upper_bound'] = dataframe['ema'] + (self.rsi_upper.value - 50) / 10 * dataframe['atr']
        dataframe['lower_bound'] = dataframe['ema'] - (50 - self.rsi_lower.value) / 10 * dataframe['atr']

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['ohlc4'], dataframe['upper_bound'])
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['ohlc4'], dataframe['lower_bound'])
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['ohlc4'], dataframe['lower_bound'])
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['ohlc4'], dataframe['upper_bound'])
            ),
            'exit_short'] = 1

        return dataframe

