import numpy as np
import pandas as pd
import pandas_ta as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class AlphaStrategy(IStrategy):
    minimal_roi = {"0": 0.1, "30": 0.05, "60": 0.03, "90": 0}
    stoploss = -0.05
    timeframe = '1h'
    can_short = True

    def alpha_factors(self, dataframe: DataFrame):
        # 计算 RSI 因子
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)
        dataframe['rsi_factor'] = dataframe['rsi'].apply(lambda x: 1 if x < 30 else (-1 if x > 70 else 0))

        # 计算布林带因子
        bollinger = ta.bbands(dataframe['close'], length=20)
        dataframe['bb_factor'] = np.where(
            (dataframe['close'] < bollinger['BBL_20_2.0']) | (dataframe['close'] > bollinger['BBU_20_2.0']), 1, 0
        )

        # 计算 ADX 因子
        adx = ta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
        dataframe['adx'] = adx['ADX_14']
        dataframe['adx_factor'] = np.where(
            (dataframe['adx'] > 20) & (dataframe['adx'] > dataframe['adx'].shift(1)), 1, 0
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.alpha_factors(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi_factor'] > 0) &
                (dataframe['bb_factor'] > 0) &
                (dataframe['adx_factor'] > 0)
            ),
            'enter_long'] = 1
        dataframe.loc[
            (
                (dataframe['rsi_factor'] < 0) &
                (dataframe['bb_factor'] < 0) &
                (dataframe['adx_factor'] < 0)
            ),
            'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi_factor'] < 0) &
                (dataframe['bb_factor'] < 0) &
                (dataframe['adx_factor'] < 0)
            ),
            'exit_long'] = 1
        dataframe.loc[
            (
                (dataframe['rsi_factor'] > 0) &
                (dataframe['bb_factor'] > 0) &
                (dataframe['adx_factor'] > 0)
            ),
            'exit_short'] = 1
        return dataframe
