from freqtrade.strategy import IStrategy
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame

import numpy as np
import pandas_ta as pta
from datetime import datetime
from typing import Optional


class SuperTrend001(IStrategy):
    # ROI table:
    minimal_roi = {"0": 100}

    leverage_ratio = 1

    # Stoploss:
    stoploss = -0.05 * leverage_ratio

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.144
    trailing_only_offset_is_reached = False

    can_short = True

    minutes = 5
    timeframe = f"{minutes}m"

    atr_period = 1
    atr_multiplier = 3.0

    startup_candle_count = 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        heikinashi = qtpylib.heikinashi(df)
        df['ha_open'] = heikinashi['open']
        df['ha_high'] = heikinashi['high']
        df['ha_low'] = heikinashi['low']
        df['ha_close'] = heikinashi['close']
        
        df['atr'] = pta.sma(close=pta.true_range(df['ha_high'], df['ha_low'], df['ha_close']), length=self.atr_period, talib=False)
        
        src = (df['ha_high'] + df['ha_low']) / 2
        
        df['up'] = src - (self.atr_multiplier * df['atr'])
        df['up1'] = df['up'].shift(1)
        df['up'] = np.where(df['close'].shift(1) > df['up1'], 
                                   np.maximum(df['up'], df['up1']), df['up'])
        
        df['dn'] = src + (self.atr_multiplier * df['atr'])
        df['dn1'] = df['dn'].shift(1)
        df['dn'] = np.where(df['close'].shift(1) < df['dn1'], 
                                   np.minimum(df['dn'], df['dn1']), df['dn'])

        df['trend'] = 1
        df['trend'] = np.where((df['trend'].shift(1) == -1) & (df['close'] > df['dn1']), 1, df['trend'])
        df['trend'] = np.where((df['trend'].shift(1) == 1) & (df['close'] < df['up1']), -1, df['trend'])
        
        df['buy'] = (df['trend'] == 1) & (df['trend'].shift(1) == -1)
        df['sell'] = (df['trend'] == -1) & (df['trend'].shift(1) == 1)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe['buy'], 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe['sell'], 'exit_long'] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.leverage_ratio
