from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy import IntParameter, DecimalParameter
from datetime import datetime
from typing import Optional, Tuple, Union

class PtaSuperTrend(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.1
    
    # Stoploss:
    stoploss = -base_stop_loss * buy_leverage.value

    can_short = True
 
    timeframe = '1h'
    
    startup_candle_count = 100

    length = IntParameter(10, 100, default=120, space='buy')
    smooth_length = IntParameter(10, 100, default=30, space='buy')
    multiplier = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space='buy')

    def calculate_supertrend(self, dataframe: DataFrame) -> DataFrame:
        high = dataframe['high']
        low = dataframe['low']
        close = dataframe['close']
        
        dataframe['ha_o'] = pta.ema(dataframe['open'], length=self.smooth_length.value)
        dataframe['ha_h'] = pta.ema(dataframe['high'], length=self.smooth_length.value)
        dataframe['ha_l'] = pta.ema(dataframe['low'], length=self.smooth_length.value)
        dataframe['ha_c'] = pta.ema(dataframe['close'], length=self.smooth_length.value)

        high = dataframe['ha_h']
        low = dataframe['ha_l']
        close = dataframe['ha_c']

        trend = pta.supertrend(high=high, low=low, close=close, length=self.length.value, multiplier=self.multiplier.value)
        _props = f"_{self.length.value}_{self.multiplier.value}"
        dataframe['trend'] = trend[f'SUPERT{_props}']
        dataframe['dir'] = trend[f'SUPERTd{_props}']
        dataframe['long'] = trend[f'SUPERTl{_props}']
        dataframe['short'] = trend[f'SUPERTs{_props}']
        
        dataframe['ema'] = pta.ema(close=dataframe['close'], length=self.length.value, talib=False)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.calculate_supertrend(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['trend'] > dataframe['ema'])
                & 
                (
                    (dataframe['dir'] == 1) 
                    # & (dataframe['dir'].shift(1) == -1)
                )
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['trend'] < dataframe['ema'])
                & 
                (
                    (dataframe['dir'] == -1) 
                    # & (dataframe['dir'].shift(1) == 1)
                )
            ),
            'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['trend'] < dataframe['ema'])
                & 
                (
                    (dataframe['dir'] == -1) 
                    # & (dataframe['dir'].shift(1) == 1)
                )
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['trend'] > dataframe['ema'])
                & 
                (
                    (dataframe['dir'] == 1) 
                    # & (dataframe['dir'].shift(1) == -1)
                )
            ),
            'exit_short'] = 1
        return dataframe

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 3
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.buy_leverage.value

