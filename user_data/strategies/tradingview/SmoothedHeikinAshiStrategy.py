from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy import IntParameter
from datetime import datetime
from typing import Optional, Tuple, Union

class SmoothedHeikinAshiStrategy(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.2
    
    # Stoploss:
    stoploss = -base_stop_loss * buy_leverage.value

    can_short = True
 
    timeframe = '1h'
    
    startup_candle_count = 100

    first_ema_length = IntParameter(5, 100, default=24*7, space='buy', optimize=True)
    second_ema_length = IntParameter(5, 100, default=24, space='buy', optimize=True)
    
    def calculate_heikinashi(self, dataframe: DataFrame) -> DataFrame:
        # First EMA applied to OHLC
        dataframe['ha_o'] = pta.ema(dataframe['open'], length=self.first_ema_length.value)
        dataframe['ha_h'] = pta.ema(dataframe['high'], length=self.first_ema_length.value)
        dataframe['ha_l'] = pta.ema(dataframe['low'], length=self.first_ema_length.value)
        dataframe['ha_c'] = pta.ema(dataframe['close'], length=self.first_ema_length.value)

        # Heikin Ashi calculation
        haclose = (dataframe['ha_o'] + dataframe['ha_h'] + dataframe['ha_l'] + dataframe['ha_c']) / 4
        
        # haclose = (dataframe['ha_o'] + dataframe['ha_c']) / 2

        # Initialize ha_open using the first available value
        haopen = [(dataframe['ha_o'].iloc[0] + dataframe['ha_c'].iloc[0]) / 2]

        # Calculate ha_open for the rest of the candles
        for i in range(1, len(dataframe)):
            previous_haopen = haopen[i - 1]
            if pd.isna(previous_haopen):  # If previous ha_open is NaN, initialize it
                haopen.append((dataframe['ha_o'].iloc[i] + dataframe['ha_c'].iloc[i]) / 2)
            else:
                haopen.append((previous_haopen + haclose[i - 1]) / 2)

        dataframe['ha_open'] = haopen
        dataframe['ha_close'] = haclose

        # Final EMA applied to ha_open and ha_close
        dataframe['ha_open_smooth'] = pta.ema(dataframe['ha_open'], length=self.second_ema_length.value)
        dataframe['ha_close_smooth'] = pta.ema(dataframe['ha_close'], length=self.second_ema_length.value)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.calculate_heikinashi(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['ha_close_smooth'] > dataframe['ha_open_smooth']  # Condition to enter long
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                dataframe['ha_close_smooth'] < dataframe['ha_open_smooth']  # Condition to enter short
            ),
            'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['ha_close_smooth'] < dataframe['ha_open_smooth']  # Condition to exit long
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                dataframe['ha_close_smooth'] > dataframe['ha_open_smooth']  # Condition to exit short
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

