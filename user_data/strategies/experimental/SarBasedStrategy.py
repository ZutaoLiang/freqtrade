from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import pandas_ta as pta

from freqtrade.strategy import IntParameter, DecimalParameter
from datetime import datetime
from typing import Optional, Tuple, Union

class SarBasedStrategy(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.15
    
    # Stoploss:
    stoploss = -base_stop_loss * buy_leverage.value

    can_short = True
 
    timeframe = '1h'
    
    startup_candle_count = 120

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        df['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        dataframe['h_close'] = df['close']
        
        df['ema'] = ta.EMA(df, timepriod=60)
        df['close'] = df['ema']
        df['sar'] = ta.SAR(df, 0.01, 0.2)
        
        dataframe['sar'] = df['sar']
        dataframe['ema'] = df['ema']
        dataframe['s_close'] = df['close']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] > dataframe['sar']),
            'enter_long'] = 1

        dataframe.loc[
            (dataframe['close'] < dataframe['sar']),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] < dataframe['sar']),
            'exit_long'] = 1

        dataframe.loc[
            (dataframe['close'] > dataframe['sar']),
            'exit_short'] = 1

        return dataframe

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.buy_leverage.value
