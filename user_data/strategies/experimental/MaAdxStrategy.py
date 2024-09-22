from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas_ta as pta
from freqtrade.strategy import IntParameter, DecimalParameter
from datetime import datetime
from typing import Optional, Tuple, Union

class MaAdxStrategy(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.15
    
    # Stoploss:
    stoploss = -base_stop_loss * buy_leverage.value

    can_short = True
 
    timeframe = '1h'
    
    startup_candle_count = 120

    length = IntParameter(10, 100, default=90, space='buy')
    smooth_length = IntParameter(10, 100, default=15, space='buy')
    adx_threshold = IntParameter(15, 40, default=30, space='buy')

    def calculate_supertrend(self, dataframe: DataFrame) -> DataFrame:
        # high = dataframe['high']
        # low = dataframe['low']
        # close = dataframe['close']
        
        dataframe['ha_o'] = pta.ema(dataframe['open'], length=self.smooth_length.value)
        dataframe['ha_h'] = pta.ema(dataframe['high'], length=self.smooth_length.value)
        dataframe['ha_l'] = pta.ema(dataframe['low'], length=self.smooth_length.value)
        dataframe['ha_c'] = pta.ema(dataframe['close'], length=self.smooth_length.value)

        # high = dataframe['ha_h']
        # low = dataframe['ha_l']
        # close = dataframe['ha_c']

        smooth_close = (dataframe['ha_o'] + dataframe['ha_h'] + dataframe['ha_l'] + dataframe['ha_c']) / 4
        dataframe['ema_ohlc4'] = pta.ema(close=smooth_close, length=self.length.value, talib=False)
       
        dataframe['ema'] = pta.ema(close=dataframe['close'], length=self.length.value, talib=False)
        dataframe['adx'] = pta.adx(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], length=self.smooth_length.value)[f'ADX_{self.smooth_length.value}']

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.calculate_supertrend(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ema'] > dataframe['ema_ohlc4'])
                & (dataframe['adx'] > self.adx_threshold.value)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['ema'] < dataframe['ema_ohlc4'])
                & (dataframe['adx'] > self.adx_threshold.value)
            ),
            'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ema'] < dataframe['ema_ohlc4'])
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['ema'] > dataframe['ema_ohlc4'])
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

