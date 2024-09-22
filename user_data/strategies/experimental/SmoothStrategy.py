import pandas as pd
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta
from datetime import datetime
from typing import Optional, Tuple, Union


class SmoothStrategy(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.15
    
    # Stoploss:
    stoploss = -base_stop_loss * buy_leverage.value

    can_short = True
 
    timeframe = '1h'
    
    startup_candle_count = 200

    rolling_period = IntParameter(low=5, high=90, default=90, space='buy', optimize=True)
    ma_period = IntParameter(low=5, high=90, default=200, space='buy', optimize=True)
    
    def heikinashi(self, dataframe: DataFrame) -> DataFrame:
        ha = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = self.heikinashi(dataframe)
        dataframe['rolling_open'] = dataframe['ha_open'].rolling(window=self.rolling_period.value).mean()
        dataframe['rolling_close'] = dataframe['ha_close'].rolling(window=self.rolling_period.value).mean()
        
        dataframe['ha_mid'] = (dataframe['rolling_open'] + dataframe['rolling_close']) / 2
        dataframe['ha_ema'] = pta.ema(close=dataframe['ha_mid'], length=self.ma_period.value, talib=False)
        dataframe['ha_sma'] = pta.sma(close=dataframe['ha_mid'], length=self.ma_period.value, talib=False)
        
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe['ha_close'] > dataframe['ha_ema'])
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['ha_close'] < dataframe['ha_ema'])
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe['ha_close'] < dataframe['ha_ema'])
            ),
            'exit_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['ha_close'] > dataframe['ha_ema'])
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
