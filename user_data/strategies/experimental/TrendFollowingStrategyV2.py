import pandas as pd
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta
from datetime import datetime
from typing import Optional, Tuple, Union


class TrendFollowingStrategyV2(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.15
    
    # Stoploss:
    stoploss = -base_stop_loss * buy_leverage.value

    can_short = True
 
    timeframe = '1d'
    
    lookback_period = 30
    breakout_long = 50
    breakout_short = 25
    
    startup_candle_count = min(breakout_long, breakout_short)
    

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['highest_long'] = dataframe['close'].rolling(window=self.breakout_long).max()
        dataframe['lowest_long'] = dataframe['close'].rolling(window=self.breakout_long).min()
        dataframe['highest_short'] = dataframe['close'].rolling(window=self.breakout_short).max()
        dataframe['lowest_short'] = dataframe['close'].rolling(window=self.breakout_short).min()

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['highest_long'].shift(1))
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['lowest_long'].shift(1))
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['lowest_short'].shift(1))
            ),
            'exit_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['highest_short'].shift(1))
            ),
            'exit_short'] = 1

        return dataframe
    
    @property
    def protections(self): # type: ignore
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
