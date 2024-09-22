from freqtrade.strategy import IStrategy
from freqtrade.strategy import (DecimalParameter, IntParameter)
from freqtrade.persistence import Trade

import freqtrade.vendor.qtpylib.indicators as qtpylib

from pandas import DataFrame
import pandas_ta as pta
from datetime import datetime
from typing import Optional, Tuple, Union

import logging
logger = logging.getLogger(__name__)

class LongEmaStrategy(IStrategy):
    
    minimal_roi = {"0": 100}
    
    buy_leverage = IntParameter(1, 5, default=3, space='buy')
    
    base_stop_loss = 0.04
    
    stoploss = -base_stop_loss * buy_leverage.value
    
    # Trailing stop:
    trailing_stop = False
    # trailing_stop_positive = -stoploss * 2
    # trailing_stop_positive_offset = 0
    # trailing_only_offset_is_reached = False
    
    can_short = True
    
    minutes = 5
    timeframe = f"{minutes}m"
    
    buy_ema = IntParameter(500, 3000, default=960, space='buy')
    buy_lookback_period = IntParameter(5, 30, default=10, space='buy')
    startup_candle_count = buy_ema.value

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.buy_leverage.value
    
    @property
    def protections(self):
        return  [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 3
            }
        ]

    def calculate_highest_lowest(self, df: DataFrame, p):
        df['period_high'] = df[['open', 'close']].max(axis=1)
        df['period_low'] = df[['open', 'close']].min(axis=1)
        df['highest'] = df['close'].rolling(window=p).max()
        df['lowest'] = df['close'].rolling(window=p).min()
     
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema'] = pta.ema(dataframe['close'], length=self.buy_ema.value)
        dataframe['ema_diff'] = dataframe['ema'] - dataframe['ema'].shift(1)
        self.calculate_highest_lowest(dataframe, self.buy_lookback_period.value)

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        cond1 = dataframe['ema_diff'].rolling(window=self.buy_lookback_period.value).apply(lambda x: (x > 0).sum() >= (self.buy_lookback_period.value * 0.8), raw=True) == 1
        
        cond2 = (dataframe['close'] - dataframe['ema']).rolling(window=self.buy_lookback_period.value).sum() > 0
        
        cond3 = dataframe['close'] >= dataframe['ema'] * 1.025

        dataframe.loc[cond1 & cond2 & cond3, 'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        cond1 = dataframe['ema_diff'].rolling(window=self.buy_lookback_period.value).apply(lambda x: (x < 0).sum() > (self.buy_lookback_period.value * 0.5), raw=True) == 1
        
        cond2 = dataframe['close'] < dataframe['ema']
        
        dataframe.loc[cond1 | cond2, 'exit_long'] = 1
        return dataframe