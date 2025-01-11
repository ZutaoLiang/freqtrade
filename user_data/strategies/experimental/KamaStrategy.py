from math import isnan
import numpy as np
import pandas_ta as pta
import pandas as pd
from pandas import DataFrame
from scipy.signal import savgol_filter

from datetime import datetime
from typing import Optional, Tuple, Union

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, informative
import freqtrade.vendor.qtpylib.indicators as qtpylib


class KamaStrategy(IStrategy):
    # common
    minimal_roi = {"0": 100}
    
    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    timeframe = '15m'
    
    stoploss = -0.1 * buy_leverage.value

    trailing_stop = True
    trailing_stop_positive = buy_leverage.value * 0.1
    trailing_stop_positive_offset = 0
    trailing_only_offset_is_reached = False
    
    can_short = True
    
    atr_length = 12
    atr_multiplier = 1.5
    
    risk_ratio = 0.001
    
    # strategy
    ema_length = int(96)
    ema_long_length = int(96 * 2)
    roc_length = 1
    
    highest_period = 3
    lowest_period = 3
    
    startup_candle_count = ema_long_length
    
    def heikinashi(self, dataframe: DataFrame) -> DataFrame:
        ha = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # common
        dataframe = self.heikinashi(dataframe)
        dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_length)
        dataframe['smooth'] = pta.ema(dataframe['ha_close'])
        dataframe['upper'] = dataframe['smooth'] + dataframe['atr']
        dataframe['lower'] = dataframe['smooth'] - dataframe['atr']
        
        # strategy
        dataframe['kama'] = pta.kama(close=dataframe['ha_close'], length=10)
        dataframe['kama_roc'] = pta.roc(close=dataframe['kama'], length=self.roc_length)
        dataframe['kama_roc_mean'] = dataframe['kama_roc'].rolling(7).mean()
        
        dataframe['ema'] = pta.ema(dataframe['kama'], length=self.ema_length)
        dataframe['ema_roc'] = pta.roc(close=dataframe['ema'], length=self.roc_length)
        dataframe['ema_roc_mean'] = dataframe['ema_roc'].rolling(7).mean()
        
        dataframe['ema_long'] = pta.ema(dataframe['kama'], length=self.ema_long_length)
        dataframe['ema_long_roc'] = pta.roc(close=dataframe['ema_long'], length=self.roc_length)
        dataframe['ema_long_roc_mean'] = dataframe['ema_long_roc'].rolling(7).mean()
        
        dataframe['highest'] = dataframe['ha_close'].rolling(window=self.highest_period).max()
        dataframe['lowest'] = dataframe['ha_close'].rolling(window=self.lowest_period).min()
        
        return dataframe
 
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(
            (dataframe['ema'] > dataframe['ema_long'])
            & (dataframe['kama'] > dataframe['ema'])
            & (dataframe['kama_roc_mean'] > 0)
            & (dataframe['ema_roc_mean'] > 0.01)
            & (dataframe['ha_close'] > dataframe['highest'].shift(1))
        ), 'enter_long'] = 1
        
        dataframe.loc[(
            (dataframe['ema'] < dataframe['ema_long'])
            & (dataframe['kama'] < dataframe['ema'])
            & (dataframe['kama_roc_mean'] < 0)
            & (dataframe['ema_roc_mean'] < -0.01)
            & (dataframe['ha_close'] < dataframe['lowest'].shift(1))
        ), 'enter_short'] = 1
        
        return dataframe
        
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(
            # (dataframe['kama'] < dataframe['ema'])
            (dataframe['ema'] < dataframe['ema_long'])
        ), 'exit_long'] = 1
        
        dataframe.loc[(
            # (dataframe['kama'] > dataframe['ema'])
            (dataframe['ema'] > dataframe['ema_long'])
        ), 'exit_short'] = 1
        return dataframe

    # def custom_stake_amount(
    #     self,
    #     pair: str,
    #     current_time: datetime,
    #     current_rate: float,
    #     proposed_stake: float,
    #     min_stake: Optional[float],
    #     max_stake: float,
    #     leverage: float,
    #     entry_tag: Optional[str],
    #     side: str,
    #     **kwargs,
    # ) -> float:
    #     stake_amount = proposed_stake
    #     if self.wallets is None or self.risk_ratio <= 0:
    #         return stake_amount

    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     if dataframe is None or dataframe.empty:
    #         return stake_amount
        
    #     last_candle = dataframe.loc[dataframe['date'] <= current_time]
    #     if last_candle.empty:
    #         return stake_amount
        
    #     last_candle = last_candle.iloc[-1]
    #     atr = last_candle['atr'] # 实际策略运行过程中当前K线可能不完整，对应计算的ATR可能不正确，这里采用倒数第二根K线的计算值
    #     if isnan(atr):
    #         return stake_amount
        
    #     balance = self.wallets.get_total(self.stake_currency)
    #     risk_amount = (self.risk_ratio * balance / atr) * current_rate
        
    #     stake_amount = risk_amount
            
    #     return stake_amount

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
     