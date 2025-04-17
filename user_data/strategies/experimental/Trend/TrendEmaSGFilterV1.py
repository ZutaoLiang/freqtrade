from scipy.signal import savgol_filter
from math import isnan
import numpy as np
import pandas_ta as pta
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, informative
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Order, Trade
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
import logging
logger = logging.getLogger(__name__)


class TrendEmaSGFilterV1(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=5, space='buy')

    base_stop_loss = 0.12
    stoploss = -base_stop_loss * buy_leverage.value

    trailing_stop = True
    trailing_stop_positive = 0.15 * buy_leverage.value
    trailing_stop_positive_offset = 0
    trailing_only_offset_is_reached = False

    can_short = True
 
    timeframe = '30m'

    lookback_period = 10
    
    window_length = IntParameter(10, 100, default=lookback_period, space='buy')
    polyorder = IntParameter(1, 5, default=1, space='fixed')

    ema_short_len = IntParameter(5, 100, default=lookback_period, space='buy')
    ema_mid_len = IntParameter(5, 100, default=lookback_period * 2, space='buy')
    ema_long_len = IntParameter(5, 100, default=lookback_period * 6, space='buy')

    startup_candle_count = int(max(window_length.value, ema_long_len.value) * 1.2)
    
    entry_ratio = DecimalParameter(1.0001, 1.0010, default=1.003, decimals=4, space='buy')
    exit_ratio = DecimalParameter(1.0001, 1.0010, default=1.002, decimals=4, space='buy')
    
    highest_period = lookback_period
    lowest_period = lookback_period
    trend = 3

    near_shift = 0
    far_shift = 2
    
    atr_period = ema_mid_len.value

    def savgol_smooth(self, data):
        smoothed_data = savgol_filter(data, self.window_length.value, self.polyorder.value, mode='nearest')
        return smoothed_data
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        
        dataframe['ema_short'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_short_len.value, talib=False)
        dataframe['smoothed_ema_short'] = self.savgol_smooth(dataframe['ema_short'].values)

        dataframe['ema_mid'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_mid_len.value, talib=False)
        dataframe['smoothed_ema_mid'] = self.savgol_smooth(dataframe['ema_mid'].values)
        
        dataframe['ema_long'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_long_len.value, talib=False)
        dataframe['smoothed_ema_long'] = self.savgol_smooth(dataframe['ema_long'].values)
        
        dataframe['highest'] = dataframe['ohlc4'].rolling(window=self.highest_period).max()
        dataframe['lowest'] = dataframe['ohlc4'].rolling(window=self.lowest_period).min()
        
        dataframe['prev_diff_short'] = dataframe['smoothed_ema_short'].shift(self.near_shift) / dataframe['smoothed_ema_short'].shift(self.far_shift)
        dataframe['prev_diff_mid'] = dataframe['smoothed_ema_mid'].shift(self.near_shift) / dataframe['smoothed_ema_mid'].shift(self.far_shift)
        
        dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_period)
        
        return dataframe
    
    def indicator_up_n_periods_mask(self, dataframe: DataFrame, indicator: str, days: int):
        indicator_up_mask = (dataframe[f'{indicator}'] > dataframe[f'{indicator}'].shift(1))
        for i in range(2, days):
            indicator_up_mask = indicator_up_mask & (dataframe[f'{indicator}'].shift(i-1) > dataframe[f'{indicator}'].shift(i))
        return indicator_up_mask
    
    def indicator_down_n_periods_mask(self, dataframe: DataFrame, indicator: str, days: int):
        indicator_down_mask = (dataframe[f'{indicator}'] < dataframe[f'{indicator}'].shift(1))
        for i in range(2, days):
            indicator_down_mask = indicator_down_mask & (dataframe[f'{indicator}'].shift(i-1) < dataframe[f'{indicator}'].shift(i))
        return indicator_down_mask
 
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['smoothed_ema_mid'].shift(self.near_shift) > self.entry_ratio.value * dataframe['smoothed_ema_mid'].shift(self.far_shift))
                & (dataframe['smoothed_ema_short'] > dataframe['smoothed_ema_mid'])
                & (dataframe['ohlc4'] > dataframe['smoothed_ema_short'])
                
                & (self.indicator_up_n_periods_mask(dataframe, 'smoothed_ema_short', self.trend))
                & (self.indicator_up_n_periods_mask(dataframe, 'smoothed_ema_mid', self.trend))
                & (self.indicator_up_n_periods_mask(dataframe, 'smoothed_ema_long', self.trend))
             ), 
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['smoothed_ema_mid'].shift(self.near_shift) * self.entry_ratio.value < dataframe['smoothed_ema_mid'].shift(self.far_shift))
                & (dataframe['smoothed_ema_short'] < dataframe['smoothed_ema_mid'])
                & (dataframe['ohlc4'] < dataframe['smoothed_ema_short'])
                
                & (self.indicator_down_n_periods_mask(dataframe, 'smoothed_ema_short', self.trend))
                & (self.indicator_down_n_periods_mask(dataframe, 'smoothed_ema_mid', self.trend))
                & (self.indicator_down_n_periods_mask(dataframe, 'smoothed_ema_long', self.trend))
            ), 
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['smoothed_ema_mid'].shift(self.near_shift) * self.exit_ratio.value < dataframe['smoothed_ema_mid'].shift(self.far_shift))
                | (dataframe['smoothed_ema_short'] < dataframe['smoothed_ema_mid'])
                | (dataframe['ohlc4'] < dataframe['smoothed_ema_mid'])
            ), 
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['smoothed_ema_mid'].shift(self.near_shift) > self.exit_ratio.value * dataframe['smoothed_ema_mid'].shift(self.far_shift))
                | (dataframe['smoothed_ema_short'] > dataframe['smoothed_ema_mid'])
                | (dataframe['ohlc4'] > dataframe['smoothed_ema_mid'])
            ), 
            'exit_short'] = 1

        # logger.info(f'{metadata["pair"]}\n{dataframe.tail(3*self.lookback_period)[["date", "close", "ohlc4", "smoothed_ema", "smoothed_ema_mid", "prev_diff_mid", "enter_long", "exit_long", "enter_short", "exit_short"]]}')
        return dataframe
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, 
                    **kwargs,) -> Optional[Union[str, bool]]:
        # if current_time >= trade.open_date + timedelta(minutes=30) and current_profit < 0.01:
        #     return 'exit_low_profit'
        return None

    @property
    def protections(self): # type: ignore
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 1
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.buy_leverage.value
