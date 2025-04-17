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


class SGFilterEmaV3(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=5, space='buy')

    base_stop_loss = 0.15
    stoploss = -base_stop_loss * buy_leverage.value

    trailing_stop = True
    trailing_stop_positive = 0.15 * buy_leverage.value
    trailing_stop_positive_offset = 0
    trailing_only_offset_is_reached = False

    can_short = True
 
    timeframe = '15m'

    lookback_period = 10
    
    window_length = IntParameter(10, 100, default=lookback_period, space='buy')
    polyorder = IntParameter(1, 5, default=1, space='fixed')

    ema_period = IntParameter(5, 100, default=lookback_period, space='buy')
    ema_mid_period = IntParameter(5, 100, default=lookback_period * 2, space='buy')

    startup_candle_count = int(max(window_length.value, ema_mid_period.value) * 1.2)
    
    up_ratio = DecimalParameter(1.0001, 1.0010, default=1.003, decimals=4, space='buy')
    down_ratio = DecimalParameter(1.0001, 1.0010, default=1.002, decimals=4, space='buy')
    
    highest_period = lookback_period
    lowest_period = lookback_period

    prev_shift = 1
    shift_interval = 3
    
    atr_period = ema_mid_period.value

    def savgol_smooth(self, data):
        smoothed_data = savgol_filter(data, self.window_length.value, self.polyorder.value, mode='nearest')
        return smoothed_data
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['ema'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_period.value, talib=False)
        dataframe['smoothed_ema'] = self.savgol_smooth(dataframe['ema'].values)

        dataframe['ema_mid'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_mid_period.value, talib=False)
        dataframe['smoothed_ema_mid'] = self.savgol_smooth(dataframe['ema_mid'].values)
        
        dataframe['highest'] = dataframe['ohlc4'].rolling(window=self.highest_period).max()
        dataframe['lowest'] = dataframe['ohlc4'].rolling(window=self.lowest_period).min()
        
        dataframe['prev_diff'] = dataframe['smoothed_ema'].shift(self.prev_shift) / dataframe['smoothed_ema'].shift(self.shift_interval)
        dataframe['prev_diff_mid'] = dataframe['smoothed_ema_mid'].shift(self.prev_shift) / dataframe['smoothed_ema_mid'].shift(self.shift_interval)
        
        # dataframe['price_percentile'] = dataframe['ohlc4'].rolling(window=self.lookback_period * 6).apply(
        #     lambda x: percentileofscore(x, x.iloc[-1])
        # )
        
        dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_period)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['smoothed_ema_mid'].shift(self.prev_shift) > self.up_ratio.value * dataframe['smoothed_ema_mid'].shift(self.shift_interval))
                & (dataframe['smoothed_ema'] > dataframe['smoothed_ema_mid'])
                & (dataframe['ohlc4'] > dataframe['smoothed_ema'])
             ), 
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['smoothed_ema_mid'].shift(self.prev_shift) * self.up_ratio.value < dataframe['smoothed_ema_mid'].shift(self.shift_interval))
                & (dataframe['smoothed_ema'] < dataframe['smoothed_ema_mid'])
                & (dataframe['ohlc4'] < dataframe['smoothed_ema'])
            ), 
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['smoothed_ema_mid'].shift(self.prev_shift) * self.down_ratio.value < dataframe['smoothed_ema_mid'].shift(self.shift_interval))
                | (dataframe['smoothed_ema'] < dataframe['smoothed_ema_mid'])
                | (dataframe['ohlc4'] < dataframe['smoothed_ema_mid'])
            ), 
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['smoothed_ema_mid'].shift(self.prev_shift) > self.down_ratio.value * dataframe['smoothed_ema_mid'].shift(self.shift_interval))
                | (dataframe['smoothed_ema'] > dataframe['smoothed_ema_mid'])
                | (dataframe['ohlc4'] > dataframe['smoothed_ema_mid'])
            ), 
            'exit_short'] = 1

        logger.info(f'{metadata["pair"]}\n{dataframe.tail(3*self.lookback_period)[["date", "close", "ohlc4", "smoothed_ema", "smoothed_ema_mid", "prev_diff_mid", "enter_long", "exit_long", "enter_short", "exit_short"]]}')
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
