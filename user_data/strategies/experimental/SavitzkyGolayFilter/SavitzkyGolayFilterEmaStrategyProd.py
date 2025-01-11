from scipy.signal import savgol_filter
from scipy.stats import linregress, percentileofscore
from freqtrade.strategy.interface import IStrategy
import numpy as np
import pandas_ta as pta
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IntParameter, DecimalParameter, informative
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from typing import Optional, Tuple, Union


class SavitzkyGolayFilterEmaStrategyProd(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.1
    stoploss = -base_stop_loss * buy_leverage.value

    trailing_stop = True
    trailing_stop_positive = 0.1 * buy_leverage.value
    trailing_stop_positive_offset = 0
    # trailing_stop_positive_offset = 0.08 * buy_leverage.value
    trailing_only_offset_is_reached = False

    can_short = True
 
    timeframe = '15m'

    lookback_period = 10
    
    window_length = IntParameter(10, 100, default=lookback_period, space='buy')
    polyorder = IntParameter(1, 5, default=1, space='fixed')

    ema_period = IntParameter(5, 100, default=lookback_period, space='buy')
    ema_mid_period = IntParameter(5, 100, default=lookback_period * 3, space='buy')
    # ema_long_period = IntParameter(5, 100, default=lookback_period * 6, space='buy')

    startup_candle_count = int(max(window_length.value, ema_mid_period.value) * 1.2)
    
    up_ratio = DecimalParameter(1.0001, 1.0010, default=1.0015, decimals=5, space='buy')
    down_ratio = DecimalParameter(1.0001, 1.0010, default=1.0001, decimals=5, space='buy')
    adx_threshold = IntParameter(20, 50, default=20, space='buy')
    
    highest_period = lookback_period
    lowest_period = lookback_period

    def savgol_smooth(self, data):
        smoothed_data = savgol_filter(data, self.window_length.value, self.polyorder.value, mode='nearest')
        return smoothed_data

    # @informative('15m')
    # @informative('30m')
    # def populate_indicators_other_timeframe(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
    #     return dataframe
    
    def linear_regression_slope(self, dataframe: DataFrame, period: int) -> pd.Series:
        return dataframe['ohlc4'].rolling(window=period).apply(
            lambda x: linregress(range(len(x)), x)[0], raw=True
        )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['ema'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_period.value, talib=False)
        dataframe['smoothed_ema'] = self.savgol_smooth(dataframe['ema'].values)

        dataframe['ema_mid'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_mid_period.value, talib=False)
        dataframe['smoothed_ema_mid'] = self.savgol_smooth(dataframe['ema_mid'].values)
        
        # dataframe['ema_long'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_long_period.value, talib=False)
        # dataframe['smoothed_ema_long'] = self.savgol_smooth(dataframe['ema_long'].values)
        
        # dataframe['adx'] = pta.adx(high=dataframe['high'], low=dataframe['low'], close=dataframe['ohlc4'], length=self.ema_period.value)[f'ADX_{self.ema_period.value}']

        dataframe['highest'] = dataframe['ohlc4'].rolling(window=self.highest_period).max()
        dataframe['lowest'] = dataframe['ohlc4'].rolling(window=self.lowest_period).min()
        
        dataframe['prev_diff'] = dataframe['smoothed_ema'] / dataframe['smoothed_ema'].shift(1)
        dataframe['prev_diff_mid'] = dataframe['smoothed_ema_mid'] / dataframe['smoothed_ema_mid'].shift(1)
        # dataframe['prev_diff_long'] = dataframe['smoothed_ema_long'] / dataframe['smoothed_ema_long'].shift(1)

        # dataframe['price_ma_diff'] = dataframe['ohlc4'] - dataframe['smoothed_ema']
        # dataframe['cum_price_ma_diff'] = dataframe['price_ma_diff'].rolling(window=self.lookback_period).sum()
        # dataframe['cum_diff_percent'] = dataframe['cum_price_ma_diff'] / dataframe['smoothed_ema'] * 100
        
        dataframe['lr_slope'] = self.linear_regression_slope(dataframe, period=self.lookback_period)
        
        dataframe['price_percentile'] = dataframe['ohlc4'].rolling(window=self.lookback_period * 6).apply(
            lambda x: percentileofscore(x, x.iloc[-1])
        )
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # (dataframe['smoothed_ema'] > self.up_ratio.value * dataframe['smoothed_ema'].shift(1))
                # & (dataframe['smoothed_ema'].shift(1) > self.up_ratio.value * dataframe['smoothed_ema'].shift(2))
                (dataframe['smoothed_ema_mid'] > self.up_ratio.value * dataframe['smoothed_ema_mid'].shift(1))
                & (dataframe['smoothed_ema'] > dataframe['smoothed_ema_mid'])
                & (dataframe['ohlc4'] > dataframe['smoothed_ema'])
                & (dataframe['ohlc4'] > dataframe['highest'].shift(1))
                # & (dataframe['adx'] > self.adx_threshold.value)
             ), 
            'enter_long'] = 1

        dataframe.loc[
            (
                # (dataframe['smoothed_ema'] * self.up_ratio.value < dataframe['smoothed_ema'].shift(1))
                # & (dataframe['smoothed_ema'].shift(1) * self.up_ratio.value < dataframe['smoothed_ema'].shift(2))
                (dataframe['smoothed_ema_mid'] * self.up_ratio.value < dataframe['smoothed_ema_mid'].shift(1))
                & (dataframe['smoothed_ema'] < dataframe['smoothed_ema_mid'])
                & (dataframe['ohlc4'] < dataframe['smoothed_ema'])
                & (dataframe['ohlc4'] < dataframe['lowest'].shift(1))
                # & (dataframe['adx'] > self.adx_threshold.value)
            ), 
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # (dataframe['smoothed_ema'] * self.down_ratio.value < dataframe['smoothed_ema'].shift(1))
                # & (dataframe['smoothed_ema'].shift(1) * self.down_ratio.value < dataframe['smoothed_ema'].shift(2))                
                (dataframe['smoothed_ema_mid'] * self.down_ratio.value < dataframe['smoothed_ema_mid'].shift(1))
                | (dataframe['smoothed_ema'] < dataframe['smoothed_ema_mid'])
                | (dataframe['ohlc4'] < dataframe['smoothed_ema_mid'])
            ), 
            'exit_long'] = 1

        dataframe.loc[
            (
                # (dataframe['chandelier_exit'] < dataframe['ohlc4'])
                # (dataframe['smoothed_ema'] > self.down_ratio.value * dataframe['smoothed_ema'].shift(1))
                # & (dataframe['smoothed_ema'].shift(1) > self.down_ratio.value * dataframe['smoothed_ema'].shift(2))
                (dataframe['smoothed_ema_mid'] > self.down_ratio.value * dataframe['smoothed_ema_mid'].shift(1))
                | (dataframe['smoothed_ema'] > dataframe['smoothed_ema_mid'])
                | (dataframe['ohlc4'] > dataframe['smoothed_ema_mid'])
            ), 
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
