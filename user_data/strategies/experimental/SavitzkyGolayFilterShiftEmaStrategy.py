import numpy as np
from scipy.signal import savgol_filter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IntParameter
from datetime import datetime
from typing import Optional, Tuple, Union


class SavitzkyGolayFilterShiftEmaStrategy(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.15
    
    # Stoploss:
    stoploss = -base_stop_loss * buy_leverage.value

    can_short = True
 
    timeframe = '1h'
    
    window_length = IntParameter(10, 100, default=10, space='buy')
    polyorder = IntParameter(1, 5, default=1, space='fixed')

    ema_period = IntParameter(5, 100, default=10, space='buy')

    startup_candle_count = max(window_length.value, ema_period.value) * 2

    def savgol_smooth(self, data):
        smoothed_data = savgol_filter(data, self.window_length.value, self.polyorder.value) # , mode='nearest'
        return smoothed_data

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.ema_period.value)

        # 将 EMA 向后移动 window_length 的长度
        dataframe['shifted_ema'] = dataframe['ema'].shift(self.window_length.value)
        
        # 基于移动后的 EMA 计算 smoothed_ema
        smoothed_values = self.savgol_smooth(dataframe['shifted_ema'].values)
        
        # 用 NaN 填充前面空出的部分，保持与原数据长度一致
        dataframe['smoothed_ema'] = np.concatenate([np.full(self.window_length.value, np.nan), smoothed_values[self.window_length.value:]])
        
        # dataframe['filtered_ema'] = self.savgol_smooth(dataframe['ema'].values)
        # dataframe['smoothed_ema'] = dataframe['filtered_ema'].shift(-self.window_length.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['smoothed_ema'] > dataframe['smoothed_ema'].shift(1))
                & (dataframe['smoothed_ema'].shift(1) > dataframe['smoothed_ema'].shift(2))
             ), 
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['smoothed_ema'] < dataframe['smoothed_ema'].shift(1))
                & (dataframe['smoothed_ema'].shift(1) < dataframe['smoothed_ema'].shift(2))
            ), 
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['smoothed_ema'] < dataframe['smoothed_ema'].shift(1))
                & (dataframe['smoothed_ema'].shift(1) < dataframe['smoothed_ema'].shift(2))
            ), 
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['smoothed_ema'] > dataframe['smoothed_ema'].shift(1))
                & (dataframe['smoothed_ema'].shift(1) > dataframe['smoothed_ema'].shift(2))
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

