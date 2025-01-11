import pandas as pd
import numpy as np
import pandas_ta as pta
from scipy.signal import savgol_filter
from freqtrade.strategy import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
from math import isnan

from pandas import DataFrame
import logging
logger = logging.getLogger(__name__)


class SarStrategy(IStrategy):
    # common 
    minimal_roi = {"0": 100}
    
    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    timeframe = '5m'
    
    stoploss = -0.2

    trailing_stop = True
    trailing_stop_positive = 0.15
    trailing_stop_positive_offset = 0
    trailing_only_offset_is_reached = False
    
    can_short = True
    
    atr_length = 12
    atr_multiplier = 1.5
    smooth_length = 5
    
    risk_ratio = 0.002

    # strategy
    sar_af0_1 = DecimalParameter(0.01, 0.1, default=0.01, space='buy')
    sar_af1 = DecimalParameter(0.01, 0.1, default=0.01, space='buy')
    sar_max_af1 = DecimalParameter(0.1, 0.5, default=0.2, space='buy')
    
    sar_af0_2 = DecimalParameter(0.01, 0.1, default=0.002, space='buy')
    sar_af2 = DecimalParameter(0.01, 0.1, default=0.002, space='buy')
    sar_max_af2 = DecimalParameter(0.1, 0.5, default=0.1, space='buy')
    
    # zlsma_length = 45
    # up_ratio = DecimalParameter(1.0001, 1.0010, default=1.0007, decimals=5, space='buy')
    # down_ratio = DecimalParameter(0.9991, 1.0010, default=0.9998, decimals=5, space='buy')    
    
    startup_candle_count = 100
 
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
        dataframe['ema'] = pta.ema(dataframe['ha_close'], length=self.smooth_length)
        dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_length)
        dataframe['lower'] = dataframe['ema'] - self.atr_multiplier * dataframe['atr']
        dataframe['upper'] = dataframe['ema'] + self.atr_multiplier * dataframe['atr']
        
        dataframe['median_short'] = dataframe['ema'].rolling(window=10).median()
        dataframe['median_middle'] = dataframe['ema'].rolling(window=20).median()
        dataframe['median_long'] = dataframe['ema'].rolling(window=40).median()
        
        # strategy
        dataframe['is_bullish'] = dataframe['close'] > dataframe['open']
        dataframe['is_bearish'] = dataframe['close'] < dataframe['open']
        
        psar1 = pta.psar(high=dataframe['ema'], low=dataframe['ema'], close=dataframe['ema'],
            af0=self.sar_af0_1.value, af=self.sar_af1.value, max_af=self.sar_max_af1.value)
        psar_long_col_1 = [col for col in psar1.columns if 'PSARl' in col][0]
        psar_short_col_1 = [col for col in psar1.columns if 'PSARs' in col][0]
        dataframe['psar_long_1'] = psar1[psar_long_col_1]
        dataframe['psar_short_1'] = psar1[psar_short_col_1]

        psar2 = pta.psar(high=dataframe['ema'], low=dataframe['ema'], close=dataframe['ema'],
            af0=self.sar_af0_2.value, af=self.sar_af2.value, max_af=self.sar_max_af2.value)
        psar_long_col_2 = [col for col in psar2.columns if 'PSARl' in col][0]
        psar_short_col_2 = [col for col in psar2.columns if 'PSARs' in col][0]
        dataframe['psar_long_2'] = psar2[psar_long_col_2]
        dataframe['psar_short_2'] = psar2[psar_short_col_2]
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['psar_long_1'].notna()
                & dataframe['psar_long_2'].notna()
                # (dataframe['psar_reversal'] == 1)
                # & 
                # (dataframe['close'] > dataframe['sar']) &  # 当前收盘价高于SAR
                # (dataframe['close'].shift(1) <= dataframe['sar'].shift(1))  # 前一根K线收盘价低于或等于SAR
            ), 
            'enter_long'] = 1

        dataframe.loc[
            (
                dataframe['psar_short_1'].notna()
                & dataframe['psar_short_2'].notna()
                # (dataframe['close'] < dataframe['sar']) &  # 当前收盘价低于SAR
                # (dataframe['close'].shift(1) >= dataframe['sar'].shift(1))  # 前一根K线收盘价高于或等于SAR
            ), 
            'enter_short'] = 1

        return dataframe
    
        # dataframe.loc[
        #     (
        #         (dataframe['zlsma'] > dataframe['smoothed_ma'])
        #         & (dataframe['smoothed_ma'] > self.up_ratio.value * dataframe['smoothed_ma'].shift(1))
        #         & (dataframe['is_bullish'].shift(1))
        #         & (dataframe['is_bullish'].shift(2))
        #     ), 'enter_long'] = 1

        # dataframe.loc[
        #     (
        #         (dataframe['zlsma'] < dataframe['smoothed_ma'])
        #         & (dataframe['smoothed_ma'] * self.up_ratio.value < dataframe['smoothed_ma'].shift(1))
        #         & (dataframe['is_bearish'].shift(1))
        #         & (dataframe['is_bearish'].shift(2))
        #     ), 'enter_short'] = 1

        # return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['psar_short_1'].notna()
                & dataframe['psar_short_2'].notna()
                )
                | (dataframe['ha_close'] < dataframe['lower'])
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['psar_long_1'].notna()
                & dataframe['psar_long_2'].notna()
                )
                | (dataframe['ha_close'] > dataframe['upper'])
            ),
            'exit_short'] = 1

        return dataframe
    
        # dataframe.loc[
        #     (
        #         (dataframe['ha_close'] < dataframe['lower'])
        #         | (dataframe['zlsma'] < dataframe['smoothed_ma'])
        #         | (dataframe['smoothed_ma'] < dataframe['smoothed_ma'].shift(1) * self.down_ratio.value)
        #     )
        #     , 'exit_long'] = 1
        
        # dataframe.loc[
        #     (
        #         (dataframe['ha_close'] > dataframe['upper'])
        #         | (dataframe['zlsma'] > dataframe['smoothed_ma'])
        #         | (dataframe['smoothed_ma'] * self.down_ratio.value > dataframe['smoothed_ma'].shift(1))
        #     ), 'exit_short'] = 1

        # return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        stake_amount = proposed_stake
        if self.wallets is None or self.risk_ratio <= 0:
            return stake_amount

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return stake_amount
        
        last_candle = dataframe.loc[dataframe['date'] <= current_time]
        if last_candle.empty:
            return stake_amount
        
        last_candle = last_candle.iloc[-1]
        atr = last_candle['atr'] # 实际策略运行过程中当前K线可能不完整，对应计算的ATR可能不正确，这里采用倒数第二根K线的计算值
        if isnan(atr):
            return stake_amount
        
        balance = self.wallets.get_total(self.stake_currency)
        risk_amount = (self.risk_ratio * balance / atr) * current_rate
        
        stake_amount = risk_amount
            
        return stake_amount
     
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
    