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


class ZLSMAStrategy(IStrategy):
    minimal_roi = {"0": 100}
    
    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    timeframe = '5m'
    
    stoploss = -0.2

    trailing_stop = True
    trailing_stop_positive = 0.2
    trailing_stop_positive_offset = 0
    trailing_only_offset_is_reached = False
    
    can_short = True

    atr_length = 10
    atr_multiplier = 2.0
    zlsma_length = 30
    sg_filter_length = 10
    up_ratio = DecimalParameter(1.0001, 1.0010, default=1.0008, decimals=5, space='buy')
    down_ratio = DecimalParameter(1.0001, 1.0010, default=1.0001, decimals=5, space='buy')
    
    risk_ratio = 0.002
    
    startup_candle_count = zlsma_length

    def savgol_smooth(self, data):
        smoothed_data = savgol_filter(data, self.sg_filter_length, 1, mode='nearest')
        return smoothed_data
 
    def heikinashi(self, dataframe: DataFrame) -> DataFrame:
        ha = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.heikinashi(dataframe)
        dataframe['zlsma'] = pta.zlma(dataframe['ha_close'], length=self.zlsma_length)
        dataframe['smoothed_ma'] = self.savgol_smooth(dataframe['zlsma'].values)
        
        dataframe['prev_diff'] = dataframe['smoothed_ma'] / dataframe['smoothed_ma'].shift(1)
        
        dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_length)
        
        # df_supertrend = pta.supertrend(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], 
        #                                length=self.atr_length, multiplier=self.atr_multiplier)
        # dataframe['supertrend_upper'] = df_supertrend[f'SUPERTl_{self.atr_length}_{self.atr_multiplier}'] # type: ignore
        # dataframe['supertrend_lower'] = df_supertrend[f'SUPERTs_{self.atr_length}_{self.atr_multiplier}'] # type: ignore
        # dataframe['supertrend_trend'] = df_supertrend[f'SUPERT_{self.atr_length}_{self.atr_multiplier}'] # type: ignore
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_condition = (
            (dataframe['ha_close'] > dataframe['zlsma'])
            & (dataframe['smoothed_ma'] > self.up_ratio.value * dataframe['smoothed_ma'].shift(1))
            # & (dataframe['ha_close'].shift(1) <= dataframe['zlsma'].shift(1)) &
            # & (dataframe['ha_close'] > dataframe['supertrend_upper'])
        )
        
        dataframe.loc[long_condition, 'enter_long'] = 1

        short_condition = (
            (dataframe['ha_close'] < dataframe['zlsma'])
            & (dataframe['smoothed_ma'] * self.up_ratio.value < dataframe['smoothed_ma'].shift(1))
            # & (dataframe['ha_close'] < dataframe['supertrend_lower'])
            # & (dataframe['ha_close'].shift(1) >= dataframe['zlsma'].shift(1)) &
        )
        
        dataframe.loc[short_condition, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_exit_condition = (
            (
                (dataframe['ha_close'] < dataframe['zlsma']) 
                # & (dataframe['ha_close'].shift(1) >= dataframe['zlsma'].shift(1))
            ) 
            | (dataframe['smoothed_ma'] * self.down_ratio.value < dataframe['smoothed_ma'].shift(1))
            # & (dataframe['ha_close'] < dataframe['supertrend_upper'])
        )
        
        dataframe.loc[long_exit_condition, 'exit_long'] = 1

        short_exit_condition = (
            (
                (dataframe['ha_close'] > dataframe['zlsma'])
                # & (dataframe['ha_close'].shift(1) <= dataframe['zlsma'].shift(1))
            ) 
            | (dataframe['smoothed_ma'] > self.down_ratio.value * dataframe['smoothed_ma'].shift(1))
            # & (dataframe['ha_close'] > dataframe['supertrend_lower'])
        )
        
        dataframe.loc[short_exit_condition, 'exit_short'] = 1

        return dataframe

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
            
        # logger.info(f'Custom stake amount:{stake_amount} for {pair}, proposed stake:{proposed_stake}, atr:{atr}, current_rate:{current_rate}, balance:{balance}, ' \
                    # f'risk_amount:{risk_amount}, min_stake:{min_stake}, max_stake:{max_stake}, laverage:{leverage}')
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
    