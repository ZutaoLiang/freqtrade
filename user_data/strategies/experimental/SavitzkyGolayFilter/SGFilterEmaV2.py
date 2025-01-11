from scipy.signal import savgol_filter
from math import isnan
import numpy as np
import pandas_ta as pta
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, informative
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Order, Trade
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
import logging
logger = logging.getLogger(__name__)


class SGFilterEmaV2(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.08
    stoploss = -base_stop_loss * buy_leverage.value

    trailing_stop = True
    trailing_stop_positive = 0.1 * buy_leverage.value
    trailing_stop_positive_offset = 0
    trailing_only_offset_is_reached = False

    can_short = True
 
    timeframe = '15m'

    lookback_period = 10
    
    window_length = IntParameter(10, 100, default=lookback_period, space='buy')
    polyorder = IntParameter(1, 5, default=1, space='fixed')

    ema_period = IntParameter(5, 100, default=lookback_period, space='buy')
    ema_mid_period = IntParameter(5, 100, default=lookback_period * 3, space='buy')

    startup_candle_count = int(max(window_length.value, ema_mid_period.value) * 1.2)
    
    up_ratio = DecimalParameter(1.0001, 1.0010, default=1.0008, decimals=5, space='buy')
    down_ratio = DecimalParameter(1.0001, 1.0010, default=1.0001, decimals=5, space='buy')
    
    highest_period = lookback_period
    lowest_period = lookback_period
    
    atr_period = ema_mid_period.value
    risk_ratio = 0.002

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
        
        dataframe['prev_diff'] = dataframe['smoothed_ema'] / dataframe['smoothed_ema'].shift(1)
        dataframe['prev_diff_mid'] = dataframe['smoothed_ema_mid'] / dataframe['smoothed_ema_mid'].shift(1)
        
        # dataframe['price_percentile'] = dataframe['ohlc4'].rolling(window=self.lookback_period * 6).apply(
        #     lambda x: percentileofscore(x, x.iloc[-1])
        # )
        
        dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_period)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['smoothed_ema_mid'] > self.up_ratio.value * dataframe['smoothed_ema_mid'].shift(1))
                & (dataframe['smoothed_ema'] > dataframe['smoothed_ema_mid'])
                & (dataframe['ohlc4'] > dataframe['smoothed_ema'])
                # & (dataframe['ohlc4'] > dataframe['highest'].shift(1))
             ), 
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['smoothed_ema_mid'] * self.up_ratio.value < dataframe['smoothed_ema_mid'].shift(1))
                & (dataframe['smoothed_ema'] < dataframe['smoothed_ema_mid'])
                & (dataframe['ohlc4'] < dataframe['smoothed_ema'])
                # & (dataframe['ohlc4'] < dataframe['lowest'].shift(1))
            ), 
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['smoothed_ema_mid'] * self.down_ratio.value < dataframe['smoothed_ema_mid'].shift(1))
                | (dataframe['smoothed_ema'] < dataframe['smoothed_ema_mid'])
                | (dataframe['ohlc4'] < dataframe['smoothed_ema_mid'])
            ), 
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['smoothed_ema_mid'] > self.down_ratio.value * dataframe['smoothed_ema_mid'].shift(1))
                | (dataframe['smoothed_ema'] > dataframe['smoothed_ema_mid'])
                | (dataframe['ohlc4'] > dataframe['smoothed_ema_mid'])
            ), 
            'exit_short'] = 1

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
        # stake_amount = min(max_stake, stake_amount)
        # if min_stake is not None:
        #     stake_amount = max(min_stake, stake_amount)
            
        logger.info(f'Custom stake amount:{stake_amount} for {pair}, proposed stake:{proposed_stake}, atr:{atr}, current_rate:{current_rate}, balance:{balance}, ' \
                    f'risk_amount:{risk_amount}, min_stake:{min_stake}, max_stake:{max_stake}, laverage:{leverage}')
        return stake_amount
 
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
