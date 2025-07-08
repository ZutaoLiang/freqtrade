import numpy as np
import pandas_ta as pta
import pandas as pd

from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, informative
from freqtrade.persistence import Order, Trade
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
import logging
logger = logging.getLogger(__name__)


class TrendV4(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 5, default=5, space='buy')

    base_stop_loss = 0.1
    stoploss = -base_stop_loss * buy_leverage.value

    trailing_stop = True
    trailing_stop_positive = 0.12 * buy_leverage.value
    trailing_stop_positive_offset = 0.15 * buy_leverage.value
    trailing_only_offset_is_reached = True

    can_short = True
 
    timeframe = '30m'

    lookback_period = 12
    
    window_length = IntParameter(10, 100, default=lookback_period, space='buy')

    ema_short_len = IntParameter(5, 100, default=lookback_period, space='buy')
    ema_mid_len = IntParameter(5, 100, default=lookback_period * 3, space='buy')
    ema_long_len = IntParameter(5, 100, default=lookback_period * 6, space='buy')

    startup_candle_count = int(max(window_length.value, ema_long_len.value) * 1.2)

    atr_period = 21
    cci_period = 21
    cci_threshold = 80
    
    volume_short = 2
    volume_mid = 20
    volume_ratio = 2
    
    trend_length = 3
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ema
        dataframe['ema_short'] = pta.ema(close=dataframe['close'], length=self.ema_short_len.value, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['close'], length=self.ema_mid_len.value, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['close'], length=self.ema_long_len.value, talib=False)
        
        # atr
        dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_period)
        dataframe['natr'] = pta.natr(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], length=self.atr_period)
        
        # ao/ac
        dataframe['ao'] = pta.ao(high=dataframe['high'], low=dataframe['low'])
        dataframe['ac'] = dataframe['ao'] - dataframe['ao'].rolling(window=5).mean()
        
        # volume
        dataframe['obv'] = pta.obv(close=dataframe['close'], volume=dataframe['volume'])
        # dataframe['ad'] = pta.ad(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], volume=dataframe['volume'])
        # dataframe['adosc'] = pta.adosc(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], volume=dataframe['volume'])
        # dataframe['pvol'] = pta.pvol(close=dataframe['close'], volume=dataframe['volume'])
        dataframe['volume_short_mean'] = dataframe['volume'].rolling(self.volume_short).mean()
        dataframe['volume_mid_mean'] = dataframe['volume'].rolling(self.volume_mid).mean()
        
        # cci
        dataframe['cci'] = pta.cci(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], length=self.cci_period)
        
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
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        dataframe.loc[
            (
                (dataframe['ema_short'] > dataframe['ema_mid'])
                & (dataframe['close'] > dataframe['ema_short'])
                & (dataframe['cci'] > self.cci_threshold)
                & (dataframe['volume_short_mean'] > self.volume_ratio * dataframe['volume_mid_mean'])
                & (self.indicator_up_n_periods_mask(dataframe, 'obv', self.trend_length))
                & (dataframe['ao'] > 0)
                & (self.indicator_up_n_periods_mask(dataframe, 'ao', self.trend_length))
                & (dataframe['ac'] > 0)
                & (self.indicator_up_n_periods_mask(dataframe, 'ac', self.trend_length))
             ), 
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['ema_short'] < dataframe['ema_mid'])
                & (dataframe['close'] < dataframe['ema_short'])
                & (dataframe['cci'] < -self.cci_threshold)
                & (dataframe['volume_short_mean'] > self.volume_ratio * dataframe['volume_mid_mean'])
                & (self.indicator_down_n_periods_mask(dataframe, 'obv', self.trend_length))
                & (dataframe['ao'] < 0)
                & (self.indicator_down_n_periods_mask(dataframe, 'ao', self.trend_length))
                & (dataframe['ac'] < 0)
                & (self.indicator_down_n_periods_mask(dataframe, 'ac', self.trend_length))
            ), 
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['ema_short'] < dataframe['ema_mid'])
                    & (dataframe['close'] < dataframe['ema_short'])
                    & (dataframe['cci'] < -self.cci_threshold)
                    & (self.indicator_down_n_periods_mask(dataframe, 'obv', self.trend_length))
                    & (dataframe['ao'] < 0)
                    & (self.indicator_down_n_periods_mask(dataframe, 'ao', self.trend_length))
                    & (dataframe['ac'] < 0)
                    & (self.indicator_down_n_periods_mask(dataframe, 'ac', self.trend_length))
                )
            | 
                (
                    # (dataframe['enter_long'] == 0)
                    # & 
                    (dataframe['ema_mid'] < dataframe['ema_long'])
                )
            ), 
            'exit_long'] = 1

        dataframe.loc[
            (
                (
                    (dataframe['ema_short'] > dataframe['ema_mid'])
                    & (dataframe['close'] > dataframe['ema_short'])
                    & (dataframe['cci'] > self.cci_threshold)
                    & (self.indicator_up_n_periods_mask(dataframe, 'obv', self.trend_length))
                    & (dataframe['ao'] > 0)
                    & (self.indicator_up_n_periods_mask(dataframe, 'ao', self.trend_length))
                    & (dataframe['ac'] > 0)
                    & (self.indicator_up_n_periods_mask(dataframe, 'ac', self.trend_length))
                )
            | 
                (
                    # (dataframe['enter_short'] == 0)
                    # & 
                    (dataframe['ema_mid'] > dataframe['ema_long'])
                )
            ), 
            'exit_short'] = 1

        return dataframe

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

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        if current_profit > self.trailing_stop_positive_offset:
            # profit is enough to activate trailing stop
            return None
        
        open_rate = trade.open_rate
        leverage = trade.leverage
        if current_profit <= 0:
            return None
        
        if trade.is_short:
            max_profit = leverage * (open_rate - trade.min_rate) / open_rate
        else:
            max_profit = leverage * (trade.max_rate - open_rate) / open_rate
        
        diff_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        if max_profit > 0.08 * leverage:
            step = round(diff_hours / 24)
        else:
            step = round(diff_hours / 6)
        
        if step > 1 and current_profit < (max_profit / step):
            return 'time_decay'
        
        return None