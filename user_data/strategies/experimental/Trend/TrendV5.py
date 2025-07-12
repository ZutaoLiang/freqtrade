import numpy as np
import pandas_ta as pta
import pandas as pd
# pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, informative
from freqtrade.persistence import Order, Trade
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
import logging
logger = logging.getLogger(__name__)


class TrendV5(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 10, default=6, space='buy')
    
    use_ha_candles = True

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
    
    def calculate_ha(self, df: DataFrame) -> DataFrame:
        if self.use_ha_candles:
            df_ref = qtpylib.heikinashi(df)
        else:
            df_ref = df
        
        df['ha_open'] = df_ref['open']
        df['ha_high'] = df_ref['high']
        df['ha_low'] = df_ref['low']
        df['ha_close'] = df_ref['close']
        return df
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # haikinashi
        dataframe = self.calculate_ha(dataframe)
        
        # ema
        dataframe['ema_short'] = pta.ema(close=dataframe['ha_close'], length=self.ema_short_len.value, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['ha_close'], length=self.ema_mid_len.value, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ha_close'], length=self.ema_long_len.value, talib=False)
        
        # atr
        dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_period)
        dataframe['natr'] = pta.natr(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], length=self.atr_period)
        
        # ao/ac
        dataframe['ao'] = pta.ao(high=dataframe['ha_high'], low=dataframe['ha_low'])
        dataframe['ac'] = dataframe['ao'] - dataframe['ao'].rolling(window=5).mean()
        
        # volume
        dataframe['obv'] = pta.obv(close=dataframe['ha_close'], volume=dataframe['volume'])
        # dataframe['ad'] = pta.ad(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], volume=dataframe['volume'])
        # dataframe['adosc'] = pta.adosc(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], volume=dataframe['volume'])
        # dataframe['pvol'] = pta.pvol(close=dataframe['ha_close'], volume=dataframe['volume'])
        dataframe['volume_short_mean'] = dataframe['volume'].rolling(self.volume_short).mean()
        dataframe['volume_mid_mean'] = dataframe['volume'].rolling(self.volume_mid).mean()
        
        # cci
        dataframe['cci'] = pta.cci(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], length=self.cci_period)
        
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
                & (dataframe['ha_close'] > dataframe['ema_short'])
                & (dataframe['cci'] > self.cci_threshold)
                & (dataframe['volume_short_mean'] > self.volume_ratio * dataframe['volume_mid_mean'])
                & (self.indicator_up_n_periods_mask(dataframe, 'obv', self.trend_length))
                & (dataframe['ao'] > 0)
                & (self.indicator_up_n_periods_mask(dataframe, 'ao', self.trend_length))
                & (dataframe['ac'] > 0)
                & (self.indicator_up_n_periods_mask(dataframe, 'ac', self.trend_length))
             ), 
            ['enter_long', 'enter_tag']] = (1, 'entry_long')

        dataframe.loc[
            (
                (dataframe['ema_short'] < dataframe['ema_mid'])
                & (dataframe['ha_close'] < dataframe['ema_short'])
                & (dataframe['cci'] < -self.cci_threshold)
                & (dataframe['volume_short_mean'] > self.volume_ratio * dataframe['volume_mid_mean'])
                & (self.indicator_down_n_periods_mask(dataframe, 'obv', self.trend_length))
                & (dataframe['ao'] < 0)
                & (self.indicator_down_n_periods_mask(dataframe, 'ao', self.trend_length))
                & (dataframe['ac'] < 0)
                & (self.indicator_down_n_periods_mask(dataframe, 'ac', self.trend_length))
            ), 
            ['enter_short', 'enter_tag']] = (1, 'entry_short')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        dataframe.loc[
            (
                (dataframe['ema_short'] < dataframe['ema_mid'])
                & (dataframe['ha_close'] < dataframe['ema_short'])
                & (dataframe['cci'] < -self.cci_threshold)
                & (self.indicator_down_n_periods_mask(dataframe, 'obv', self.trend_length))
                & (dataframe['ao'] < 0)
                & (self.indicator_down_n_periods_mask(dataframe, 'ao', self.trend_length))
                & (dataframe['ac'] < 0)
                & (self.indicator_down_n_periods_mask(dataframe, 'ac', self.trend_length))
            ), 
            ['exit_long', 'exit_tag']] = (1, 'exit_signal')
        
        dataframe.loc[
            (
                (dataframe['ema_mid'] < dataframe['ema_long'])
                & (dataframe['ema_short'] < dataframe['ema_long'])
                # & (dataframe['enter_long'] == 0)
            ), 
            ['exit_long', 'exit_tag']] = (1, 'exit_ema')
        
        dataframe.loc[
            (
                (dataframe['ema_short'] > dataframe['ema_mid'])
                & (dataframe['ha_close'] > dataframe['ema_short'])
                & (dataframe['cci'] > self.cci_threshold)
                & (self.indicator_up_n_periods_mask(dataframe, 'obv', self.trend_length))
                & (dataframe['ao'] > 0)
                & (self.indicator_up_n_periods_mask(dataframe, 'ao', self.trend_length))
                & (dataframe['ac'] > 0)
                & (self.indicator_up_n_periods_mask(dataframe, 'ac', self.trend_length))
            ), 
            ['exit_short', 'exit_tag']] = (1, 'exit_signal')
        
        dataframe.loc[
            (
                (dataframe['ema_mid'] > dataframe['ema_long'])
                & (dataframe['ema_short'] > dataframe['ema_long'])
                # & (dataframe['enter_short'] == 0)
            ), 
            ['exit_short', 'exit_tag']] = (1, 'exit_ema')
        
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
            # trailing stop
            return None
        
        open_rate = trade.open_rate
        leverage = trade.leverage
        if current_profit <= 0:
            return None
        
        if trade.is_short:
            max_profit = leverage * (open_rate - trade.min_rate) / open_rate
        else:
            max_profit = leverage * (trade.max_rate - open_rate) / open_rate
        
        open_hours = round((current_time - trade.open_date_utc).total_seconds() / 3600)
        if max_profit > 0.08 * leverage:
            profit_decay_exit = "profit_decay_slow"
            step = round(open_hours / 24)
        else:
            profit_decay_exit = "profit_decay_fast"
            step = round(open_hours / 6)
        
        if step > 1 and current_profit < (max_profit / step):
            return profit_decay_exit
        
        if open_hours >= 12:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or df.empty:
                return None

            trades_df = df.loc[(df['date'] <= current_time) & (df['date'] > trade.open_date_utc)].copy()
            if trades_df.empty:
                return None
            
            total_trades = trades_df.shape[0]
            if trade.is_short:
                negative_trades = trades_df[trades_df['close'] > trade.open_rate].shape[0]
            else:
                negative_trades = trades_df[trades_df['close'] < trade.open_rate].shape[0]

            if negative_trades / total_trades > 0.6 and current_profit > 0.003 * leverage:
                # print(f'time:{current_time},{negative_trades}/{total_trades}\ntrade:{trade}')
                return 'too_much_negative'
        
        return None