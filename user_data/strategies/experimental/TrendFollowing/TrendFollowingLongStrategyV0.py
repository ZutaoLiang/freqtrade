from math import isnan
import numpy as np
import pandas_ta as pta
import pandas as pd
from pandas import DataFrame

from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
from functools import reduce

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_helper import stoploss_from_absolute
from freqtrade.strategy import IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib


class TrendFollowingLongStrategyV0(IStrategy):
    minimal_roi = {"0": 100}
    
    buy_leverage = 3

    timeframe = '5m'

    use_custom_stoploss = True
    stoploss = -0.05 * buy_leverage

    trailing_stop = True
    trailing_stop_positive = 0.06 * buy_leverage
    trailing_stop_positive_offset = 0.07 * buy_leverage
    trailing_only_offset_is_reached = False

    can_short = True

    atr_length = 12
    atr_multiplier = 1

    short_period = 10
    mid_period = 30
    long_period = 60

    adx_length = short_period
    adx_threshold = 20
    
    rsi_length = short_period
    rsi_long_threshold = 65
    rsi_short_threshold = 35
    
    ema_short_length = short_period
    ema_mid_length = mid_period
    ema_long_length = long_period

    startup_candle_count = ema_long_length + max(long_period, atr_length)

    def heikinashi(self, dataframe: DataFrame) -> DataFrame:
        ha = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.heikinashi(dataframe)
        dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_length)
        
        dataframe['volume_ma'] = pta.sma(dataframe['volume'], length=5)

        adx = pta.adx(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.adx_length)
        dataframe['adx'] = adx[f'ADX_{self.adx_length}']

        dataframe['ema_short'] = pta.ema(dataframe['ha_close'], length=self.ema_short_length)

        dataframe['ema_mid'] = pta.ema(dataframe['ha_close'], length=self.ema_mid_length)

        dataframe['ema_long'] = pta.ema(dataframe['ha_close'], length=self.ema_long_length)

        dataframe['recent_high_mid'] = dataframe['ha_close'].rolling(window=self.mid_period).max()
        dataframe['recent_low_mid'] = dataframe['ha_close'].rolling(window=self.mid_period).min()
        dataframe['in_range_mid'] = (dataframe['ha_close'] <= dataframe['recent_high_mid'].shift(1)) & \
                                    (dataframe['ha_close'] >= dataframe['recent_low_mid'].shift(1))

        dataframe['rsi'] = pta.rsi(dataframe['ha_close'], length=self.rsi_length, talib=False)
        dataframe['is_bullish'] = dataframe['ha_close'] > dataframe['ha_open']
        dataframe['bullish_count'] = dataframe['is_bullish'].rolling(window=self.short_period, min_periods=self.short_period).sum()

        dataframe['is_bearish'] = dataframe['ha_close'] < dataframe['ha_open']
        dataframe['bearish_count'] = dataframe['is_bearish'].rolling(window=self.short_period, min_periods=self.short_period).sum()
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        long_entry_conditions = []
        long_entry_conditions.append(dataframe['volume_ma'] >= dataframe['volume_ma'].shift(1))
        long_entry_conditions.append(dataframe['ha_close'] >= dataframe['ema_short'])
        long_entry_conditions.append(dataframe['ema_short'] >= dataframe['ema_mid'])
        long_entry_conditions.append(dataframe['ema_mid'] >= dataframe['ema_long'])
        long_entry_conditions.append(dataframe['ema_short'] >= dataframe['ema_short'].shift(1))
        long_entry_conditions.append(dataframe['ema_mid'] >= dataframe['ema_mid'].shift(1))
        long_entry_conditions.append(dataframe['ema_long'] >= dataframe['ema_long'].shift(1))
        long_entry_conditions.append(dataframe['ha_close'] >= dataframe['recent_high_mid'].shift(1))
        long_entry_conditions.append(dataframe['adx'] >= self.adx_threshold)
        long_entry_conditions.append(dataframe['rsi'] >= self.rsi_long_threshold)
        long_entry_conditions.append(dataframe['bullish_count'] >= (int)(self.short_period * 0.6))
        
        dataframe.loc[
            (reduce(lambda x, y: x & y, long_entry_conditions))
        , 'enter_long'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        dataframe.loc[
            dataframe['ema_mid'] < dataframe['ema_long'],
            ['exit_long', 'exit_tag']
        ] = (1, 'ema_exit')

        dataframe.loc[
            dataframe['bearish_count'] >= (int)(self.short_period * 0.9),
            ['exit_long', 'exit_tag']
        ] = (1, 'bearish_exit')

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        # dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # if dataframe is None or dataframe.empty:
        #     return None

        # last_candle = dataframe.loc[dataframe['date'] <= current_time]
        # if last_candle.empty:
        #     return None
        
        if current_time - timedelta(minutes=30) > trade.open_date_utc and current_profit < 0.01:
            return stoploss_from_absolute(stop_rate=current_rate, current_rate=current_rate, is_short=trade.is_short,
                                      leverage=trade.leverage)
        return self.trailing_stop_positive
    
        # last_candle = last_candle.iloc[-1]
        # atr = last_candle['atr'] # 实际策略运行过程中当前K线可能不完整，对应计算的ATR可能不正确，这里采用倒数第二根K线的计算值

        # min_rate = trade.min_rate if trade.min_rate is not None else trade.open_rate
        # max_rate = trade.max_rate if trade.max_rate is not None else trade.open_rate
        
        # stop_loss_price = None
        # if trade.is_short:
        #     stop_loss_price = min_rate + (self.atr_multiplier * atr)
        # else:
        #     stop_loss_price = max_rate - (self.atr_multiplier * atr)

        # return stoploss_from_absolute(stop_rate=stop_loss_price, current_rate=current_rate, is_short=trade.is_short,
        #                               leverage=trade.leverage)
 
    @property
    def protections(self):  # type: ignore
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.mid_period,
                "trade_limit": 2,
                "stop_duration_candles": 2,
                "required_profit": 0.0,
                "only_per_pair": True,
                "only_per_side": False
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.buy_leverage
