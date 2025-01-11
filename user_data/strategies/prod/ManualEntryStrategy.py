from math import isnan
import numpy as np
import pandas_ta as pta
import pandas as pd
from pandas import DataFrame
from functools import reduce

from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

from freqtrade.strategy.interface import IStrategy, Trade
from freqtrade.strategy.strategy_helper import stoploss_from_absolute, stoploss_from_open
from freqtrade.strategy import IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib

import logging
logger = logging.getLogger(__name__)


class ManualEntryStrategy(IStrategy):
    
    # common
    minimal_roi = {"0": 100}
    
    trade_leverage = 1

    timeframe = '5m'
    
    use_custom_stoploss = True
    stoploss = -0.07 * trade_leverage

    trailing_stop = False
    
    can_short = True
    position_adjustment_enable = True
    
    atr_length = 14
    atr_multiplier = 1.5
 
    # strategy
    short_period = 14
    mid_period = 30
    long_period = 60

    adx_length = short_period
    adx_threshold = 30
    
    rsi_length = short_period
    rsi_long_threshold = 65
    rsi_short_threshold = 35
    
    ema_short_length = short_period
    ema_mid_length = mid_period
    ema_long_length = long_period
        
    startup_candle_count = ema_long_length
    
    enable_logging = False

    position_adjustment_enable = True
    max_additional_positions = 3
    first_profit_threshold = 0.15
    second_profit_threshold = 0.3
    third_profit_threshold = 0.4
    position_addition_dict = {}
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float, min_stake: float, max_stake: float, 
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        if not self.position_adjustment_enable:
            return None

        filled_entries = trade.select_filled_orders()
        count_of_entries = len(filled_entries)
        if count_of_entries == 0:
            return None
        
        if count_of_entries > self.max_additional_positions:
            return None
            
        leverage = trade.leverage
        current_profit_threshold = current_profit / leverage
        
        stake_amount = filled_entries[0].cost
        
        pair = trade.pair
        if pair not in self.position_addition_dict.keys():
            self.position_addition_dict[pair] = 0
        
        addition_count = self.position_addition_dict.get(pair)
        if addition_count > self.max_additional_positions:
            return None
        
        if (addition_count == 0 and 
            current_profit_threshold >= self.first_profit_threshold and 
            current_profit_threshold < self.second_profit_threshold):
            # if self.enable_logging:
            logger.info(f'First position addition for {trade.pair} triggered({current_time}) at profit: {current_profit_threshold}')
            self.position_addition_dict[pair] += 1
            return stake_amount
            
        if (addition_count == 1 and 
            current_profit_threshold >= self.second_profit_threshold and 
            current_profit_threshold < self.third_profit_threshold):
            # if self.enable_logging:
            logger.info(f'Second position addition for {trade.pair} triggered({current_time}) at profit: {current_profit_threshold}')
            self.position_addition_dict[pair] += 1
            return stake_amount
            
        # Third position addition
        if (addition_count == 2 and 
            current_profit_threshold >= self.third_profit_threshold):
            # if self.enable_logging:
            logger.info(f'Third position addition for {trade.pair} triggered({current_time}) at profit: {current_profit_threshold}')
            self.position_addition_dict[pair] += 1
            return stake_amount
            
        return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        leverage = trade.leverage

        # 1. 当前处于高利润状态，持仓时间久一点，除非遇到较大的回撤
        if current_profit > 0.32 * leverage:
            return 0.15 * leverage

        if current_profit > 0.16 * leverage:
            return stoploss_from_open(0.7 * 0.16 * leverage, current_profit, is_short=trade.is_short, leverage=leverage)

        if current_profit > 0.08 * leverage:
            # 固定价格止损，至少拿到价格涨幅4个点的利润
            return stoploss_from_open(0.5 * 0.08 * leverage, current_profit, is_short=trade.is_short, leverage=leverage)

        if current_profit > 0.04 * leverage:
            # 固定价格止损，至少拿到价格涨幅2个点的利润
            return stoploss_from_open(0.5 * 0.04 * leverage, current_profit, is_short=trade.is_short, leverage=leverage)

        # 2. 处理长时间持仓没有达到目标涨幅的情况
        long_time = 15
        if (current_time - timedelta(minutes=long_time)) > trade.open_date_utc:
            if current_profit < 0.02 * leverage:
                # 已经低利润或者是亏损状态，扛到亏损
                open_relative = 0.5 * 0.01 * leverage
                if self.enable_logging:
                    logger.info(f'Holding {pair} over {long_time} minutes with low profit:{current_profit}, set a new stoploss:{open_relative} relative to open')
                # return stoploss_from_open(open_relative, current_profit, is_short=trade.is_short, leverage=leverage)
                return None
            else:
                # 已经有一些利润，那么就等回落了一点就平仓了结
                trailing_stoploss = 0.02 * leverage
                if self.enable_logging:
                    logger.info(f'Holding over {long_time} minutes with some profit:{current_profit}, set a new trailing stoploss:{trailing_stoploss}')
                return trailing_stoploss
            
        long_time = 120
        if (current_time - timedelta(minutes=long_time)) > trade.open_date_utc:
            if current_profit < 0.02 * leverage:
                return 0.02 * leverage

        # # 3. 持仓时间不长，但是有小幅利润的情况下，设置一个保底出场价格，尽量能保证盈利出局
        # if current_profit > 0.04 * leverage:
        #     return stoploss_from_open(0.5 * 0.04 * leverage, current_profit, is_short=trade.is_short, leverage=leverage)
        
        # if current_profit > 0.02 * leverage:
        #     return stoploss_from_open(0.5 * 0.02 * leverage, current_profit, is_short=trade.is_short, leverage=leverage)
        
        # 4. 其它情况就使用初始设置的止损先抗一抗看是否能起来到目标价位，实在不行就触发止损出局
        return None
 
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
        dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_length)
        
        # strategy
        dataframe['ema_short'] = pta.ema(close=dataframe['ha_close'], length=self.ema_short_length, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['ha_close'], length=self.ema_mid_length, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ha_close'], length=self.ema_long_length, talib=False)
        
        dataframe['recent_high_mid'] = dataframe['ha_close'].rolling(window=self.mid_period).max()
        dataframe['recent_low_mid'] = dataframe['ha_close'].rolling(window=self.mid_period).min()
        
        dataframe['volume_ma'] = pta.sma(dataframe['volume'], length=5)

        adx = pta.adx(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.adx_length)
        dataframe['adx'] = adx[f'ADX_{self.adx_length}']
        
        dataframe['rsi'] = pta.rsi(dataframe['ha_close'], length=self.rsi_length, talib=False)
        
        dataframe['is_bullish'] = dataframe['ha_close'] > dataframe['ha_open']
        dataframe['bullish_count'] = dataframe['is_bullish'].rolling(window=self.short_period, min_periods=self.short_period).sum()

        dataframe['is_bearish'] = dataframe['ha_close'] < dataframe['ha_open']
        dataframe['bearish_count'] = dataframe['is_bearish'].rolling(window=self.short_period, min_periods=self.short_period).sum()
        
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
        long_entry_conditions.append(dataframe['bullish_count'] >= (int)(self.short_period * 0.7))
        
        dataframe['long_dir'] = 0
        dataframe.loc[
            (reduce(lambda x, y: x & y, long_entry_conditions))
        , 'long_dir'] = 1
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # manual entry
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
       
        return dataframe
  
    @property
    def protections(self): # type: ignore
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.mid_period,
                "trade_limit": 2,
                "stop_duration_candles": 2,
                "only_per_pair": True,
                "only_per_side": False
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage
 