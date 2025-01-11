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


class BreakoutScalping1mStrategyV3(IStrategy):
    
    # common
    minimal_roi = {"0": 100}
    
    trade_leverage = 2

    timeframe = '1m'
    
    stoploss = -0.07 * trade_leverage
    trailing_stop = False
    use_custom_stoploss = True

    can_short = True
    
    atr_length = 15
    atr_multiplier = 1.5
    
    # strategy
    short_period = 10
    mid_period = 20
    long_period = 30

    adx_length = short_period
    adx_threshold = 30
    
    rsi_length = short_period
    rsi_long_threshold = 65
    rsi_short_threshold = 35
    
    ema_short_length = short_period
    ema_mid_length = mid_period
    ema_long_length = long_period
    
    breakout_long_length = 120
    breakout_short_length = breakout_long_length
    
    startup_candle_count = max(breakout_long_length, breakout_short_length)
    
    enable_logging = False

    position_adjustment_enable = True
    max_additional_positions = 3
    first_profit_threshold = 0.2
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
            return 0.2 * leverage

        if current_profit > 0.16 * leverage:
            return stoploss_from_open(0.6 * 0.16 * leverage, current_profit, is_short=trade.is_short, leverage=leverage)

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
 
    def count_breakout(self, row_idx, prices, is_long: bool):
        if row_idx == 0:
            return 0
            
        current_price = prices.iloc[row_idx]
        count = 0
        
        for i in range(row_idx-1, -1, -1):
            if (is_long and current_price <= prices.iloc[i]) or (not is_long and current_price >= prices.iloc[i]):
                break
            count += 1
        
        return count
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['breakout_long'] = dataframe.index.map(
            lambda x: self.count_breakout(x, dataframe['close'], True)
        )
        
        dataframe['breakout_short'] = dataframe.index.map(
            lambda x: self.count_breakout(x, dataframe['close'], False)
        )
        
        dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_length)
        
        dataframe['ema_short'] = pta.ema(close=dataframe['close'], length=self.ema_short_length, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['close'], length=self.ema_mid_length, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['close'], length=self.ema_long_length, talib=False)

        dataframe['diff_ema_short'] = dataframe['close'] - dataframe['ema_short']
        dataframe['diff_ema_mid'] = dataframe['close'] - dataframe['ema_mid']
        dataframe['diff_ema_long'] = dataframe['close'] - dataframe['ema_long']
        
        dataframe['diff_ema_short_sum'] = dataframe['diff_ema_short'].rolling(self.short_period).sum()
        dataframe['diff_ema_mid_sum'] = dataframe['diff_ema_mid'].rolling(self.mid_period).sum()
        dataframe['diff_ema_long_sum'] = dataframe['diff_ema_long'].rolling(self.long_period).sum()
        
        dataframe['diff_short_mid'] = dataframe['ema_short'] - dataframe['ema_mid']
        dataframe['diff_mid_long'] = dataframe['ema_mid'] - dataframe['ema_long']
        
        dataframe['diff_short_mid_sum'] = dataframe['diff_short_mid'].rolling(self.short_period).sum()
        dataframe['diff_mid_long_sum'] = dataframe['diff_mid_long'].rolling(self.mid_period).sum()
 
        dataframe['diff_short_mid_0'] = dataframe['diff_short_mid'].apply(lambda x: 0 if x < 0 else x)
        dataframe['diff_mid_long_0'] = dataframe['diff_mid_long'].apply(lambda x: 0 if x < 0 else x)
                
        adx = pta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=self.adx_length)
        dataframe['adx'] = adx[f'ADX_{self.adx_length}']
        
        dataframe['rsi'] = pta.rsi(dataframe['close'], length=self.rsi_length, talib=False)
        
        return dataframe
        
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        long_entry_conditions = []
        long_entry_conditions.append(dataframe['breakout_long'] >= self.breakout_long_length)
        long_entry_conditions.append(dataframe['close'] >= dataframe['ema_short'])
        long_entry_conditions.append(dataframe['ema_short'] >= dataframe['ema_mid'])
        long_entry_conditions.append(dataframe['ema_mid'] >= dataframe['ema_long'])
        long_entry_conditions.append(dataframe['ema_short'] >= dataframe['ema_short'].shift(1))
        long_entry_conditions.append(dataframe['ema_mid'] >= dataframe['ema_mid'].shift(1))
        long_entry_conditions.append(dataframe['ema_long'] >= dataframe['ema_long'].shift(1))

        long_entry_conditions.append(dataframe['diff_short_mid'] >= dataframe['diff_short_mid'].shift(1))
        long_entry_conditions.append(dataframe['diff_short_mid_sum'] >= dataframe['diff_short_mid_sum'].shift(1))
        
        long_entry_conditions.append(dataframe['adx'] >= self.adx_threshold)
        long_entry_conditions.append(dataframe['rsi'] >= self.rsi_long_threshold)
        
        dataframe.loc[
        (
            (reduce(lambda x, y: x & y, long_entry_conditions))
        )
        , 'enter_long'] = 1
        
        short_entry_conditions = []
        short_entry_conditions.append(dataframe['breakout_short'] >= self.breakout_long_length)
        short_entry_conditions.append(dataframe['close'] <= dataframe['ema_short'])
        short_entry_conditions.append(dataframe['ema_short'] <= dataframe['ema_mid'])
        short_entry_conditions.append(dataframe['ema_mid'] <= dataframe['ema_long'])
        short_entry_conditions.append(dataframe['ema_short'] <= dataframe['ema_short'].shift(1))
        short_entry_conditions.append(dataframe['ema_mid'] <= dataframe['ema_mid'].shift(1))
        short_entry_conditions.append(dataframe['ema_long'] <= dataframe['ema_long'].shift(1))
        
        short_entry_conditions.append(dataframe['adx'] >= self.adx_threshold)
        short_entry_conditions.append(dataframe['rsi'] <= self.rsi_short_threshold)
        
        # dataframe.loc[
        # (
        #     (reduce(lambda x, y: x & y, short_entry_conditions))
        # )
        # , 'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # dataframe.loc[(
        #     (dataframe['close'] < dataframe['ema_long'])
        # ), ['exit_long', 'exit_tag']] = (1, 'ema_exit')
        
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
 