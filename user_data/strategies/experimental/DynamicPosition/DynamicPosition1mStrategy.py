from math import isnan
import numpy as np
import talib.abstract as ta
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


class DynamicPosition1mStrategy(IStrategy):
    
    # common
    minimal_roi = {"0": 100}
    
    trade_leverage = 2

    timeframe = '1m'
    
    stoploss = -0.04 * trade_leverage
    trailing_stop = False
    use_custom_stoploss = True

    can_short = True
 
    startup_candle_count = 60
    
    enable_logging = False
    
    position_adjustment_enable = True
    initial_position_ratio = 1/5
    position_adjustment_pct = 0.02
    position_adjustment_stake_ratio = 0.95
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        # 返回初始仓位比例的金额
        return (proposed_stake * self.initial_position_ratio)
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float, min_stake: float, max_stake: float, 
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        if not self.position_adjustment_enable:
            return None

        has_open_orders = any(order.status == "open" and not order.ft_is_open for order in trade.orders)
        if has_open_orders:
            # 还有之前加仓或者减仓的订单没有成交，则等待订单成交后再继续处理
            return None
        
        # 一、判断是否减仓
        leverage = trade.leverage if trade.leverage else 1
        low_profit = current_profit < 0.01 * leverage
        drawback = 0
        large_drawback = False
        if trade.max_rate is not None:
            drawback = (trade.max_rate - current_rate) / current_rate
            large_drawback = drawback > 0.05
        if low_profit or large_drawback:
            stake_to_decrease = 0
            for order in trade.select_filled_orders():
                price_percent = (current_rate - order.average) / order.average
                if trade.is_short:
                    price_percent *= -1
                if price_percent >= self.position_adjustment_pct * 2:
                    stake_to_decrease -= order.cost
            
            if stake_to_decrease < -1e-4:
                if abs(stake_to_decrease) < abs(min_stake):
                    logger.info(f'Adjusting decrease stake {stake_to_decrease:.4f} to {-abs(min_stake):4f} according to min_stake')
                    stake_to_decrease = -abs(min_stake)
                    
                logger.info(f'Position decrease for {trade.pair} with amount {stake_to_decrease:.4f} at {current_rate:.5f}, current profit:{current_profit:.2f}, low_profit={low_profit}. Drawback:{drawback:.2%}, large_drawback={large_drawback}')
                if low_profit:
                    return (stake_to_decrease, 'low profit decrease')
                elif large_drawback:
                    return (stake_to_decrease, 'large drawback decrease')

        # 二、判断是否加仓
        filled_entries = trade.select_filled_orders()
        count_of_entries = len(filled_entries)
        if count_of_entries == 0:
            # 没有任何成交，则不加仓
            return None
        
        latest_order = trade.select_order(is_open=False, only_filled=True)
        if latest_order is None:
            return None
      
        price_increase = (current_rate - latest_order.average) / latest_order.average
        if trade.is_short:
            price_increase *= -1
        
        if price_increase >= self.position_adjustment_pct:
            stake_to_increase = pow(self.position_adjustment_stake_ratio, len(filled_entries)) * filled_entries[0].cost / trade.leverage
            logger.info(f'Position addition #{count_of_entries+1} for {trade.pair} with amount {stake_to_increase:.4f} triggered at price increase: {price_increase:.2%}({current_rate:.5f}/{latest_order.average:.5f}), profit:{current_profit:.2f}')
            return (stake_to_increase, 'price increase')
            
        return None
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        leverage = trade.leverage

        # # 1. 检查是否超过4小时
        # if (current_time - timedelta(hours=2)) > trade.open_date_utc:
        #     if current_profit < 0:
        #         return self.position_adjustment_pct * 2 * leverage

        # 2. 当前处于高利润状态，持仓时间久一点，除非遇到较大的回撤
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

        # # 3. 处理长时间持仓没有达到目标涨幅的情况
        # long_time = 15
        # if (current_time - timedelta(minutes=long_time)) > trade.open_date_utc:
        #     if current_profit < 0.02 * leverage:
        #         return None
        #     else:
        #         # 已经有一些利润，那么就等回落了一点就平仓了结
        #         trailing_stoploss = 0.02 * leverage
        #         if self.enable_logging:
        #             logger.info(f'Holding over {long_time} minutes with some profit:{current_profit}, set a new trailing stoploss:{trailing_stoploss}')
        #         return trailing_stoploss
        
        # 4. 其它情况就使用初始设置的止损先抗一抗看是否能起来到目标价位，实在不行就触发止损出局
        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 计算EMA5
        dataframe['ema5'] = ta.EMA(dataframe['close'], timeperiod=5)
        # 计算EMA5斜率
        dataframe['ema5_slope'] = dataframe['ema5'] - dataframe['ema5'].shift(1)

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe.loc[
            (
                (dataframe['ema5_slope'] > 0) &  # EMA5向上
                (dataframe['close'] > dataframe['ema5']) &  # 收盘价在EMA5上方
                (dataframe['volume'] > 0)  # 确保有交易量
            ),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
 
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage