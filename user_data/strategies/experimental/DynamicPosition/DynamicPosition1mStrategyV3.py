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


class DynamicPositionLong1mStrategyV3(IStrategy):
    
    # common
    minimal_roi = {"0": 100}
    
    trade_leverage = 2

    timeframe = '1m'
    
    stoploss = -0.03 * trade_leverage
    trailing_stop = False
    use_custom_stoploss = True

    can_short = True
 
    enable_logging = False
    
    position_adjustment_enable = True
    initial_position_ratio = 1/5
    position_adjustment_pct = 0.015
    position_adjustment_stake_ratio = 0.98
    
    period = 10
    ema_length = period
    adx_length = period
    adx_threshold = 25
    rsi_length = period
    rsi_long_threshold = 60
    startup_candle_count = period
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        stake_amount = min(max(min_stake, proposed_stake * self.initial_position_ratio), max_stake)
        logger.info(f'Stake amount for {pair}={stake_amount}')
        return stake_amount
    
    def get_leverage(self, trade: Trade):
        return trade.leverage if trade.leverage else 1
    
    def is_low_profit(self, current_profit: float, trade: Trade):
        return current_profit < self.position_adjustment_pct
    
    def is_long_time(self, current_time: datetime, trade: Trade):
        return (current_time - timedelta(minutes=32)) > trade.open_date_utc
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs,) -> Optional[Union[str, bool]]:
        # if self.is_long_time(current_time, trade) and self.is_low_profit(current_profit, trade) and (current_profit > 0.005 * trade.leverage):
        #     logger.info(f'Long time low profit so exit all {trade.pair} at {current_rate:.5f}, current profit:{current_profit:.2f}')
        #     return 'Long time low profit'
            
        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float, min_stake: float, max_stake: float, 
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        if not self.position_adjustment_enable:
            return None

        has_open_orders = any(order.status == "open" and not order.ft_is_open for order in trade.orders)
        if has_open_orders:
            logger.info(f'Skip {trade.pair} position adjustment when there are open orders')
            return None
        
        # 一、判断是否减仓
        leverage = self.get_leverage(trade)
        low_profit = self.is_low_profit(current_profit, trade)
        drawback = 0
        large_drawback = False
        if trade.max_rate is not None:
            drawback = (trade.max_rate - current_rate) / current_rate
            large_drawback = drawback > 0.1
            
        if low_profit or large_drawback:
            # logger.info(f'current profit:{current_profit:.2f}, low_profit={low_profit}. Drawback:{drawback:.2%}, large_drawback={large_drawback}')
            stake_to_decrease = 0
            
            for order in trade.select_filled_orders():
                price_percent = (current_rate - order.average) / order.average
                if trade.is_short:
                    price_percent *= -1
                if price_percent >= self.position_adjustment_pct * 2:
                    stake_to_decrease -= order.average * order.filled
            
            if stake_to_decrease < -0.001:
                if abs(stake_to_decrease) < abs(min_stake):
                    logger.info(f'Adjusting partial decrease stake {stake_to_decrease:.4f} to {-abs(min_stake):4f} according to min_stake:{min_stake:.4f} for {trade.pair}')
                    stake_to_decrease = -abs(min_stake)
                elif abs(stake_to_decrease) > abs(max_stake):
                    logger.info(f'Adjusting partial decrease stake {stake_to_decrease:.4f} to {-abs(max_stake):4f} according to max_stake:{max_stake:.4f} for {trade.pair}')
                    stake_to_decrease = -abs(max_stake)
                    
                logger.info(f'Position partial decrease for {trade.pair} with amount {stake_to_decrease:.4f} at {current_rate:.5f}, current profit:{current_profit:.2f}, low_profit={low_profit}. Drawback:{drawback:.2%}, large_drawback={large_drawback}')
                if low_profit:
                    return (stake_to_decrease, 'low profit decrease')
                elif large_drawback:
                    return (stake_to_decrease, 'large drawback decrease')
                
            # filled_count = len(trade.select_filled_orders())
            # if filled_count > 3:
            #     stake_to_decrease = -(trade.amount * current_rate) / 3
                
            #     if abs(stake_to_decrease) < abs(min_stake):
            #         logger.info(f'Adjusting partial exit stake {stake_to_decrease:.4f} to {-abs(min_stake):4f} according to min_stake for {trade.pair}')
            #         stake_to_decrease = -abs(min_stake)
                
            #     logger.info(f'Position partial exit for {trade.pair} with amount {stake_to_decrease:.4f} at {current_rate:.5f}, current profit:{current_profit:.2f}, low_profit={low_profit}. Drawback:{drawback:.2%}, large_drawback={large_drawback}')
            #     if low_profit:
            #         return (stake_to_decrease, 'low profit partial exit')
            #     elif large_drawback:
            #         return (stake_to_decrease, 'large drawback partial exit')

        # 二、判断是否加仓
        filled_entries = trade.select_filled_orders()
        count_of_entries = len(filled_entries)
        if count_of_entries == 0:
            return None
        
        latest_order = trade.select_order(order_side=trade.entry_side, is_open=False, only_filled=True)
        if latest_order is None:
            return None
      
        price_increase = (current_rate - latest_order.average) / latest_order.average
        if trade.is_short:
            price_increase *= -1
        
        if price_increase >= self.position_adjustment_pct:
            stake_ratio = pow(self.position_adjustment_stake_ratio, len(filled_entries))
            first_stake = filled_entries[0].stake_amount
            stake_to_increase = stake_ratio * first_stake
            
            logger.info(f'Set {trade.pair} stake to increase to {stake_to_increase:.4f} with stake_ratio:{stake_ratio:.2f} for #{len(filled_entries)} based on first stake:{first_stake:.4f}')
            
            if stake_to_increase < min_stake:
                if stake_to_increase < 0.5 * min_stake:
                    if self.wallets.get_available_stake_amount() < stake_to_increase:
                        return None
                    
                    logger.info(f'Try to increase {trade.pair} position while amount:{stake_to_increase:.4f} is smaller than min_stake:{min_stake:.4f}')
                    return None
                else:
                    if self.wallets.get_available_stake_amount() < min_stake:
                        return None
                    
                    logger.info(f'Adjusting {trade.pair} increase stake:{stake_to_increase:.4f} to min_stake:{min_stake:.4f}')
                    stake_to_increase = min_stake
                    
            logger.info(f'Position addition #{count_of_entries+1} for {trade.pair} with amount {stake_to_increase:.4f} triggered at price increase: {price_increase:.2%}({current_rate:.5f}/{latest_order.average:.5f}), profit:{current_profit:.2f}')
            return (stake_to_increase, 'price increase')
            
        return None
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        leverage = trade.leverage

        if after_fill:
            new_stoploss = 0.04 * trade.leverage
            logger.info(f'Reset {pair} stoploss {current_rate*(1-new_stoploss/trade.leverage):.4f} after fill {new_stoploss:.2f} relative to current_rate:{current_rate:.4f}')
            return new_stoploss
            
        if current_profit > 0.3 * leverage:
            return 0.15 * leverage

        # if current_profit > 0.16 * leverage:
        #     return 0.1 * leverage

        # if current_profit > 0.08 * leverage:
        #     return 0.5 * 0.08 * leverage

        # if current_profit > 0.04 * leverage:
        #     return 0.5 * 0.04 * leverage

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema'] = pta.ema(close=dataframe['close'], length=self.ema_length, talib=False)
        dataframe['ema_slope'] = dataframe['ema'] - dataframe['ema'].shift(1)
        dataframe['adx'] = pta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=self.adx_length)[f'ADX_{self.adx_length}']
        dataframe['rsi'] = pta.rsi(dataframe['close'], length=self.rsi_length, talib=False)
        dataframe['recent_high'] = dataframe['close'].rolling(window=self.period).max()
        # dataframe['recent_low'] = dataframe['close'].rolling(window=self.period).min()
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['recent_high'].shift(1)) &
                (dataframe['close'] > dataframe['ema']) &
                (dataframe['ema_slope'] > 0) &
                (dataframe['rsi'] > self.rsi_long_threshold) & 
                (dataframe['adx'] > self.adx_threshold)
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