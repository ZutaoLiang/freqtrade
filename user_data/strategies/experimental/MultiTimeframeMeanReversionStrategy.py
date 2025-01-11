from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import pandas_ta as pta
import numpy as np
import pandas_ta as pta
from typing import Dict, List, Optional
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.persistence.trade_model import Order

import logging
logger = logging.getLogger(__name__)


class MultiTimeframeMeanReversionStrategy(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {"0": 100}
    timeframe = '3m'
    
    trade_leverage = 10
    
    trailing_stop = False
    use_custom_stoploss = False
    stoploss = -0.05 * trade_leverage
    
    use_custom_exit = True
    position_adjustment_enable = True
    
    ema_period = IntParameter(5, 50, default=20, space="buy")
    ema_long_period = IntParameter(20, 200, default=180, space="buy")
    ema_up_ratio = 1.01
    ema_trend_length = 7
    
    position_adjustment_threshold = DecimalParameter(0.005, 0.05, default=0.004 * trade_leverage, space="buy")
    stake_ratio = DecimalParameter(0.1, 1.0, default=0.75, space="buy")

    startup_candle_count = ema_long_period.value
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return self.trade_leverage
    
    def get_initial_market_value(self, trade: Trade) -> float:
        if not trade.orders:
            return 0.0
            
        initial_order = None
        for order in trade.orders:
            if order.ft_order_side == trade.entry_side and order.status == 'closed' and order.filled:
                initial_order = order
                break
        
        if initial_order:
            return initial_order.average * initial_order.filled
        
        return 0.0
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema'] = pta.ema(close=dataframe['close'], length=self.ema_period.value, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['close'], length=self.ema_long_period.value, talib=False)
        return dataframe
        
    def ema_up_n_days_mask(self, dataframe: DataFrame, ema: str, days: int):
        ema_up_mask = (dataframe[f'{ema}'] > dataframe[f'{ema}'].shift(1))
        for i in range(2, days):
            ema_up_mask = ema_up_mask & (dataframe[f'{ema}'].shift(i-1) > dataframe[f'{ema}'].shift(i))
        return ema_up_mask
    
    def ema_down_n_days_mask(self, dataframe: DataFrame, ema: str, days: int):
        ema_down_mask = (dataframe[f'{ema}'] < dataframe[f'{ema}'].shift(1))
        for i in range(2, days):
            ema_down_mask = ema_down_mask & (dataframe[f'{ema}'].shift(i-1) < dataframe[f'{ema}'].shift(i))
        return ema_down_mask
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_up_mask = self.ema_up_n_days_mask(dataframe, 'ema', self.ema_trend_length)
        ema_long_up_mask = self.ema_up_n_days_mask(dataframe, 'ema_long', self.ema_trend_length * 2)
        
        dataframe.loc[
                (
                    (dataframe['close'] > self.ema_up_ratio * dataframe['ema_long']) &
                    (ema_up_mask) &
                    (ema_long_up_mask)
                ),
                ['enter_long', 'enter_tag']] = (1, 'entry')
        
        return dataframe
        
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_long_down_mask = self.ema_down_n_days_mask(dataframe, 'ema_long', self.ema_trend_length * 5)
        
        dataframe.loc[
            (
                # (dataframe['close'] < dataframe['ema_long']) |
                (ema_long_down_mask)
            ), ['exit_long', 'exit_tag']] = (1, 'exit')
        
        return dataframe
        
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float,
                            min_stake: float, max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs) -> float:
        initial_market_value = self.get_initial_market_value(trade)
        if initial_market_value == 0.0:
            return None
        
        leverage = trade.leverage
        stake_amount = trade.stake_amount * leverage
        current_value = stake_amount * (1 + current_profit)
        value_change = (current_value / initial_market_value - 1)
        if abs(value_change) >= self.position_adjustment_threshold.value:
            diff_stake = -stake_amount * value_change / leverage
            abs_target = min(max(abs(diff_stake), min_stake / leverage), max_stake / leverage)
            adjusted_stake = abs_target if diff_stake > 0 else -abs_target
            logger.info(f'{trade.pair} adjust stake:{adjusted_stake:.4f}(after leverage:{trade.leverage*adjusted_stake:.4f}), initial market_value:{initial_market_value:.4f}, current_value:{current_value:.4f}, {value_change:.2%} at {current_time}')
            return adjusted_stake
            
        return None
        
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        leverage = trade.leverage
        profit_pct = current_profit / leverage

        if after_fill:
            return self.stoploss
        
        if profit_pct >= 0.25:
            return 0.1 * leverage
        elif profit_pct >= 0.15:
            return 0.07 * leverage
        elif profit_pct >= 0.08:
            return 0.05 * leverage
            
        return None

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        total_profit = trade.close_profit or 0
        if trade.is_open:
            total_profit += current_profit
            
        if total_profit < self.stoploss:
            return True
            
        return False

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: float, max_stake: float,
                          entry_tag: str, side: str, **kwargs) -> float:
        adjusted_stake = proposed_stake * self.stake_ratio.value
        return min(max(adjusted_stake, min_stake), max_stake)
