from pandas import DataFrame
import pandas_ta as pta
import numpy as np
from datetime import datetime
from typing import Optional

from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.constants import Config

import logging
logger = logging.getLogger(__name__)


class DirectionalManualV1(IStrategy):
    """
    Manual trading strategy with automatic stoploss management.
    - Provides EMA and SuperTrend indicators for manual entry decisions.
    - Implements dynamic tiered trailing stop loss.
    """

    timeframe = '1h'

    minimal_roi = {"0": 100}
    can_short = True
    process_only_new_candles = True
    position_adjustment_enable = True
    
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.trade_leverage = self.get_config("trade_leverage", 1)
        self.base_stop_loss = self.get_config("base_stop_loss", 0.15)
        self.stoploss = -float(self.base_stop_loss * self.trade_leverage)

        self.use_custom_stoploss = True

        # EMA parameters
        self.ema_short_length = self.get_config("ema_short_length", 20)
        self.ema_long_length = self.get_config("ema_long_length", 100)

        # SuperTrend parameters
        self.supertrend_length = self.get_config("supertrend_length", 10)
        self.supertrend_multiplier = self.get_config("supertrend_multiplier", 3.0)

        # Dynamic tiered trailing stop configuration
        # Each tier: (profit_threshold, trailing_stop_distance)
        # When _current_profit >= threshold, use the corresponding trailing distance
        # Tiers should be sorted ascending by threshold
        default_trailing_tiers = [
            [0.10, 0.05],
            [0.25, 0.10],
            [0.35, 0.15],
            [0.50, 0.20],
        ]
        self.trailing_tiers = self.get_config("trailing_tiers", default_trailing_tiers)
        # Sort tiers descending so we match the highest applicable tier first
        self.trailing_tiers = sorted(self.trailing_tiers, key=lambda x: x[0], reverse=True)

        # Custom exit parameters
        self.fee = self.get_config("fee", 0.0005)
        self.long_time_low_profit_hours = self.get_config("long_time_low_profit_hours", 12)
        self.long_time_low_profit_max = self.get_config("long_time_low_profit_max", 0.05)
        self.long_time_low_profit_lower_bound = self.get_config("long_time_low_profit_lower_bound", 0.003)
        self.long_time_low_profit_upper_bound = self.get_config("long_time_low_profit_upper_bound", 0.02)

        self.very_long_time_hours = self.get_config("very_long_time_hours", 24)
        self.very_long_time_profit_lower = self.get_config("very_long_time_profit_lower", 0.03)
        self.very_long_time_profit_upper = self.get_config("very_long_time_profit_upper", 0.06)

        self.startup_candle_count = max(self.ema_short_length, self.ema_long_length, self.supertrend_length) + 10

    def get_config(self, key: str, default):
        return self.config.get(key, default)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            # Short-term EMA
            dataframe['ema_short'] = pta.ema(close=dataframe['close'], length=self.ema_short_length, talib=False)

            # Long-term EMA
            dataframe['ema_long'] = pta.ema(close=dataframe['close'], length=self.ema_long_length, talib=False)

            # SuperTrend
            st = pta.supertrend(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                length=self.supertrend_length,
                multiplier=self.supertrend_multiplier,
            )
            if st is not None and not st.empty:
                # pandas_ta supertrend returns columns like:
                # SUPERT_{length}_{multiplier}, SUPERTd_{length}_{multiplier}, SUPERTl_{length}_{multiplier}, SUPERTs_{length}_{multiplier}
                st_col = f"SUPERT_{self.supertrend_length}_{self.supertrend_multiplier}"
                std_col = f"SUPERTd_{self.supertrend_length}_{self.supertrend_multiplier}"
                dataframe['supertrend'] = st[st_col]
                dataframe['supertrend_direction'] = st[std_col]  # 1 = bullish, -1 = bearish

            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_indicators: {e}")
            return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        No automatic entries - this is a manual trading strategy.
        Indicators (EMA short/long, SuperTrend) are calculated for visual reference only.
        """
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """
        Dynamic tiered trailing stop loss.
        Once profit (without leverage) reaches a tier threshold, a trailing stop is applied
        at the configured distance (trail_distance * leverage) from the current profit high.
        """
        leverage = trade.leverage
        _current_profit = current_profit / leverage

        # Check trailing tiers from highest to lowest threshold
        for threshold, trail_distance in self.trailing_tiers:
            if _current_profit >= threshold:
                # Return the trailing distance scaled by leverage
                # freqtrade will trail at this distance from the highest profit seen
                return trail_distance * leverage
        
        # No tier matched, use default stoploss
        return None

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        open_rate = trade.open_rate
        leverage = trade.leverage
        _current_profit = current_profit / leverage

        if self.long_time_low_profit_hours > 0:
            if trade.is_short:
                max_profit = (open_rate - trade.min_rate) / open_rate - 2 * self.fee
            else:
                max_profit = (trade.max_rate - open_rate) / open_rate - 2 * self.fee

            open_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
            if open_hours > self.long_time_low_profit_hours:
                if max_profit < self.long_time_low_profit_max and self.long_time_low_profit_lower_bound < _current_profit < self.long_time_low_profit_upper_bound:
                    return "longtime_low_profit"

        if self.very_long_time_hours > 0:
            open_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
            if open_hours > self.very_long_time_hours:
                if self.very_long_time_profit_lower <= _current_profit <= self.very_long_time_profit_upper:
                    return "very_long_time_low_profit"

        return None
