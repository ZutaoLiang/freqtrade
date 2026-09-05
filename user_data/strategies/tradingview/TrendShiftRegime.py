"""Faithful full-strategy port of "TrendShift | Supertrend + ADX Regime-Adaptive".

Reference: TradingView script TbK1rB5B (Pine v6), published source title
"Supertrend + Volatility Regime Switch".

Logic
-----
* ADX(14, smoothing 14) picks a regime with hysteresis: TRENDING above 25,
  CHOPPY below 20, and the band in between holds whatever regime was last set.
* The Supertrend factor switches with the regime - 1.75 while TRENDING, 4.5
  while CHOPPY - so the band width adapts. ATR length is 10 throughout.
* Every Supertrend flip is a reversal signal, long and short, gated by the chop
  filter (``disableChop``, on by default, blocks entries while ADX < 20).
* Position size is risk based: 1% of equity divided by the distance between
  close and the Supertrend line.

Beware of the defaults
----------------------
``useTrailing`` is false by default, and the Pine ``strategy.exit`` calls that
carry the Supertrend stop and the 2R take profit sit *inside* that branch.
``flattenOnChop`` is false and ``maxBarsInTrade`` is 0. With stock settings the
strategy therefore has **no exit orders at all**: it is always in the market and
only leaves a position when the opposite entry fires. The advertised risk
management is off unless the user turns it on.

A second consequence of that structure: the reversal is the exit. When a flip
happens while ADX < 20 the opposite entry is blocked, so the existing position
is simply held through the chop. This port reproduces that, exits are tied to
the opposite signal rather than to the raw flip.

This class keeps the original risk modules so the published strategy can be
reproduced as written. The pure-signal baseline used by the research matrix is
``TrendShiftSignal`` in this same directory.

Pine defaults: ADX 14/14 with 25/20 thresholds, ATR 10, multipliers 1.75
(trending) / 4.5 (choppy), chop entries disabled, trailing off, TP 2R (inert
while trailing is off), 1% equity risk, 0.05% commission.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib.abstract as ta

from trendshift_core import add_signal_columns

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import (
    BooleanParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
)


class TrendShiftRegime(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = True  # Only bites when use_trailing is enabled.
    use_custom_roi = True  # Only bites when use_trailing and use_tp are enabled.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    process_only_new_candles = True

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    startup_candle_count = 200

    # --- Regime detection ---
    adx_len = IntParameter(7, 28, default=14, space="buy", optimize=True)
    adx_trend_th = DecimalParameter(15.0, 40.0, default=25.0, decimals=1, space="buy", optimize=True)
    adx_chop_th = DecimalParameter(10.0, 30.0, default=20.0, decimals=1, space="buy", optimize=True)

    # --- Supertrend ---
    atr_len = IntParameter(5, 30, default=10, space="buy", optimize=True)
    tight_mult = DecimalParameter(0.5, 4.0, default=1.75, decimals=2, space="buy", optimize=True)
    wide_mult = DecimalParameter(2.0, 8.0, default=4.5, decimals=2, space="buy", optimize=True)
    disable_chop = BooleanParameter(default=True, space="buy", optimize=False)
    flatten_on_chop = BooleanParameter(default=False, space="sell", optimize=False)

    # --- Risk management (all off by default, as in Pine) ---
    use_trailing = BooleanParameter(default=False, space="sell", optimize=False)
    use_tp = BooleanParameter(default=True, space="sell", optimize=False)
    r_multiple = DecimalParameter(0.5, 5.0, default=2.0, decimals=1, space="sell", optimize=False)
    max_bars_in_trade = IntParameter(0, 200, default=0, space="sell", optimize=False)
    risk_pct = DecimalParameter(0.1, 5.0, default=1.0, decimals=1, space="buy", optimize=False)

    leverage_value = 1.0

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return add_signal_columns(
            dataframe,
            adx_len=self.adx_len.value,
            adx_trend_th=self.adx_trend_th.value,
            adx_chop_th=self.adx_chop_th.value,
            atr_len=self.atr_len.value,
            tight_mult=self.tight_mult.value,
            wide_mult=self.wide_mult.value,
            disable_chop=self.disable_chop.value,
        )

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        allowed = dataframe["entries_allowed"] & (dataframe["stop_dist"] > 0)
        dataframe.loc[
            dataframe["bull_flip"] & allowed, ["enter_long", "enter_tag"]
        ] = (1, "regime_long")
        dataframe.loc[
            dataframe["bear_flip"] & allowed, ["enter_short", "enter_tag"]
        ] = (1, "regime_short")
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # The reversal entry is the only exit in stock Pine: a flip that cannot
        # open the opposite side leaves the current position untouched.
        allowed = dataframe["entries_allowed"] & (dataframe["stop_dist"] > 0)
        dataframe.loc[
            dataframe["bear_flip"] & allowed, ["exit_long", "exit_tag"]
        ] = (1, "reversal")
        dataframe.loc[
            dataframe["bull_flip"] & allowed, ["exit_short", "exit_tag"]
        ] = (1, "reversal")

        if self.flatten_on_chop.value:
            dataframe.loc[dataframe["is_choppy"], ["exit_long", "exit_tag"]] = (1, "choppy")
            dataframe.loc[dataframe["is_choppy"], ["exit_short", "exit_tag"]] = (1, "choppy")
        return dataframe

    def custom_stake_amount(
        self, pair, current_time, current_rate, proposed_stake, min_stake, max_stake,
        leverage, entry_tag, side, **kwargs
    ):
        """qty = (equity * risk%) / |close - supertrend|, expressed as stake."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return proposed_stake
        stop_ratio = float(dataframe["stop_ratio"].iloc[-1])
        if not np.isfinite(stop_ratio) or stop_ratio <= 0:
            return proposed_stake
        equity = self.wallets.get_total(self.config["stake_currency"])
        stake = equity * (self.risk_pct.value / 100.0) / stop_ratio
        # Pine's default margin settings do not allow borrowing, so the order
        # is capped by available funds rather than levered up.
        return max(min(stake, max_stake), min_stake or 0.0)

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill, **kwargs):
        """Supertrend line as a trailing stop - only when use_trailing is set."""
        if not self.use_trailing.value:
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        st = float(dataframe["st"].iloc[-1])
        if not np.isfinite(st) or current_rate <= 0:
            return None
        # Ratio relative to the current rate, as freqtrade expects.
        ratio = (st / current_rate - 1.0) if not trade.is_short else (1.0 - st / current_rate)
        return min(ratio, -1e-4)

    def custom_roi(self, pair, trade, current_time, trade_duration, entry_tag, side, **kwargs):
        """R-multiple take profit; inert unless the trailing exit is enabled."""
        if not (self.use_trailing.value and self.use_tp.value):
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        stop_dist = float(dataframe["stop_dist"].iloc[-1])
        if not np.isfinite(stop_dist) or trade.open_rate <= 0:
            return None
        # Pine recomputes the target from the *current* bar's stop distance.
        return stop_dist * self.r_multiple.value / trade.open_rate

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        max_bars = self.max_bars_in_trade.value
        if max_bars > 0:
            minutes = (current_time - trade.open_date_utc).total_seconds() / 60
            if minutes // timeframe_to_minutes(self.timeframe) >= max_bars:
                return "max_bars"
        return None

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return self.leverage_value
