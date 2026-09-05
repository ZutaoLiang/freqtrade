"""Pure-signal baseline of TradingView script TbK1rB5B for the research matrix.

Published source: "Supertrend + Volatility Regime Switch" (Pine v6), page
title "TrendShift | Supertrend + ADX Regime-Adaptive Strategy".

This class exists to answer one question only, per skills/tradingview/SKILL.md
section 5: do the raw signals pay for their trading costs? Everything that is
not a signal is removed:

* no stop loss, no trailing stop, no break-even, no ATR backstop
* no take profit, no R-multiple target, no ROI exit (``minimal_roi = {}``)
* no timed exit, no cooldown, no protections, no position adjustment
* no risk-based position sizing - the config sets one fixed stake per pair

Kept from the original: the ADX regime latch, the regime-switched Supertrend
factor, the chop entry gate, and the reversal exit.

Exit definition (SKILL.md section 5, second case)
-------------------------------------------------
The Pine script has no independent exit signal. Its only exit is the opposite
``strategy.entry``, which flips the position. Two consequences are reproduced
here rather than smoothed over:

1. The exit fires on the opposite *entry condition*, not on the raw Supertrend
   flip. A flip while ADX < 20 is blocked by the chop gate, so the position is
   held through the chop instead of being closed.
2. Freqtrade cannot flip a position in one order. It closes on the exit signal
   and opens the opposite side from the entry signal; both land on the same
   candle's open, and both pay fees. Pine's single reversal order pays the same
   two commissions, but this is an engine difference, not an identity - the
   matrix reports how often a same-candle flip actually happened.

The residual freqtrade stop loss field cannot be removed. It is parked at -99%
and every trade that exits through it (or through liquidation) is counted and
reported separately; such a unit is not a clean pure-signal result.
"""

from __future__ import annotations

import pandas as pd
from trendshift_core import add_signal_columns

from freqtrade.strategy import BooleanParameter, DecimalParameter, IntParameter, IStrategy


class TrendShiftSignal(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"  # Overridden per matrix unit from the command line.
    can_short = True

    # --- Everything below is the "no intervention" baseline ---
    minimal_roi = {}  # No ROI exit at all, not a far-away number.
    stoploss = -0.99  # Engine-mandated field; triggers are counted and reported.
    trailing_stop = False
    use_custom_stoploss = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = False
    process_only_new_candles = True

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # ADX(14) + ATR(10) need ~50 bars; 200 leaves headroom on every timeframe.
    startup_candle_count = 200

    # Pine defaults, frozen for the matrix.
    adx_len = IntParameter(7, 28, default=14, space="buy", optimize=False)
    adx_trend_th = DecimalParameter(15.0, 40.0, default=25.0, decimals=1, space="buy", optimize=False)
    adx_chop_th = DecimalParameter(10.0, 30.0, default=20.0, decimals=1, space="buy", optimize=False)
    atr_len = IntParameter(5, 30, default=10, space="buy", optimize=False)
    tight_mult = DecimalParameter(0.5, 4.0, default=1.75, decimals=2, space="buy", optimize=False)
    wide_mult = DecimalParameter(2.0, 8.0, default=4.5, decimals=2, space="buy", optimize=False)
    disable_chop = BooleanParameter(default=True, space="buy", optimize=False)

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
        dataframe.loc[dataframe["bull_flip"] & allowed, ["enter_long", "enter_tag"]] = (
            1,
            "regime_long",
        )
        dataframe.loc[dataframe["bear_flip"] & allowed, ["enter_short", "enter_tag"]] = (
            1,
            "regime_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        allowed = dataframe["entries_allowed"] & (dataframe["stop_dist"] > 0)
        dataframe.loc[dataframe["bear_flip"] & allowed, ["exit_long", "exit_tag"]] = (
            1,
            "reversal",
        )
        dataframe.loc[dataframe["bull_flip"] & allowed, ["exit_short", "exit_tag"]] = (
            1,
            "reversal",
        )
        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return 1.0
