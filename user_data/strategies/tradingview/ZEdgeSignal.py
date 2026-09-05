"""Pure-signal baseline of TradingView script W67tJltC for the research matrix.

Published source: "Multi-Factor Adaptive Z-Score Strategy" (Pine v6), page
title "Z-Edge | Confluence Z-score Strategy" by blitz_locked (MPL-2.0).

Per skills/tradingview/SKILL.md section 5 this class answers one question only:
do the raw signals pay for their trading costs? Removed here:

* the author's ATR stop loss (``use_stop_loss = true`` in the Pine defaults) -
  unlike the same author's TrendShift script, this one *does* ship an active
  default risk module, so this baseline is NOT the author's default behaviour
* the risk-based position sizing (``risk 1% / (ATR * 2)``, capped at 100% of a
  static 10000 equity input) - the config sets one fixed stake per pair
* no ROI exit (``minimal_roi = {}``), no trailing stop, no timed exit, no
  cooldown, no protections, no position adjustment

Kept: the three z-scored factors, the ATR-percentile adaptive smoothing, the
zero-cross entries and the zero-cross exits.

Exit definition (SKILL.md section 5)
------------------------------------
With the Pine defaults ``exit_level_long = exit_level_short = 0``, so
``longExit`` is the same condition as ``shortEntry`` and ``shortExit`` the same
as ``longEntry``: the default script is always in the market and flips on the
zero line. Both are emitted here; the exit tag records which rule fired. As in
Pine, an opposite entry also closes the open position - freqtrade needs two
orders (close + open) where Pine reverses with one, and both pay fees, so the
matrix counts how often the flip actually completed inside one candle.

The residual freqtrade stoploss field cannot be removed. It is parked at -99%
and every ``stop_loss`` / ``liquidation`` exit is counted and reported; a unit
with a non-zero count is not a clean pure-signal result.
"""

from __future__ import annotations

import pandas as pd
from zedge_core import add_order_columns, add_signal_columns

from freqtrade.strategy import IStrategy


class ZEdgeSignal(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"  # Overridden per matrix unit from the command line.
    can_short = True

    entry_mode = "Zero Cross"

    # --- "no intervention" baseline ---
    minimal_roi = {}
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

    # z-score(100) over roc(14) needs 114 bars, atr(14) + percentrank(100)
    # needs 114 as well; 250 leaves headroom on every timeframe.
    startup_candle_count = 250

    # Pine defaults, frozen for the matrix.
    p_zscore_period = 100
    p_momentum_length = 14
    p_rsi_length = 14
    p_vol_length = 20
    p_w_price = 0.4
    p_w_rsi = 0.3
    p_w_vol = 0.3
    p_adaptive_on = True
    p_atr_length = 14
    p_atr_rank_length = 100
    p_min_smoothing = 2
    p_max_smoothing = 15
    p_smoothing_base = 5
    p_atr_mult_stop = 2.0
    p_long_threshold = -1.5
    p_short_threshold = 1.5
    p_exit_level_long = 0.0
    p_exit_level_short = 0.0

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = add_signal_columns(
            dataframe,
            zscore_period=self.p_zscore_period,
            momentum_length=self.p_momentum_length,
            rsi_length=self.p_rsi_length,
            vol_length=self.p_vol_length,
            w_price=self.p_w_price,
            w_rsi=self.p_w_rsi,
            w_vol=self.p_w_vol,
            adaptive_on=self.p_adaptive_on,
            atr_length=self.p_atr_length,
            atr_rank_length=self.p_atr_rank_length,
            min_smoothing=self.p_min_smoothing,
            max_smoothing=self.p_max_smoothing,
            smoothing_base=self.p_smoothing_base,
            atr_mult_stop=self.p_atr_mult_stop,
        )
        return add_order_columns(
            dataframe,
            entry_mode=self.entry_mode,
            long_threshold=self.p_long_threshold,
            short_threshold=self.p_short_threshold,
            exit_level_long=self.p_exit_level_long,
            exit_level_short=self.p_exit_level_short,
        )

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        sizable = dataframe["sizable"]
        dataframe.loc[dataframe["long_entry"] & sizable, ["enter_long", "enter_tag"]] = (
            1,
            "z_long",
        )
        dataframe.loc[dataframe["short_entry"] & sizable, ["enter_short", "enter_tag"]] = (
            1,
            "z_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        sizable = dataframe["sizable"]
        # Rule 1: the script's own exit level cross.
        dataframe.loc[dataframe["long_exit"], ["exit_long", "exit_tag"]] = (1, "exit_level")
        dataframe.loc[dataframe["short_exit"], ["exit_short", "exit_tag"]] = (1, "exit_level")
        # Rule 2: an opposite entry reverses the position in Pine, so it must
        # close the open trade here. With the defaults this coincides with
        # rule 1; the tag says which one is being reproduced.
        dataframe.loc[dataframe["short_entry"] & sizable, ["exit_long", "exit_tag"]] = (
            1,
            "reversal",
        )
        dataframe.loc[dataframe["long_entry"] & sizable, ["exit_short", "exit_tag"]] = (
            1,
            "reversal",
        )
        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage,
                 entry_tag, side, **kwargs):
        return 1.0
