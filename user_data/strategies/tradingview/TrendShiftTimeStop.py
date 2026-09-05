"""TrendShift signals plus an explicit time stop - a labelled experiment.

Motivation: in the monthly matrix, 41-87% of 1d trades were closed by the
month boundary rather than by a signal, and that boundary handling was what
made the 1d row look profitable (see skills/tradingview/SKILL.md section 4.1).
The month end acts as a hidden time stop, so the obvious question is whether an
explicit time stop is itself the edge.

This is **not** the pure-signal baseline. It deliberately adds one exit that
SKILL.md section 5 forbids in the baseline, so it lives in its own class and its
results are never merged into the baseline matrix.

Note the time stop is not an invention: the Pine source exposes
``maxBarsInTrade`` (default 0 = off). Enabling it is a parameter choice the
author already provided.

Everything else - the ADX regime latch, the regime-switched Supertrend factor,
the chop entry gate, the reversal exit, fixed notional sizing, leverage 1 - is
identical to ``TrendShiftSignal``.
"""

from __future__ import annotations

import pandas as pd
from trendshift_core import add_signal_columns

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import BooleanParameter, DecimalParameter, IntParameter, IStrategy


class TrendShiftTimeStop(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1d"
    can_short = True

    minimal_roi = {}
    stoploss = -0.99  # Engine-mandated field; triggers counted and reported.
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

    startup_candle_count = 200

    adx_len = IntParameter(7, 28, default=14, space="buy", optimize=False)
    adx_trend_th = DecimalParameter(15.0, 40.0, default=25.0, decimals=1, space="buy", optimize=False)
    adx_chop_th = DecimalParameter(10.0, 30.0, default=20.0, decimals=1, space="buy", optimize=False)
    atr_len = IntParameter(5, 30, default=10, space="buy", optimize=False)
    tight_mult = DecimalParameter(0.5, 4.0, default=1.75, decimals=2, space="buy", optimize=False)
    wide_mult = DecimalParameter(2.0, 8.0, default=4.5, decimals=2, space="buy", optimize=False)
    disable_chop = BooleanParameter(default=True, space="buy", optimize=False)

    # The experiment variable. 0 disables it, reproducing TrendShiftSignal.
    max_bars_in_trade = IntParameter(0, 60, default=0, space="sell", optimize=True)

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
            1, "regime_long",
        )
        dataframe.loc[dataframe["bear_flip"] & allowed, ["enter_short", "enter_tag"]] = (
            1, "regime_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        allowed = dataframe["entries_allowed"] & (dataframe["stop_dist"] > 0)
        dataframe.loc[dataframe["bear_flip"] & allowed, ["exit_long", "exit_tag"]] = (1, "reversal")
        dataframe.loc[dataframe["bull_flip"] & allowed, ["exit_short", "exit_tag"]] = (1, "reversal")
        return dataframe

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        n = self.max_bars_in_trade.value
        if n <= 0:
            return None
        minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        if minutes // timeframe_to_minutes(self.timeframe) >= n:
            return "max_bars"
        return None

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return 1.0
