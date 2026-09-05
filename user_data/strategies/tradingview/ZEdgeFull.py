"""Faithful full-strategy port of "Z-Edge | Confluence Z-score Strategy".

Reference: TradingView script W67tJltC (Pine v6), published source title
"Multi-Factor Adaptive Z-Score Strategy" by blitz_locked (MPL-2.0).

Logic
-----
* Three factors are z-scored over the same 100-bar window and weighted:
  ROC(14) 0.4, RSI(14) 0.3, volume / SMA(volume, 20) 0.3.
* The composite is smoothed by an EMA whose length adapts to volatility: the
  ATR(14) percentile rank over 100 bars maps 15 bars (calm) down to 2 (wild).
* Default entries are zero-line crosses; ``Threshold Reversion`` mode instead
  buys the cross back up through -1.5 and sells the cross back down through
  +1.5. Exits cross the (default 0.0) exit levels.
* Sizing is risk based: 1% of a *static* equity input divided by ATR(14) x 2,
  capped so the position value stays under ``max_pct_equity`` of that equity.
* ``use_stop_loss`` is **on** by default: a fixed protective stop is placed at
  ``close -/+ ATR(14) * 2`` measured on the entry bar. It never moves.

Beware of the defaults
----------------------
* ``exit_level_long = exit_level_short = 0`` makes ``longExit`` the same
  condition as ``shortEntry`` (and vice versa), so the stock script is always
  in the market and flips on the zero line - the ATR stop is the only thing
  that can take it out early.
* The volume factor is direction-blind: ``volume / SMA(volume)`` is large on any
  high-volume bar, up or down, and it carries 30% of the weight, so volume
  spikes push the composite towards the long side regardless of price.
* Divergence detection is computed but never used for orders; it only drives
  plot shapes. It is therefore not implemented here.
* ``account_equity`` is a static input rather than ``strategy.equity``, so the
  "1% risk" never compounds. This port sizes off the real wallet by default
  (``use_static_equity = False``) because a fixed 10000 has no meaning against
  a different backtest wallet; set the flag to reproduce the Pine number.
* Pine runs with ``process_orders_on_close = true`` (fills at the signal bar's
  close). Freqtrade fills at the next bar's open. That is an engine difference,
  not something to tune away.

The pure-signal baselines used by the research matrix are ``ZEdgeSignal``
(zero cross) and ``ZEdgeReversion`` (threshold reversion) in this directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from zedge_core import add_order_columns, add_signal_columns

from freqtrade.strategy import IStrategy


class ZEdgeFull(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.99  # Replaced per trade by the ATR stop below.
    trailing_stop = False
    use_custom_stoploss = True
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

    startup_candle_count = 250

    # Pine inputs, defaults as published.
    entry_mode = "Zero Cross"  # or "Threshold Reversion"
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
    p_long_threshold = -1.5
    p_short_threshold = 1.5
    p_exit_level_long = 0.0
    p_exit_level_short = 0.0
    p_allow_longs = True
    p_allow_shorts = True

    use_stop_loss = True
    p_atr_mult_stop = 2.0
    p_risk_percent = 1.0
    p_max_pct_equity = 100.0
    use_static_equity = False
    p_account_equity = 10000.0

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
            allow_longs=self.p_allow_longs,
            allow_shorts=self.p_allow_shorts,
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
        dataframe.loc[dataframe["long_exit"], ["exit_long", "exit_tag"]] = (1, "exit_level")
        dataframe.loc[dataframe["short_exit"], ["exit_short", "exit_tag"]] = (1, "exit_level")
        # An opposite entry reverses the position in Pine.
        dataframe.loc[dataframe["short_entry"] & sizable, ["exit_long", "exit_tag"]] = (
            1,
            "reversal",
        )
        dataframe.loc[dataframe["long_entry"] & sizable, ["exit_short", "exit_tag"]] = (
            1,
            "reversal",
        )
        return dataframe

    def _signal_stop_ratio(self, pair: str, when=None) -> float | None:
        """``ATR(14) * 2 / close`` on the signal bar, as Pine measures it."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        row = dataframe.iloc[-1]
        if when is not None:
            earlier = dataframe.loc[dataframe["date"] < when]
            if not earlier.empty:
                row = earlier.iloc[-1]
        ratio = float(row["stop_ratio"])
        return ratio if np.isfinite(ratio) and ratio > 0 else None

    def custom_stake_amount(
        self, pair, current_time, current_rate, proposed_stake, min_stake, max_stake,
        leverage, entry_tag, side, **kwargs
    ):
        ratio = self._signal_stop_ratio(pair)
        if ratio is None:
            return proposed_stake
        equity = (
            self.p_account_equity
            if self.use_static_equity
            else self.wallets.get_total(self.config["stake_currency"])
        )
        risk_stake = equity * (self.p_risk_percent / 100.0) / ratio
        cap = equity * (self.p_max_pct_equity / 100.0)
        stake = min(risk_stake, cap)
        return max(min(stake, max_stake), min_stake or 0.0)

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit,
                        after_fill, **kwargs):
        """Fixed ATR stop placed on the entry bar; it does not move afterwards."""
        if not self.use_stop_loss:
            return None
        ratio = trade.get_custom_data("stop_ratio")
        if ratio is None:
            ratio = self._signal_stop_ratio(pair, when=trade.open_date_utc)
            if ratio is None:
                return None
            trade.set_custom_data("stop_ratio", ratio)
        # Pine measures the distance from the signal bar's close and leaves the
        # order there. freqtrade's custom_stoploss return value is relative to
        # the *current* rate, so a constant ratio would silently trail; the
        # level is anchored to the entry rate once and re-expressed each call.
        ratio = abs(float(ratio))
        stop_rate = trade.open_rate * (1 + ratio) if trade.is_short else trade.open_rate * (1 - ratio)
        if current_rate <= 0:
            return None
        # Not stoploss_from_absolute: it clamps the ratio at 1.0, which drags a
        # deep-in-profit short's stop down with the price (a trailing stop).
        rel = (
            (stop_rate / current_rate - 1.0) if trade.is_short else (1.0 - stop_rate / current_rate)
        )
        return rel if rel > 0 else -1e-6

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage,
                 entry_tag, side, **kwargs):
        return 1.0
