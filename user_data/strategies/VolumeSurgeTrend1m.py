"""Pair-agnostic long/short strategy for an abnormal-volume market regime.

Long entries require persistent quote-volume expansion, a price impulse and a
breakout.  Short entries are deliberately asymmetric: they require an extreme
blow-off move followed by a confirmed rejection from a new high.  Signals are
evaluated on the completed candle; Freqtrade enters at the next candle's price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class VolumeSurgeTrend1m(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1m"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 1500

    # The exit signal and adaptive chandelier stop do the profit-taking.  The
    # static stop is a disaster cap and remains active if indicator data fails.
    minimal_roi = {"0": 10.0}
    stoploss = -0.18
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    trailing_stop = False

    order_types = {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Persisted on each Trade so the chandelier stop survives bot restarts.
    STOP_STATE_KEY = "atr_chandelier_v1"

    def bot_start(self, **kwargs) -> None:
        self._candles: dict[str, DataFrame] = {}

    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 240,
                "trade_limit": 2,
                "stop_duration_candles": 120,
                "only_per_pair": True,
                "only_per_side": True,
            },
        ]

    @staticmethod
    def _wilder_atr(dataframe: DataFrame, period: int = 14) -> pd.Series:
        previous_close = dataframe["close"].shift(1)
        true_range = pd.concat(
            [
                dataframe["high"] - dataframe["low"],
                (dataframe["high"] - previous_close).abs(),
                (dataframe["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.ewm(alpha=1.0 / period, adjust=False).mean()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        close = dataframe["close"]
        dataframe["ema20"] = close.ewm(span=20, adjust=False).mean()
        dataframe["ema60"] = close.ewm(span=60, adjust=False).mean()
        dataframe["ema120"] = close.ewm(span=120, adjust=False).mean()
        dataframe["atr"] = self._wilder_atr(dataframe)
        dataframe["atr_pct"] = dataframe["atr"] / close

        # Base-volume units change mechanically when price explodes.  The
        # close-times-base approximation keeps the volume regime in USDT.
        dataframe["quote_volume_approx"] = dataframe["volume"] * close
        dataframe["quote_volume_15m"] = dataframe["quote_volume_approx"].rolling(15).sum()
        dataframe["quote_volume_baseline"] = (
            dataframe["quote_volume_15m"].shift(1).rolling(1440, min_periods=720).median()
        )
        dataframe["volume_ratio"] = (
            dataframe["quote_volume_15m"] / dataframe["quote_volume_baseline"]
        )
        dataframe["return_30m"] = close.pct_change(30)
        dataframe["return_60m"] = close.pct_change(60)
        dataframe["prior_high_60m"] = dataframe["high"].shift(1).rolling(60).max()
        dataframe["prior_high_120m"] = dataframe["high"].shift(1).rolling(120).max()
        dataframe["prior_high_10m"] = dataframe["high"].shift(1).rolling(10).max()
        dataframe["prior_low_10m"] = dataframe["low"].shift(1).rolling(10).min()
        dataframe["close_from_high"] = close / dataframe["high"] - 1.0
        self._candles[metadata["pair"]] = dataframe.set_index("date")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        setup = (
            (dataframe["volume"] > 0)
            & (dataframe["volume_ratio"] >= 8.0)
            & (dataframe["return_30m"] >= 0.06)
            & (dataframe["close"] > dataframe["prior_high_60m"])
            & (dataframe["ema20"] > dataframe["ema120"])
            # Do not chase a candle whose volatility already makes the
            # disaster stop meaningless; wait for another breakout instead.
            & (dataframe["atr_pct"].between(0.001, 0.08))
        )
        fresh_setup = setup & ~setup.shift(1, fill_value=False)
        dataframe.loc[fresh_setup, "enter_long"] = 1
        dataframe.loc[fresh_setup, "enter_tag"] = "volume_breakout"

        # Short only after a genuine blow-off top.  Requiring a new 120-minute
        # high and a 12% rejection from that candle's high avoids repeatedly
        # shorting ordinary pullbacks while the squeeze is still accelerating.
        blowoff_reversal = (
            (dataframe["volume"] > 0)
            & (dataframe["volume_ratio"] >= 15.0)
            & (dataframe["return_60m"] >= 1.0)
            & (dataframe["high"] >= dataframe["prior_high_120m"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["close_from_high"] <= -0.12)
        )
        fresh_reversal = blowoff_reversal & ~blowoff_reversal.shift(1, fill_value=False)
        dataframe.loc[fresh_reversal, "enter_short"] = 1
        dataframe.loc[fresh_reversal, "enter_tag"] = "blowoff_reversal"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Two independent close-confirmed failures: a fast-trend breakdown or
        # a slower EMA reversal.  The hard/adaptive stop still handles a crash
        # inside the minute.
        fast_break = (dataframe["close"] < dataframe["ema20"]) & (
            dataframe["close"] < dataframe["prior_low_10m"]
        )
        slow_break = (dataframe["close"] < dataframe["ema60"]) & (
            dataframe["ema20"] < dataframe["ema60"]
        )
        dataframe.loc[fast_break | slow_break, "exit_long"] = 1
        dataframe.loc[fast_break, "exit_tag"] = "fast_trend_break"
        dataframe.loc[slow_break, "exit_tag"] = "ema_reversal"

        # Close-confirmed recovery against an open short.  The prior-high
        # requirement prevents an immediate exit while price is still above
        # EMA20 directly after the blow-off candle.
        short_squeeze = (dataframe["close"] > dataframe["ema20"]) & (
            dataframe["close"] > dataframe["prior_high_10m"]
        )
        dataframe.loc[short_squeeze, "exit_short"] = 1
        dataframe.loc[short_squeeze, "exit_tag"] = "short_squeeze"
        return dataframe

    def _entry_state(
        self, pair: str, current_time, entry_rate: float, is_short: bool
    ) -> dict[str, float | bool | str]:
        # Live callbacks include seconds and microseconds, while candle indexes
        # are aligned to the minute.  Normalize before selecting the signal bar.
        signal_time = pd.Timestamp(current_time).floor("min") - pd.Timedelta(minutes=1)
        history = self._candles.get(pair)
        if history is None or signal_time not in history.index:
            raise ValueError("missing completed entry-signal candle")
        atr = float(history.at[signal_time, "atr"])
        if not np.isfinite(atr) or atr <= 0 or entry_rate <= 0:
            raise ValueError("invalid ATR or entry rate")
        # Four ATRs avoids ordinary 1m noise.  The bounds prevent a quiet
        # regime from producing a hair-trigger stop and cap crash exposure.
        distance = min(0.18, max(0.03, 4.0 * atr / entry_rate))
        stop = entry_rate * (1.0 + distance if is_short else 1.0 - distance)
        return {
            "extreme": entry_rate,
            "stop": stop,
            "is_short": is_short,
            "last_candle": signal_time.isoformat(),
        }

    def _load_stop_state(self, trade: Trade) -> dict[str, float | bool | str] | None:
        raw = trade.get_custom_data(self.STOP_STATE_KEY)
        if not isinstance(raw, dict):
            return None
        try:
            extreme = float(raw["extreme"])
            stop = float(raw["stop"])
            is_short = bool(raw["is_short"])
            last_candle = pd.Timestamp(raw["last_candle"])
        except (KeyError, TypeError, ValueError):
            return None
        if (
            not np.isfinite(extreme)
            or extreme <= 0
            or not np.isfinite(stop)
            or stop <= 0
            or is_short != trade.is_short
        ):
            return None
        return {
            "extreme": extreme,
            "stop": stop,
            "is_short": is_short,
            "last_candle": last_candle.isoformat(),
        }

    def _ensure_stop_state(
        self, pair: str, trade: Trade, reference_time
    ) -> dict[str, float | bool | str] | None:
        state = self._load_stop_state(trade)
        if state is not None:
            return state
        try:
            # This also migrates an already-open trade after a strategy update.
            state = self._entry_state(pair, trade.open_date_utc, trade.open_rate, trade.is_short)
        except ValueError:
            # If the entry candle has fallen out of the live dataframe, retain
            # the persisted Freqtrade stop and price extreme as a safe fallback.
            extreme = trade.min_rate if trade.is_short else trade.max_rate
            extreme = float(extreme or trade.open_rate)
            stop = float(trade.stop_loss or 0.0)
            if not np.isfinite(extreme) or extreme <= 0 or stop <= 0:
                return None
            state = {
                "extreme": extreme,
                "stop": stop,
                "is_short": trade.is_short,
                "last_candle": (
                    pd.Timestamp(reference_time).floor("min") - pd.Timedelta(minutes=1)
                ).isoformat(),
            }
        trade.set_custom_data(self.STOP_STATE_KEY, state)
        return state

    def confirm_trade_entry(
        self,
        pair,
        order_type,
        amount,
        rate,
        time_in_force,
        current_time,
        entry_tag,
        side,
        **kwargs,
    ) -> bool:
        try:
            self._entry_state(pair, current_time, rate, side == "short")
            return side in ("long", "short")
        except ValueError:
            return False

    def order_filled(self, pair, trade, order, current_time, **kwargs) -> None:
        if order.ft_order_side == trade.entry_side:
            state = self._entry_state(pair, current_time, trade.open_rate, trade.is_short)
            trade.set_custom_data(self.STOP_STATE_KEY, state)

    def custom_stoploss(
        self,
        pair,
        trade,
        current_time,
        current_rate,
        current_profit,
        after_fill,
        **kwargs,
    ) -> float | None:
        # Set the initial stop at the fill.  Later stops are advanced in
        # bot_loop_start from the previous completed candle only.
        state = self._ensure_stop_state(pair, trade, current_time)
        if after_fill and state is not None:
            risk = abs(trade.open_rate - float(state["stop"]))
            return risk / current_rate * trade.leverage
        return None

    def bot_loop_start(self, current_time, **kwargs) -> None:
        completed_time = pd.Timestamp(current_time).floor("min") - pd.Timedelta(minutes=1)
        for trade in Trade.get_trades_proxy(is_open=True):
            history = self._candles.get(trade.pair)
            if history is None:
                continue
            state = self._ensure_stop_state(trade.pair, trade, current_time)
            if state is None:
                continue

            last_candle = pd.Timestamp(state["last_candle"])
            pending = history.loc[(history.index > last_candle) & (history.index <= completed_time)]
            changed = False
            for candle_time, candle in pending.iterrows():
                atr = float(candle["atr"])
                if np.isfinite(atr) and atr > 0:
                    if trade.is_short:
                        state["extreme"] = min(float(state["extreme"]), float(candle["low"]))
                        distance = min(0.18, max(0.03, 4.0 * atr / float(state["extreme"])))
                        candidate = float(state["extreme"]) * (1.0 + distance)
                        state["stop"] = min(float(state["stop"]), candidate)
                    else:
                        state["extreme"] = max(float(state["extreme"]), float(candle["high"]))
                        distance = min(0.18, max(0.03, 4.0 * atr / float(state["extreme"])))
                        candidate = float(state["extreme"]) * (1.0 - distance)
                        state["stop"] = max(float(state["stop"]), candidate)
                state["last_candle"] = pd.Timestamp(candle_time).isoformat()
                changed = True

            if changed:
                trade.set_custom_data(self.STOP_STATE_KEY, state)

            # Reapply the persisted absolute stop even when there is no new
            # candle, so a restarted bot immediately restores stop management.
            anchor = float(state["stop"]) / (1.001 if trade.is_short else 0.999)
            trade.adjust_stop_loss(anchor, 0.001 * trade.leverage)

    def leverage(
        self,
        pair,
        current_time,
        current_rate,
        proposed_leverage,
        max_leverage,
        entry_tag,
        side,
        **kwargs,
    ) -> float:
        # A contract moving 10-25% inside one minute does not need leverage.
        return 1.0
