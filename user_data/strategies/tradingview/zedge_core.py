"""Shared indicator core for the TradingView script W67tJltC port.

Published source title: "Multi-Factor Adaptive Z-Score Strategy" (Pine v6),
page title "Z-Edge | Confluence Z-score Strategy" by blitz_locked, MPL-2.0.

Every strategy class built on this file - the pure-signal matrix variants
(``ZEdgeSignal``, ``ZEdgeReversion``) and the faithful full port
(``ZEdgeFull``) - reads its columns from here so they cannot drift apart.

Pine semantics that are easy to get wrong, and how they are handled:

* ``ta.stdev`` is the *population* standard deviation (divides by n), while
  ``pandas.rolling.std()`` defaults to the sample one - ``ddof=0`` everywhere.
* ``ta.percentrank(x, len)`` counts how many of the ``len`` *previous* bars are
  <= the current value; the current bar is not part of the window.
* The smoothing EMA has a per-bar alpha, so it must be iterated bar by bar.
  ``ewm(span=...)`` cannot express it.
* ``na`` propagation is reproduced rather than patched: while the composite or
  alpha is undefined the smoothed line is undefined too, and the recursion
  re-seeds from the composite on the first bar after a gap - exactly what the
  Pine ``na(smoothComposite[1]) ? composite : ...`` line does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib.abstract as ta
from numpy.lib.stride_tricks import sliding_window_view


def zscore(series: pd.Series, length: int) -> pd.Series:
    """Pine ``(s - ta.sma(s, len)) / ta.stdev(s, len)``, 0.0 when stdev == 0."""
    roll = series.rolling(length, min_periods=length)
    mean = roll.mean()
    sd = roll.std(ddof=0)  # population stdev, like ta.stdev
    out = (series - mean) / sd
    out[sd == 0] = 0.0
    return out.where(~mean.isna())


def percentrank(series: pd.Series, length: int) -> pd.Series:
    """Pine ``ta.percentrank``: % of the previous ``length`` bars <= current."""
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    if n <= length:
        return pd.Series(out, index=series.index)
    # Window i covers values[i-length : i]; compared against values[i].
    windows = sliding_window_view(values[:-1], length)  # rows start at 0..n-length-1
    current = values[length:]
    counts = (windows <= current[:, None]).sum(axis=1)
    valid = ~np.isnan(windows).any(axis=1) & ~np.isnan(current)
    ranks = np.where(valid, counts * 100.0 / length, np.nan)
    out[length:] = ranks
    return pd.Series(out, index=series.index)


def adaptive_ema(composite: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """``prev + alpha * (composite - prev)``, re-seeded whenever prev is na."""
    n = len(composite)
    out = np.full(n, np.nan)
    prev = np.nan
    for i in range(n):
        cur = composite[i] if np.isnan(prev) else prev + alpha[i] * (composite[i] - prev)
        out[i] = cur
        prev = cur
    return out


def add_signal_columns(
    dataframe: pd.DataFrame,
    zscore_period: int = 100,
    momentum_length: int = 14,
    rsi_length: int = 14,
    vol_length: int = 20,
    w_price: float = 0.4,
    w_rsi: float = 0.3,
    w_vol: float = 0.3,
    adaptive_on: bool = True,
    atr_length: int = 14,
    atr_rank_length: int = 100,
    min_smoothing: int = 2,
    max_smoothing: int = 15,
    smoothing_base: int = 5,
    atr_mult_stop: float = 2.0,
) -> pd.DataFrame:
    """Attach every column the Pine order logic reads."""
    src = dataframe["close"]

    # roc: guard the Pine division by zero instead of letting inf poison the
    # z-score window. A non-positive reference price means "no signal here".
    ref = src.shift(momentum_length)
    roc = (src - ref) / ref * 100.0
    roc = roc.where(ref > 0)

    rsi_val = ta.RSI(dataframe, timeperiod=rsi_length)

    vol_sma = dataframe["volume"].rolling(vol_length, min_periods=vol_length).mean()
    vol_ratio = (dataframe["volume"] / vol_sma).where(vol_sma > 0)

    dataframe["z_price"] = zscore(roc, zscore_period)
    dataframe["z_rsi"] = zscore(rsi_val, zscore_period)
    dataframe["z_vol"] = zscore(vol_ratio, zscore_period)

    w_sum = w_price + w_rsi + w_vol
    w_sum_safe = 1.0 if w_sum == 0 else w_sum
    dataframe["composite"] = (
        dataframe["z_price"] * w_price
        + dataframe["z_rsi"] * w_rsi
        + dataframe["z_vol"] * w_vol
    ) / w_sum_safe

    atr_val = ta.ATR(dataframe, timeperiod=atr_length)
    dataframe["atr"] = atr_val
    atr_rank = percentrank(atr_val, atr_rank_length)
    dataframe["atr_pct_rank"] = atr_rank

    if adaptive_on:
        dyn_len_float = max_smoothing - (max_smoothing - min_smoothing) * (atr_rank / 100.0)
    else:
        dyn_len_float = pd.Series(float(smoothing_base), index=dataframe.index)
    # Pine math.round is half away from zero; these values are always positive.
    dyn_len = np.floor(dyn_len_float.to_numpy(dtype=float) + 0.5)
    dyn_len = np.maximum(dyn_len, 1.0)
    dataframe["dyn_len"] = dyn_len
    alpha = 2.0 / (dyn_len + 1.0)
    dataframe["alpha"] = alpha

    smooth = adaptive_ema(dataframe["composite"].to_numpy(dtype=float), alpha)
    dataframe["smooth"] = smooth

    # Sizing / stop inputs (used by the faithful port, reported by the others).
    dataframe["stop_dist"] = atr_val * atr_mult_stop
    dataframe["stop_ratio"] = dataframe["stop_dist"] / dataframe["close"]
    return dataframe


def crossover(series: pd.Series, level: float) -> pd.Series:
    """Pine ``ta.crossover(series, level)``."""
    prev = series.shift(1)
    return (prev <= level) & (series > level)


def crossunder(series: pd.Series, level: float) -> pd.Series:
    """Pine ``ta.crossunder(series, level)``."""
    prev = series.shift(1)
    return (prev >= level) & (series < level)


def add_order_columns(
    dataframe: pd.DataFrame,
    entry_mode: str = "Zero Cross",
    long_threshold: float = -1.5,
    short_threshold: float = 1.5,
    exit_level_long: float = 0.0,
    exit_level_short: float = 0.0,
    allow_longs: bool = True,
    allow_shorts: bool = True,
) -> pd.DataFrame:
    """Pine entry/exit conditions, before any freqtrade-specific handling."""
    smooth = dataframe["smooth"]
    if entry_mode == "Zero Cross":
        long_entry = crossover(smooth, 0.0)
        short_entry = crossunder(smooth, 0.0)
    elif entry_mode == "Threshold Reversion":
        long_entry = crossover(smooth, long_threshold)
        short_entry = crossunder(smooth, short_threshold)
    else:
        raise ValueError(f"unknown entry_mode {entry_mode!r}")

    dataframe["long_entry"] = long_entry & allow_longs
    dataframe["short_entry"] = short_entry & allow_shorts
    dataframe["long_exit"] = crossunder(smooth, exit_level_long)
    dataframe["short_exit"] = crossover(smooth, exit_level_short)
    # Pine gates both entries on positionSizeUnits > 0, i.e. on a usable stop
    # distance; ATR is 0 only on degenerate flat data.
    dataframe["sizable"] = dataframe["stop_dist"] > 0
    return dataframe


class AtrStopMixin:
    """The Pine ``use_stop_loss`` module: a fixed ATR stop placed on entry.

    Pine measures ``stopDistance = ATR(atr_length) * atr_mult_stop`` on the
    signal bar and leaves the order at ``close -/+ stopDistance``; it never
    moves afterwards. freqtrade's ``custom_stoploss`` return value is relative
    to the *current* rate, so returning a constant ratio would silently turn
    the fixed stop into a trailing one. The ratio is therefore captured once
    from the signal bar, anchored to the entry rate, and re-expressed on every
    call through ``stoploss_from_absolute``.
    """

    use_custom_stoploss = True
    use_stop_loss = True

    def _signal_stop_ratio(self, pair: str, when=None) -> float | None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        row = dataframe.iloc[-1]
        if when is not None:
            earlier = dataframe.loc[dataframe["date"] < when]
            if earlier.empty:
                return None
            row = earlier.iloc[-1]
        ratio = float(row["stop_ratio"])
        return ratio if np.isfinite(ratio) and ratio > 0 else None

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit,
                        after_fill, **kwargs):
        if not self.use_stop_loss:
            return None
        ratio = trade.get_custom_data("stop_ratio")
        if ratio is None:
            ratio = self._signal_stop_ratio(pair, when=trade.open_date_utc)
            if ratio is None:
                return None
            trade.set_custom_data("stop_ratio", ratio)
        ratio = abs(float(ratio))
        stop_rate = (
            trade.open_rate * (1 + ratio) if trade.is_short else trade.open_rate * (1 - ratio)
        )
        if current_rate <= 0:
            return None
        # freqtrade turns the return value into an absolute level via
        # ``current_rate * (1 -/+ abs(value))``, so the ratio has to be
        # re-expressed against the current rate on every call to keep the
        # level fixed. ``stoploss_from_absolute`` cannot be used here: it
        # clamps the ratio at 1.0, which for a short more than 50% in profit
        # silently drags the stop down with the price and turns the fixed stop
        # into a trailing one.
        rel = (stop_rate / current_rate - 1.0) if trade.is_short else (1.0 - stop_rate / current_rate)
        return rel if rel > 0 else -1e-6
