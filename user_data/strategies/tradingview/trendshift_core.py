"""Shared indicator core for the TradingView script TbK1rB5B port.

Published source title: "Supertrend + Volatility Regime Switch" (Pine v6).
Both the faithful full-strategy port (``TrendShiftRegime``) and the
pure-signal matrix strategy (``TrendShiftSignal``) build their columns here so
the two can never drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib.abstract as ta


def supertrend_dynamic(
    dataframe: pd.DataFrame, period: int, multiplier: np.ndarray
) -> pd.DataFrame:
    """Pine ``ta.supertrend`` with a per-bar factor.

    The band ratchet always compares against the previous bar's bands, which
    were built with the previous bar's multiplier - that is what makes a
    regime switch shift the line rather than restart it.

    Returns ``st`` (the line) and ``dir`` with 1 = uptrend, -1 = downtrend.
    """
    high = dataframe["high"].to_numpy(dtype=float)
    low = dataframe["low"].to_numpy(dtype=float)
    close = dataframe["close"].to_numpy(dtype=float)

    atr = ta.ATR(dataframe, timeperiod=period).to_numpy(dtype=float)
    hl2 = (high + low) / 2.0
    upper_raw = hl2 - multiplier * atr
    lower_raw = hl2 + multiplier * atr

    n = len(close)
    direction = np.zeros(n, dtype=float)
    line = np.full(n, np.nan)

    prev_up = np.nan
    prev_dn = np.nan
    prev_dir = 1.0
    for i in range(n):
        u = upper_raw[i]
        d = lower_raw[i]
        if np.isnan(u):
            direction[i] = prev_dir
            continue
        prev_u = prev_up if not np.isnan(prev_up) else u
        prev_d = prev_dn if not np.isnan(prev_dn) else d
        if i > 0 and close[i - 1] > prev_u:
            u = max(u, prev_u)
        if i > 0 and close[i - 1] < prev_d:
            d = min(d, prev_d)

        if prev_dir == -1 and close[i] > prev_d:
            cur_dir = 1.0
        elif prev_dir == 1 and close[i] < prev_u:
            cur_dir = -1.0
        else:
            cur_dir = prev_dir

        direction[i] = cur_dir
        line[i] = u if cur_dir == 1 else d
        prev_up, prev_dn, prev_dir = u, d, cur_dir

    return pd.DataFrame({"st": line, "dir": direction}, index=dataframe.index)


def regime_multiplier(
    adx: np.ndarray, trend_th: float, chop_th: float, tight: float, wide: float
) -> tuple[np.ndarray, np.ndarray]:
    """Latched regime with a neutral hysteresis band; returns (regime, factor).

    regime: 1 = TRENDING, -1 = CHOPPY, 0 = NEUTRAL (only before the first
    threshold cross, matching the Pine ``var`` initialisation).
    """
    n = len(adx)
    regime = np.zeros(n)
    mult = np.full(n, tight)  # Pine seeds effectiveMult with tightMult.
    cur_regime = 0.0
    cur_mult = tight
    for i in range(n):
        a = adx[i]
        if not np.isnan(a):
            if a > trend_th:
                cur_regime, cur_mult = 1.0, tight
            elif a < chop_th:
                cur_regime, cur_mult = -1.0, wide
        regime[i] = cur_regime
        mult[i] = cur_mult
    return regime, mult


def add_signal_columns(
    dataframe: pd.DataFrame,
    adx_len: int,
    adx_trend_th: float,
    adx_chop_th: float,
    atr_len: int,
    tight_mult: float,
    wide_mult: float,
    disable_chop: bool,
) -> pd.DataFrame:
    """Attach every column the Pine entry/exit logic reads."""
    adx = ta.ADX(dataframe, timeperiod=adx_len).to_numpy(dtype=float)
    dataframe["adx"] = adx

    regime, mult = regime_multiplier(adx, adx_trend_th, adx_chop_th, tight_mult, wide_mult)
    dataframe["regime"] = regime
    dataframe["st_mult"] = mult

    st = supertrend_dynamic(dataframe, atr_len, mult)
    dataframe["st"] = st["st"]
    dataframe["st_dir"] = st["dir"]

    flip = dataframe["st_dir"] != dataframe["st_dir"].shift(1)
    dataframe["bull_flip"] = flip & (dataframe["st_dir"] == 1)
    dataframe["bear_flip"] = flip & (dataframe["st_dir"] == -1)

    # Pine gates on the raw isChoppy test, not on the latched regime.
    dataframe["is_choppy"] = dataframe["adx"] < adx_chop_th
    dataframe["entries_allowed"] = ~(dataframe["is_choppy"] & disable_chop)

    dataframe["stop_dist"] = (dataframe["close"] - dataframe["st"]).abs()
    dataframe["stop_ratio"] = dataframe["stop_dist"] / dataframe["close"]
    return dataframe
