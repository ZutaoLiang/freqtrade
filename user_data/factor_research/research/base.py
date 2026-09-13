"""Base series -- the inputs the operator library is swept over.

Two rules shape this list. First, prefer stationary quantities: a rolling
z-score of a price level is dominated by the trend, so ratios and differences
go in and raw levels stay out except where an operator will normalise them
anyway. Second, include the series a generic factor library cannot produce --
funding rate, basis against mark price, and taker aggressor imbalance are
perpetual-futures specific and are where an edge is most likely to survive.

Each series is a (time, symbol) float32 array on the panel's own grid, so the
sweep is base series x operator x window with no per-symbol loop anywhere.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import operators as ops  # noqa: E402
from config import atomic_save  # noqa: E402
from panel import CACHE  # noqa: E402

BARS_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48,
                "1h": 24, "4h": 6, "1d": 1,
                "1h_hist": 24, "1d_long": 1, "4h_long": 6}   # panel variants share their base grid

# Series that need funding or mark price; absent until merge_aux has run.
AUX_SERIES = ("funding_rate", "funding_annual", "basis", "funding_z")
# Series that need the metrics dataset; absent until merge_metrics has run.
# Defined in reports/PREREGISTRATION_round4.md before any of them was measured.
OI_SERIES = ("log_oi", "oi_change", "oi_turnover", "oi_price_agree",
             "ls_top_position", "ls_top_account", "ls_retail", "ls_divergence",
             "taker_ls", "oi_per_trade")


def _div(a, b):
    """Ratio that yields NaN rather than inf where the denominator vanishes."""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return np.where(np.isfinite(out), out, np.nan)


def build(panel, include_aux=True):
    """All base series for one panel, as a dict of (T, N) float32 arrays."""
    bpd = BARS_PER_DAY[panel.tf]
    day = max(2, bpd)
    close = np.asarray(panel["close"], dtype="float64")
    open_ = np.asarray(panel["open"], dtype="float64")
    high = np.asarray(panel["high"], dtype="float64")
    low = np.asarray(panel["low"], dtype="float64")
    volume = np.asarray(panel["volume"], dtype="float64")
    qv = np.asarray(panel["quote_volume"], dtype="float64")
    count = np.asarray(panel["count"], dtype="float64")
    taker = np.asarray(panel["taker_buy_volume"], dtype="float64")

    prev_close = ops.ts_delay(close, 1)
    ret1 = _div(close, prev_close) - 1.0
    vwap = _div(qv, volume)

    s = {
        # Price shape within and between bars
        "ret1": ret1,
        "gap": _div(open_, prev_close) - 1.0,
        "body": _div(close - open_, open_),
        "hl_range": _div(high - low, close),
        "upper_wick": _div(high - np.maximum(open_, close), close),
        "lower_wick": _div(np.minimum(open_, close) - low, close),
        "close_loc": _div(close - low, high - low),      # where in the bar it closed
        "vwap_dev": _div(close, vwap) - 1.0,
        # Activity, all scaled so they are comparable across symbols
        "log_qv": ops.sign_log(qv),
        "vol_ratio": _div(volume, ops.ts_mean(volume, day)),
        "qv_ratio": _div(qv, ops.ts_mean(qv, day)),
        "count_ratio": _div(count, ops.ts_mean(count, day)),
        "avg_trade_qv": _div(qv, count),
        # Order-flow aggression: the share of volume that lifted the offer
        "taker_imbalance": 2.0 * _div(taker, volume) - 1.0,
        # Illiquidity: price move per unit of turnover
        "amihud": _div(np.abs(ret1), qv) * 1e9,
        # Level, kept only because several operators normalise it themselves
        "close": close,
    }

    if include_aux:
        fields = set(panel.meta.get("fields", []))
        if "funding_rate" in fields:
            fr = np.asarray(panel["funding_rate"], dtype="float64")
            s["funding_rate"] = fr
            if "funding_interval_hours" in fields:
                iv = np.asarray(panel["funding_interval_hours"], dtype="float64")
                # Annualised so 4h and 8h symbols are on one scale.
                s["funding_annual"] = fr * _div(24.0, iv) * 365.0
            # Crowding: how extreme funding is against its own recent history.
            s["funding_z"] = ops.ts_zscore(fr, 7 * day)
        if "mark_close" in fields:
            mark = np.asarray(panel["mark_close"], dtype="float64")
            s["basis"] = _div(mark - close, close)

        # Open interest and positioning. Open interest is the one quantity here
        # that OHLCV cannot approximate at all: price rising on growing open
        # interest is new money taking the other side, price rising on
        # shrinking open interest is shorts being closed out, and the two say
        # opposite things about what comes next.
        if "sum_open_interest_value" in fields:
            oiv = np.asarray(panel["sum_open_interest_value"], dtype="float64")
            s["log_oi"] = ops.sign_log(oiv)
            s["oi_change"] = _div(oiv, ops.ts_delay(oiv, 1)) - 1.0
            s["oi_turnover"] = _div(qv, oiv)
            s["oi_per_trade"] = _div(oiv, count)
            # Four regimes in one series: +1 when price and open interest move
            # together (positions opening into the move), -1 when they diverge
            # (positions closing).
            s["oi_price_agree"] = np.sign(ret1) * np.sign(s["oi_change"])
        if "sum_toptrader_long_short_ratio" in fields:
            s["ls_top_position"] = np.asarray(
                panel["sum_toptrader_long_short_ratio"], dtype="float64")
        if "count_toptrader_long_short_ratio" in fields:
            s["ls_top_account"] = np.asarray(
                panel["count_toptrader_long_short_ratio"], dtype="float64")
        if "count_long_short_ratio" in fields:
            s["ls_retail"] = np.asarray(
                panel["count_long_short_ratio"], dtype="float64")
        if "ls_top_position" in s and "ls_retail" in s:
            # Log difference, so "large accounts twice as long as the crowd" is
            # the same distance as "half as long".
            with np.errstate(invalid="ignore", divide="ignore"):
                d = np.log(s["ls_top_position"]) - np.log(s["ls_retail"])
            s["ls_divergence"] = np.where(np.isfinite(d), d, np.nan)
        if "sum_taker_long_short_vol_ratio" in fields:
            s["taker_ls"] = np.asarray(
                panel["sum_taker_long_short_vol_ratio"], dtype="float64")

    return {k: v.astype("float32") for k, v in s.items()}


def cache_dir(tf):
    return os.path.join(CACHE, tf, "base")


def cache(panel, include_aux=True):
    """Materialise the base series once so sweep workers can mmap them.

    Recomputing 16 series inside every one of 49 workers would cost more than
    the operator call itself; writing them once turns that into a page-cache
    read shared across the pool.
    """
    d = cache_dir(panel.tf)
    os.makedirs(d, exist_ok=True)
    series = build(panel, include_aux)
    for k, v in series.items():
        atomic_save(os.path.join(d, f"{k}.npy"), lambda f, v=v: np.save(f, v))
    atomic_save(os.path.join(d, "index.json"),
                lambda f: f.write(json.dumps(sorted(series), indent=2).encode()))
    return sorted(series)


def names(tf):
    path = os.path.join(cache_dir(tf), "index.json")
    if os.path.exists(path):
        return json.load(open(path))
    return []


def load(tf, name, mmap=True):
    return np.load(os.path.join(cache_dir(tf), f"{name}.npy"),
                   mmap_mode="r" if mmap else None)


if __name__ == "__main__":
    from panel import Panel

    for tf in (sys.argv[1:] or ["1h"]):
        p = Panel(tf)
        series = build(p)
        print(f"{p!r}  {len(series)} base series")
        for k in sorted(series):
            v = series[k]
            finite = np.isfinite(v)
            frac = finite.mean()
            med = float(np.nanmedian(v)) if finite.any() else float("nan")
            sd = float(np.nanstd(v)) if finite.any() else float("nan")
            print(f"  {k:16s} finite {frac:6.2%}  median {med:12.6g}  sd {sd:12.6g}")
        d = cache_dir(tf)
        os.makedirs(d, exist_ok=True)
        for k, v in series.items():
            atomic_save(os.path.join(d, f"{k}.npy"), lambda f, v=v: np.save(f, v))
        atomic_save(os.path.join(d, "index.json"),
                    lambda f: f.write(json.dumps(sorted(series), indent=2).encode()))
        print(f"  cached to {d}")
