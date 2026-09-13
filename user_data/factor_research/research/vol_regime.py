"""Causal market-volatility regime, frozen in reports/PREREGISTRATION_vol_regime.md.

Primary: rv30 = std of BTCUSDT daily log returns over the trailing 30 days; vol_pct = percentile rank
of rv30 within the trailing 365 daily values; LOW <= 0.33, HIGH >= 0.67, MID between. Everything
trailing, so the state at bar t uses no information after t.

Alternatives reported alongside (never used to select): cross-sectional dispersion of trailing 7-day
returns, and the mean pair volatility, each ranked the same way.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel import CACHE, Panel  # noqa: E402

RV_DAYS, RANK_DAYS = 30, 365
LO_Q, HI_Q = 0.33, 0.67
MIN_OBS = 120


def _pct(series, rank_days=RANK_DAYS):
    return series.rolling(rank_days, min_periods=MIN_OBS).rank(pct=True)


def daily_states(rv_days=RV_DAYS, rank_days=RANK_DAYS, lo_q=LO_Q, hi_q=HI_Q, kind="btc"):
    """Regime state per DAY as a Series of {'LOW','MID','HIGH','NA'} on the 1d_long index."""
    p = Panel("1d_long")
    idx, syms = p.index, p.symbols
    close = np.asarray(p["close"], dtype="float64")
    if kind == "btc":
        b = pd.Series(close[:, syms.index("BTCUSDT")], index=idx).ffill()
        rv = np.log(b).diff().rolling(rv_days).std()
    else:
        mask = np.asarray(np.load(os.path.join(CACHE, "1d_long", "universe_mask.npy"), mmap_mode="r"))
        r = pd.DataFrame(close, index=idx).pct_change()
        r = r.where(pd.DataFrame(mask, index=idx))
        if kind == "dispersion":
            rv = r.rolling(7).sum().std(axis=1)
        elif kind == "meanvol":
            rv = r.rolling(rv_days).std().mean(axis=1)
        else:
            raise ValueError(kind)
    q = _pct(rv, rank_days)
    return pd.Series(np.where(q.isna(), "NA", np.where(q <= lo_q, "LOW", np.where(q >= hi_q, "HIGH", "MID"))),
                     index=idx, name="state")


def states_for(tf, **kw):
    """Regime state per bar of panel `tf`, forward-filled from the daily state."""
    s = daily_states(**kw)
    p = Panel(tf)
    return s.reindex(p.index, method="ffill").fillna("NA").to_numpy()


if __name__ == "__main__":
    for kind in ("btc", "dispersion", "meanvol"):
        s = daily_states(kind=kind)
        v = s[s != "NA"]
        print(f"{kind:11s} {len(v)} days  " + "  ".join(f"{k} {n} ({n/len(v):.0%})" for k, n in v.value_counts().items()))
    s = daily_states()
    for lab, lo, hi in [("train", "2022-11-01", "2025-01-01"), ("valid", "2025-01-01", "2026-01-01"), ("holdout", "2026-01-01", "2026-08-17")]:
        m = (s.index >= pd.Timestamp(lo, tz="UTC")) & (s.index < pd.Timestamp(hi, tz="UTC"))
        print(f"  {lab}: " + "  ".join(f"{k} {n}" for k, n in s[m].value_counts().items()))
