"""Causal universe screens for the time-series factor study.

Every screen is a (T, N) boolean mask on the panel grid, intersected with the tradeable
universe mask, and depends only on data up to and including the bar it is evaluated at.
Definitions are fixed in reports/PREREGISTRATION_ts_screen.md.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import operators as ops  # noqa: E402
from panel import CACHE, Panel  # noqa: E402

AGE_DAYS = 90
VOL_DAYS = 7
FR_BPS = 3.0
TREND_WIN_H = 720
ACT_SHORT_D, ACT_LONG_D = 1, 30
LIQ_TIERS = (30, 50, 100, 200)


def _bpd(tf):
    tf = tf.split("_")[0]
    return int(pd.Timedelta("1D") / pd.Timedelta(tf.replace("m", "min").replace("d", "D")))


def _cs_tercile(x, mask):
    """Cross-sectional tercile membership at each bar inside the mask: (lo, hi) bool arrays."""
    v = np.where(mask & np.isfinite(x), x, np.nan)
    lo_q = np.nanpercentile(v, 100 / 3, axis=1, keepdims=True)
    hi_q = np.nanpercentile(v, 200 / 3, axis=1, keepdims=True)
    return (v <= lo_q) & mask, (v >= hi_q) & mask


def build(tf, cache=True):
    p = Panel(tf)
    bpd = _bpd(tf)
    mask = np.asarray(np.load(os.path.join(CACHE, tf, "universe_mask.npy"), mmap_mode="r")).astype(bool)
    close = np.asarray(p["close"], dtype="float64")
    qv = np.asarray(p["quote_volume"], dtype="float64")
    out = {"all": mask}
    # liquidity tiers (same causal rank as tiers.py, recomputed here so 1h_hist works too)
    win = max(1, 7 * bpd)
    med = pd.DataFrame(qv).rolling(win, min_periods=win // 2).median().to_numpy()
    med = np.where(mask & np.isfinite(med), med, -np.inf)
    order = np.argsort(-med, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(order.shape[1])[None, :], axis=1)
    rank = np.where(med > -np.inf, rank, 10 ** 6)
    for k in LIQ_TIERS:
        out[f"top{k}"] = (rank < k) & mask
    # age since first tradeable bar
    first = np.where(mask.any(axis=0), mask.argmax(axis=0), 10 ** 9)
    bars_since = np.arange(mask.shape[0])[:, None] - first[None, :]
    out["young"] = (bars_since >= 0) & (bars_since < AGE_DAYS * bpd) & mask
    out["mature"] = (bars_since >= AGE_DAYS * bpd) & mask
    # volatility regime
    ret1 = close / ops.ts_delay(close, 1) - 1.0
    vol = ops.ts_std(ret1, VOL_DAYS * bpd)
    out["vol_lo"], out["vol_hi"] = _cs_tercile(vol, mask)
    # funding regime (panel carries the last settled rate forward)
    fields = set(p.meta.get("fields", []))
    if "funding_rate" in fields:
        fr = np.asarray(p["funding_rate"], dtype="float64")
        if (fr != 0).mean() < 0.5:   # settlement-only storage (1h_hist): carry the last settled rate forward
            fr = pd.DataFrame(np.where(fr != 0, fr, np.nan)).ffill(limit=8 * bpd).to_numpy()
        fr24 = ops.ts_mean(fr, max(2, bpd))
        out["fr_neg"] = (fr24 <= -FR_BPS * 1e-4) & mask
        out["fr_pos"] = (fr24 >= FR_BPS * 1e-4) & mask
        out["fr_flat"] = (np.abs(fr24) < FR_BPS * 1e-4) & np.isfinite(fr24) & mask
    # trend regime
    tr = ops.ts_rank(close, int(TREND_WIN_H * bpd / 24))
    out["trend_up"] = (tr >= 0.67) & mask
    out["trend_dn"] = (tr <= 0.33) & mask
    # activity
    act = ops.ts_mean(qv, ACT_SHORT_D * bpd) / ops.ts_mean(qv, ACT_LONG_D * bpd)
    out["act_lo"], out["act_hi"] = _cs_tercile(act, mask)
    out = {k: v.astype(bool) for k, v in out.items()}
    if cache:
        d = os.path.join(CACHE, tf, "screens")
        os.makedirs(d, exist_ok=True)
        for k, v in out.items():
            np.save(os.path.join(d, f"{k}.npy"), v)
    return out


def load(tf):
    d = os.path.join(CACHE, tf, "screens")
    if not os.path.isdir(d):
        return build(tf)
    return {f[:-4]: np.load(os.path.join(d, f), mmap_mode="r") for f in sorted(os.listdir(d)) if f.endswith(".npy")}


if __name__ == "__main__":
    for tf in sys.argv[1:] or ["1h"]:
        s = build(tf)
        p = Panel(tf)
        print(tf, {k: f"{v.mean()*100:.1f}% of cells, {v.sum(axis=1).mean():.0f} pairs/bar" for k, v in s.items()})
