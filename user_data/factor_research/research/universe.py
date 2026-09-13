"""Causal tradeable-universe mask.

A factor measured on pairs that were not actually tradeable at the time is
measuring nothing. Every criterion here uses trailing windows only, so the mask
at bar t depends on no information after t -- which also keeps delisted pairs
in the sample for as long as they really traded, rather than silently dropping
them and introducing survivorship bias.
"""
import json
import os

import numpy as np
import pandas as pd

from panel import Panel, CACHE

# Binance lists perpetuals with thin, erratic books; the opening stretch is not
# representative of the pair's normal behaviour and would dominate any factor
# fitted on it.
MIN_HISTORY_DAYS = 30
LIQUIDITY_WINDOW_DAYS = 7
# Expressed per DAY, not per bar. A per-bar threshold silently shrinks the
# universe as the timeframe shortens -- 1e5/bar admitted 529 pairs at 1d but
# only 236 at 30m, which reflects bar duration, not liquidity. Scaling by bars
# per day keeps the tradeable set comparable across timeframes.
MIN_DAILY_QUOTE_VOLUME = 2.4e6


def _bars_per_day(tf):
    return int(pd.Timedelta("1D") / pd.Timedelta(tf.replace("m", "min").replace("d", "D")))


def build(tf, min_history_days=MIN_HISTORY_DAYS,
          liquidity_window_days=LIQUIDITY_WINDOW_DAYS,
          min_daily_quote_volume=MIN_DAILY_QUOTE_VOLUME):
    p = Panel(tf)
    bpd = _bars_per_day(tf)
    min_quote_volume = min_daily_quote_volume / bpd
    close = np.asarray(p["close"])
    qv = np.asarray(p["quote_volume"])

    present = np.isfinite(close)

    # Bars elapsed since each symbol's first observation, per symbol.
    first = np.argmax(present, axis=0).astype("float64")
    first[~present.any(axis=0)] = np.inf
    age = np.arange(len(p.index))[:, None] - first[None, :]
    seasoned = age >= min_history_days * bpd

    # Trailing median quote volume; median rather than mean so a single
    # wash-traded bar cannot lift a dead pair into the universe.
    win = max(1, liquidity_window_days * bpd)
    qv_df = pd.DataFrame(qv)
    liquid = qv_df.rolling(win, min_periods=win // 2).median().to_numpy() >= min_quote_volume

    mask = present & seasoned & liquid

    out = os.path.join(CACHE, tf)
    np.save(os.path.join(out, "universe_mask.npy"), mask)
    meta = {
        "min_history_days": min_history_days,
        "liquidity_window_days": liquidity_window_days,
        "min_daily_quote_volume": min_daily_quote_volume,
        "min_quote_volume_per_bar": min_quote_volume,
        "bars": int(mask.shape[0]),
        "symbols": int(mask.shape[1]),
        "median_width": float(np.median(mask.sum(axis=1))),
        "max_width": int(mask.sum(axis=1).max()),
        "ever_tradeable": int((mask.any(axis=0)).sum()),
    }
    with open(os.path.join(out, "universe_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return p, mask, meta


def load(tf):
    return np.load(os.path.join(CACHE, tf, "universe_mask.npy"), mmap_mode="r")


if __name__ == "__main__":
    import sys
    for tf in (sys.argv[1:] or ["1h"]):
        p, mask, meta = build(tf)
        width = mask.sum(axis=1)
        print(f"\n=== {tf} universe mask ===")
        print(f"panel                : {p}")
        print(f"ever tradeable       : {meta['ever_tradeable']} / {len(p.symbols)} symbols")
        print(f"width median/max     : {meta['median_width']:.0f} / {meta['max_width']}")
        q = pd.Series(width, index=p.index).resample("QE").median()
        print("median width by quarter:")
        for ts, v in q.items():
            print(f"  {ts.date()} : {v:.0f}")
