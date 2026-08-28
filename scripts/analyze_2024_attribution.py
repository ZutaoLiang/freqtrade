#!/usr/bin/env python3
"""Signal-level attribution of the XsMomEnsembleV1 2024 failure.

Same basket construction as the strategy (3d/14d/30d rank ensemble, top/bottom
5, 30M point-in-time volume floor, daily at 00:00), evaluated as an event
study on the expanded binance-hist universe.  For each year 2023-2025 (and
2026 from the live datadir):

  longR   mean next-24h return of the long basket
  shortR  mean next-24h return of the short basket
  uniR    mean next-24h return of all eligible pairs (the universe drift)
  aL      long-basket alpha  = longR - uniR
  aS      short-basket alpha = uniR - shortR

If aL and aS are both negative in 2024, momentum RANKS were inverted (tops
mean-reverted / bottoms bounced) — a market-shape statement independent of
fees, sizing or stops.  Monthly detail printed for 2024.
Float32 panels keep memory inside the host budget.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

MIN_QVOL = 30e6


def load_dir(pdir: str, a: str, b: str):
    closes, qvols = {}, {}
    for f in sorted(glob.glob(f"{pdir}/*-1h-futures.feather")):
        p = os.path.basename(f).split("-1h-")[0]
        try:
            px = pd.read_feather(f, columns=["date", "close", "volume"]).set_index("date").sort_index()
        except Exception:
            continue
        if px.empty:
            continue
        px = px[(px.index >= pd.Timestamp(a, tz="UTC")) & (px.index < pd.Timestamp(b, tz="UTC"))]
        if len(px) < 24 * 40:
            continue
        px = px[~px.index.duplicated(keep="last")]
        closes[p] = px["close"].astype(np.float32)
        qvols[p] = (px["close"] * px["volume"]).astype(np.float32)
    close = pd.DataFrame(closes).sort_index()
    qvol = pd.DataFrame(qvols).sort_index().reindex(close.index)
    return close, qvol


def year_attribution(close, qvol, y0, label_months=None):
    elig = qvol.rolling(24, min_periods=12).sum().rolling(168, min_periods=84).mean() > MIN_QVOL
    ens = sum((close / close.shift(n) - 1.0).where(elig).rank(axis=1, pct=True)
              for n in (72, 336, 720)) / 3
    daily = close.index.hour == 0
    ensd = ens[daily]
    closed = close[daily]
    fwd = closed.shift(-1) / closed - 1.0
    ranks = ensd.rank(axis=1)
    nn = ensd.notna().sum(axis=1)
    k = 5
    is_long = ranks.ge(nn - k + 1, axis=0) & ensd.notna()
    is_short = ranks.le(k) & ensd.notna()
    ok = nn >= 12
    rows = {}
    def seg_stats(mask_rows):
        m = mask_rows & ok
        lr = fwd.where(is_long)[m].stack().mean()
        sr = fwd.where(is_short)[m].stack().mean()
        ur = fwd.where(ensd.notna())[m].stack().mean()
        return lr, sr, ur
    lr, sr, ur = seg_stats(pd.Series(ensd.index >= y0, index=ensd.index))
    print(f"  full: longR={lr*100:+.2f}%/d shortR={sr*100:+.2f}%/d uniR={ur*100:+.2f}%/d "
          f"aL={(lr-ur)*100:+.2f} aS={(ur-sr)*100:+.2f}")
    if label_months:
        for mth in label_months:
            m = pd.Series(ensd.index.strftime("%Y-%m") == mth, index=ensd.index)
            if m.sum() < 5:
                continue
            lr, sr, ur = seg_stats(m)
            print(f"  {mth}: aL={(lr-ur)*100:+.2f} aS={(ur-sr)*100:+.2f} uni={ur*100:+.2f}%/d")


def main() -> None:
    jobs = [
        ("2023", "user_data/data/binance-hist/futures", "2022-11-20", "2024-01-01", None),
        ("2024", "user_data/data/binance-hist/futures", "2023-11-20", "2025-01-01",
         [f"2024-{m:02d}" for m in range(1, 13)]),
        ("2025", "user_data/data/binance-hist/futures", "2024-11-20", "2026-01-01", None),
        ("2026", "user_data/data/binance/futures", "2026-01-01", "2026-08-14", None),
    ]
    for tag, pdir, a, b, months in jobs:
        close, qvol = load_dir(pdir, a, b)
        # clip evaluation rows to the target year
        print(f"### {tag} (pairs={close.shape[1]})")
        y0 = pd.Timestamp(f"{tag}-01-01", tz="UTC")
        y1 = pd.Timestamp(f"{int(tag)+1}-01-01", tz="UTC")
        year_attribution(close[close.index < y1], qvol[qvol.index < y1], y0, months)
        del close, qvol


if __name__ == "__main__":
    main()
