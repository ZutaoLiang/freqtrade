#!/usr/bin/env python3
"""Portfolio-level simulation of the short-young-listings effect.

Rule: short every pair aged 8..30 days since first candle whose trailing 7d
mean 24h quote volume exceeds the floor; equal weight across active shorts,
at most MAX_POS concurrent, gross exposure 1.0 when full.  Daily positions,
price P&L + funding received - taker fees on turnover.  Idle days are flat.

Reported per half-year: annualized return on gross, Sharpe over ACTIVE days
only, number of active days, and per-name trade count.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

MIN_QVOL = 30e6
FEE_SIDE = 0.0007
MAX_POS = 5
AGE_LO, AGE_HI = 8, 30


def run(pdir, a, b, cuts):
    closes, qvols, funds, first = {}, {}, {}, {}
    for f in sorted(glob.glob(f"{pdir}/*-1h-futures.feather")):
        p = os.path.basename(f).split("-1h-")[0]
        try:
            px = pd.read_feather(f, columns=["date", "close", "volume"]).set_index("date").sort_index()
        except Exception:
            continue
        if px.empty:
            continue
        px = px[~px.index.duplicated(keep="last")]
        first[p] = px.index.min()
        closes[p] = px["close"]
        qvols[p] = px["close"] * px["volume"]
        ff = f"{pdir}/{p}-1h-funding_rate.feather"
        if os.path.exists(ff):
            try:
                fr = pd.read_feather(ff, columns=["date", "open"]).set_index("date").sort_index()
                funds[p] = fr["open"][~fr.index.duplicated(keep="last")]
            except Exception:
                pass
    close = pd.DataFrame(closes).sort_index()
    close = close[(close.index >= pd.Timestamp(a, tz="UTC")) & (close.index <= pd.Timestamp(b, tz="UTC"))]
    daily = close[close.index.hour == 0]
    qvol = pd.DataFrame(qvols).sort_index().reindex(close.index)
    qtrail = qvol.rolling(24, min_periods=12).sum().rolling(168, min_periods=84).mean()
    elig = (qtrail > MIN_QVOL)[close.index.hour == 0]
    fund = pd.DataFrame(funds).sort_index().reindex(close.index).fillna(0.0)
    fund_daily = fund.rolling(24, min_periods=1).sum()[close.index.hour == 0]  # funding paid per day
    age = pd.DataFrame({p: (daily.index - first[p]).days for p in daily.columns}, index=daily.index)

    young = (age >= AGE_LO) & (age <= AGE_HI) & elig & daily.notna()
    # cap at MAX_POS: keep the youngest ones
    def cap_row(row_young, row_age):
        names = [c for c in row_young.index if row_young[c]]
        if len(names) > MAX_POS:
            names = sorted(names, key=lambda c: row_age[c])[:MAX_POS]
        return names

    fwd = daily.shift(-1) / daily - 1.0
    fund_recv = fund_daily.shift(-1)  # funding over the held day; short receives +rate
    w = pd.DataFrame(0.0, index=daily.index, columns=daily.columns)
    for t in daily.index:
        names = cap_row(young.loc[t], age.loc[t])
        for c in names:
            w.loc[t, c] = -1.0 / max(len(names), 1)
    pnl = (w * fwd).sum(axis=1) + (w * -fund_recv).sum(axis=1)
    to = (w - w.shift(1)).abs().sum(axis=1)
    net = pnl - to * FEE_SIDE
    active = w.abs().sum(axis=1) > 0
    for lbl, x, y in cuts:
        m = (daily.index >= pd.Timestamp(x, tz="UTC")) & (daily.index < pd.Timestamp(y, tz="UTC"))
        seg = net[m]
        act = active[m]
        sa = seg[act].dropna()
        shp = sa.mean() / sa.std() * np.sqrt(365) if len(sa) > 5 and sa.std() > 0 else float("nan")
        print(f"  {lbl}: total={seg.sum()*100:+7.2f}% activedays={act.sum():3d}/{m.sum():3d} "
              f"mean/activeday={sa.mean()*100:+6.3f}% shp(act)={shp:5.2f}")


def main() -> None:
    print("### 2025 pool")
    run("user_data/data/binance-2025/futures", "2025-02-01", "2026-01-31",
        [("25H1", "2025-02-01", "2025-08-01"), ("25H2", "2025-08-01", "2026-02-01")])
    print("### 2026 all-perps")
    run("user_data/data/binance/futures", "2026-01-15", "2026-08-13",
        [("26H1", "2026-01-15", "2026-05-01"), ("26H2", "2026-05-01", "2026-08-14")])


if __name__ == "__main__":
    main()
