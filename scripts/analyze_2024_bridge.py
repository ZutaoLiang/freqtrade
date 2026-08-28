#!/usr/bin/env python3
"""Bridge the gap: gross signal alpha was POSITIVE in 2024, backtest lost 60%.

Decomposition ladder, per year, all on the expanded universe and the exact
strategy baskets (daily 00:00, k=5, 30M floor):

  A  arithmetic: mean(long fwd) - mean(short fwd), x365            (what the
     attribution study showed)
  B  compounded, signal prices: daily portfolio return at 0.95 gross split
     50/50, entered at the 00:00 close, no fees -> volatility drag isolated
  C  B + fees (0.07%/side on daily turnover)
  D  C but entered/exited at the 01:00 close (the backtest acts one candle
     after the signal) -> execution-lag cost isolated

D should approximate the freqtrade result; the step where the number
collapses names the killer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_2024_attribution import load_dir, MIN_QVOL

FEE_SIDE = 0.0007


def run_year(tag, pdir, a, b):
    close, qvol = load_dir(pdir, a, b)
    elig = qvol.rolling(24, min_periods=12).sum().rolling(168, min_periods=84).mean() > MIN_QVOL
    ens = sum((close / close.shift(n) - 1.0).where(elig).rank(axis=1, pct=True)
              for n in (72, 336, 720)) / 3
    y0 = pd.Timestamp(f"{tag}-01-01", tz="UTC")

    for lag, lbl in [(0, "signal px"), (1, "lag 1h px")]:
        d0 = close.index.hour == 0
        sigd = ens[d0]
        # execution index: same day at 00:00+lag hours
        exec_idx = sigd.index + pd.Timedelta(hours=lag)
        px = close.reindex(exec_idx)
        px.index = sigd.index
        fwd = px.shift(-1) / px - 1.0
        ranks = sigd.rank(axis=1)
        nn = sigd.notna().sum(axis=1)
        k = 5
        w = pd.DataFrame(0.0, index=sigd.index, columns=sigd.columns)
        w[(ranks.ge(nn - k + 1, axis=0) & sigd.notna()).values] = 1.0 / (2 * k)
        w[(ranks.le(k) & sigd.notna()).values] = -1.0 / (2 * k)
        w[nn < 12] = 0.0
        w = w.where(px.notna(), 0.0)
        gross = 0.95 * 2  # 0.95 equity split across both legs at leverage 1
        ret_p = (w * fwd).sum(axis=1) * gross
        to = (w - w.shift(1)).abs().sum(axis=1) * gross
        yr = sigd.index >= y0
        r = ret_p[yr].dropna()
        t = to[yr]
        if lag == 0:
            print(f"### {tag}  (days={len(r)})")
            print(f"  A arithmetic net-leg spread: {r.mean()*365*100:+7.1f}%/yr  "
                  f"daily sigma={r.std()*100:.2f}%")
            eq = (1 + r).prod() - 1
            print(f"  B compounded no fees      : {eq*100:+7.1f}%")
            eqf = (1 + r - t[r.index] * FEE_SIDE).prod() - 1
            print(f"  C compounded + fees       : {eqf*100:+7.1f}%  (turnover/day={t.mean():.2f})")
        else:
            eqf = (1 + r - t[r.index] * FEE_SIDE).prod() - 1
            print(f"  D lag-1h + fees compounded: {eqf*100:+7.1f}%")


def main() -> None:
    run_year("2023", "user_data/data/binance-hist/futures", "2022-11-20", "2024-01-01")
    run_year("2024", "user_data/data/binance-hist/futures", "2023-11-20", "2025-01-01")
    run_year("2025", "user_data/data/binance-hist/futures", "2024-11-20", "2026-01-01")
    run_year("2026", "user_data/data/binance/futures", "2026-01-01", "2026-08-14")


if __name__ == "__main__":
    main()
