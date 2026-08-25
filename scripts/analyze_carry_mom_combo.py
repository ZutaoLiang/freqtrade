#!/usr/bin/env python3
"""Combo study: carry + momentum cross-sectional portfolio, plus controls.

Findings so far: XS funding carry on the 15 majors nets ~+0.7 Sharpe in 2025
but ~0 in 2026; XS 7d momentum is the mirror image.  Their P&L streams look
anti-correlated across years, so test the 50/50 combo explicitly, plus:

* per-pair demeaned funding (subtract each pair's trailing 30d own mean, so
  chronically-high-funding names like ZEC/XMR don't sit permanently short)
* time-series momentum per pair (sign of 20d return, vol-weighted)
* carry+mom rank blend at several mixes and rebalance frequencies

All net of 0.07%/side taker fee.  Runs in seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_funding_carry_event_study import PAIRS, load_pair, FEE_SIDE, annualized_sharpe


def main() -> None:
    data = {p: load_pair(p) for p in PAIRS}
    idx = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in data.values()])))
    close = pd.DataFrame({p: data[p]["close"] for p in PAIRS}, index=idx)
    fund = pd.DataFrame({p: data[p]["funding"] for p in PAIRS}, index=idx)
    ret_fwd = close.shift(-1) / close - 1.0
    fund_fwd = fund.shift(-1)
    years = {"2025": idx.year == 2025, "2026": idx.year == 2026}

    def run(w: pd.DataFrame, name: str, reb_ev: int = 3) -> pd.Series:
        if reb_ev > 1:
            keep = np.arange(len(idx)) % reb_ev == 0
            w = w.where(pd.Series(keep, index=idx), np.nan).ffill().fillna(0.0)
        pnl = (w * ret_fwd).sum(axis=1) + (w * -fund_fwd).sum(axis=1)
        to = (w - w.shift(1)).abs().sum(axis=1)
        net = pnl - to * FEE_SIDE
        cells = []
        for yr, mask in years.items():
            s = net[mask].dropna()
            cells += [f"{s.mean()*3*365*100:8.2f}%", f"shp={annualized_sharpe(s, 3*365):6.2f}"]
        print(f"  {name:28s} 2025: {cells[0]} {cells[1]} | 2026: {cells[2]} {cells[3]} | to={to.mean():.3f}")
        return net

    def rank_weights(score: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        ranks = score.rank(axis=1)
        nn = score.notna().sum(axis=1)
        w = pd.DataFrame(0.0, index=idx, columns=close.columns)
        w[ranks.le(k, axis=0).values & score.notna().values] = 1.0 / k
        w[ranks.ge(nn - k + 1, axis=0).values & score.notna().values] = -1.0 / k
        return w

    trail = fund.rolling(21, min_periods=10).mean()          # 7d carry signal
    mom = close / close.shift(21) - 1.0                       # 7d momentum
    mom30 = close / close.shift(90) - 1.0                     # 30d momentum
    demeaned = trail - fund.rolling(90, min_periods=45).mean()  # 7d vs own 30d mean

    print("=== components (rebalance 1d) ===")
    net_carry = run(rank_weights(trail), "carry7d rank5")
    net_mom = run(rank_weights(-mom), "mom7d rank5")
    net_mom30 = run(rank_weights(-mom30), "mom30d rank5")
    run(rank_weights(demeaned), "carry demeaned rank5")

    print("\n=== P&L correlation carry vs mom7d ===")
    both = pd.concat([net_carry, net_mom], axis=1).dropna()
    print(f"  corr = {both.corr().iloc[0, 1]:.3f}")

    print("\n=== rank blends carry & mom (rebalance 1d, k=5) ===")
    rc = trail.rank(axis=1, pct=True)
    for lam in [0.3, 0.5, 0.7]:
        for mom_sig, mlbl in [(mom, "7d"), (mom30, "30d")]:
            rm = (-mom_sig).rank(axis=1, pct=True)
            blend = lam * rc + (1 - lam) * rm
            run(rank_weights(blend), f"blend c{lam:.0%}+m{mlbl}")

    print("\n=== time-series momentum per pair (vol-scaled, rebalance 1d) ===")
    ret1 = close / close.shift(3) - 1.0
    vol = ret1.rolling(90, min_periods=45).std() * np.sqrt(3 * 365)
    for look, lbl in [(30, "10d"), (60, "20d"), (180, "60d")]:
        sig = np.sign(close / close.shift(look) - 1.0)
        w = (sig * (0.15 / vol)).clip(-0.3, 0.3).fillna(0.0) / len(PAIRS) * 4
        run(w, f"tsmom {lbl} volscaled")


if __name__ == "__main__":
    main()
