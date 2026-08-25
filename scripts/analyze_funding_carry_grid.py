#!/usr/bin/env python3
"""Turnover-reduction grid for the cross-sectional funding-carry portfolio.

The first event study (analyze_funding_carry_event_study.py) showed gross
carry P&L positive in both 2025 and 2026 but fully consumed by fees at
per-event rank rebalancing.  This grid varies:

* trailing funding window: 3d / 7d / 14d
* rebalance frequency: every event (8h) / daily / every 3 days
* portfolio scheme: rank top-bottom k=5, z-score weights, banded k=5
  (enter bottom/top 4, exit only when rank leaves bottom/top 7)

and reports net-of-fee annualized return and Sharpe per calendar year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_funding_carry_event_study import PAIRS, load_pair, FEE_SIDE, annualized_sharpe


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = {p: load_pair(p) for p in PAIRS}
    idx = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in data.values()])))
    close = pd.DataFrame({p: data[p]["close"] for p in PAIRS}, index=idx)
    fund = pd.DataFrame({p: data[p]["funding"] for p in PAIRS}, index=idx)
    ret_fwd = close.shift(-1) / close - 1.0
    fund_fwd = fund.shift(-1)
    return fund, ret_fwd, fund_fwd


def weights_rank(score: pd.DataFrame, k: int) -> pd.DataFrame:
    ranks = score.rank(axis=1)
    nn = score.notna().sum(axis=1)
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    w[ranks.le(k, axis=0).values & score.notna().values] = 1.0 / k
    w[ranks.ge(nn - k + 1, axis=0).values & score.notna().values] = -1.0 / k
    return w


def weights_zscore(score: pd.DataFrame) -> pd.DataFrame:
    z = score.sub(score.mean(axis=1), axis=0).div(score.std(axis=1), axis=0)
    z = z.clip(-2.5, 2.5)
    w = -z.div(z.abs().sum(axis=1), axis=0)  # short high funding; gross exposure 1 per side approx
    return w.fillna(0.0) * 2.0               # scale so |w| sums to ~2 like the rank scheme


def weights_banded(score: pd.DataFrame, k_in: int, k_out: int) -> pd.DataFrame:
    """Enter when rank enters extreme k_in, exit only when it leaves extreme k_out."""
    ranks = score.rank(axis=1)
    nn = score.notna().sum(axis=1)
    rows = []
    held: dict[str, int] = {}
    for t in score.index:
        r = ranks.loc[t]
        n = nn.loc[t]
        if n < 10:
            rows.append({c: 0.0 for c in score.columns})
            held = {}
            continue
        new_held: dict[str, int] = {}
        for c in score.columns:
            rc = r.get(c)
            if pd.isna(rc):
                continue
            side = held.get(c, 0)
            if side == 1 and rc <= k_out:
                new_held[c] = 1
            elif side == -1 and rc >= n - k_out + 1:
                new_held[c] = -1
            if rc <= k_in:
                new_held[c] = 1
            elif rc >= n - k_in + 1:
                new_held[c] = -1
        held = new_held
        longs = [c for c, s in held.items() if s == 1]
        shorts = [c for c, s in held.items() if s == -1]
        row = {c: 0.0 for c in score.columns}
        for c in longs:
            row[c] = 1.0 / max(len(longs), 1)
        for c in shorts:
            row[c] = -1.0 / max(len(shorts), 1)
        rows.append(row)
    return pd.DataFrame(rows, index=score.index)


def main() -> None:
    fund, ret_fwd, fund_fwd = build()
    idx = fund.index
    years = {"2025": idx.year == 2025, "2026": idx.year == 2026}

    print(f"{'window':>7s} {'rebal':>6s} {'scheme':>12s} | "
          f"{'25 net%':>8s} {'25 shp':>7s} | {'26 net%':>8s} {'26 shp':>7s} | {'to/ev':>6s}")
    for win_d, win_ev in [("3d", 9), ("7d", 21), ("14d", 42)]:
        trail = fund.rolling(win_ev, min_periods=win_ev // 2).mean()
        for reb_lbl, reb_ev in [("8h", 1), ("1d", 3), ("3d", 9)]:
            for scheme in ["rank5", "zscore", "band4/7"]:
                score = trail.iloc[::1]
                if scheme == "rank5":
                    w = weights_rank(trail, 5)
                elif scheme == "zscore":
                    w = weights_zscore(trail)
                else:
                    w = weights_banded(trail, 4, 7)
                if reb_ev > 1:
                    keep = np.arange(len(idx)) % reb_ev == 0
                    w = w.where(pd.Series(keep, index=idx), np.nan).ffill().fillna(0.0)
                pnl = (w * ret_fwd).sum(axis=1) + (w * -fund_fwd).sum(axis=1)
                to = (w - w.shift(1)).abs().sum(axis=1)
                net = pnl - to * FEE_SIDE
                cells = []
                for yr, mask in years.items():
                    s = net[mask].dropna()
                    cells += [f"{s.mean()*3*365*100:8.2f}", f"{annualized_sharpe(s, 3*365):7.2f}"]
                print(f"{win_d:>7s} {reb_lbl:>6s} {scheme:>12s} | "
                      f"{cells[0]} {cells[1]} | {cells[2]} {cells[3]} | "
                      f"{to.mean():6.3f}")


if __name__ == "__main__":
    main()
