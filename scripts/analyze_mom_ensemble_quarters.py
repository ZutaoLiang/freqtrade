#!/usr/bin/env python3
"""Multi-horizon momentum ensemble on the high-volume universe, quarterly view.

Single-horizon momentum failed the 4-split rule with horizon-dependent signs.
The standard remedy is an equal-weight multi-horizon ensemble (3d/14d/30d rank
blend).  Pre-registered bar: net Sharpe positive in both full years AND in at
least 6 of the ~7 quarters, before any freqtrade backtest is attempted.
Also reports long-leg-only and short-leg-only decomposition of the ensemble.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_smallcap_carry_xs import load_universe, annualized_sharpe, FEE_SIDE, MIN_QVOL


def rank_weights(s: pd.DataFrame, k: int) -> pd.DataFrame:
    ranks = s.rank(axis=1)
    nn = s.notna().sum(axis=1)
    w = pd.DataFrame(0.0, index=s.index, columns=s.columns)
    w[ranks.le(k, axis=0).values & s.notna().values] = 1.0 / k
    w[ranks.ge(nn - k + 1, axis=0).values & s.notna().values] = -1.0 / k
    w[nn < 2 * k + 2] = 0.0
    return w


def run(close, fund, elig, score, k=5, reb_ev=3, legs="both"):
    ret_fwd = close.shift(-1) / close - 1.0
    fund_fwd = fund.shift(-1)
    w = rank_weights(score.where(elig), k)
    if legs == "long":
        w = w.clip(lower=0.0)
    elif legs == "short":
        w = w.clip(upper=0.0)
    if reb_ev > 1:
        keep = np.arange(len(w.index)) % reb_ev == 0
        w = w.where(pd.Series(keep, index=w.index), np.nan).ffill().fillna(0.0)
    w = w.where(close.notna(), 0.0)
    pnl = (w * ret_fwd).sum(axis=1) + (w * -fund_fwd).sum(axis=1)
    to = (w - w.shift(1)).abs().sum(axis=1)
    return pnl - to * FEE_SIDE, to


def quarters(idx):
    out = {}
    for y in (2025, 2026):
        for q, (m0, m1) in enumerate([(1, 4), (4, 7), (7, 10), (10, 13)], 1):
            a = pd.Timestamp(f"{y}-{m0:02d}-01", tz="UTC")
            b = pd.Timestamp(f"{y + (m1 > 12)}-{(m1 - 1) % 12 + 1:02d}-01", tz="UTC")
            m = (idx >= a) & (idx < b)
            if m.sum() > 60:
                out[f"{y % 100}Q{q}"] = m
    return out


def main() -> None:
    uni = {
        "2025pool": ("user_data/data/binance-2025/futures", "2025-02-01", "2026-01-31"),
        "2026all": ("user_data/data/binance/futures", "2026-01-15", "2026-07-31"),
    }
    for name, (pdir, a, b) in uni.items():
        close, qvol, fund = load_universe(pdir, a, b)
        elig = qvol > MIN_QVOL
        idx = close.index
        qs = quarters(idx)
        mom = {h: -(close / close.shift(n) - 1.0) for h, n in [("3d", 9), ("14d", 42), ("30d", 90)]}
        ens = sum(m.rank(axis=1, pct=True) for m in mom.values()) / 3
        print(f"\n### {name}")
        for lbl, score in [("ens(3/14/30)", ens),
                           ("ens long-leg", ens), ("ens short-leg", ens)]:
            legs = "both" if lbl == "ens(3/14/30)" else ("long" if "long" in lbl else "short")
            net, to = run(close, fund, elig, score, legs=legs)
            parts = [f"{q}:{annualized_sharpe(net[m]):6.2f}" for q, m in qs.items()]
            print(f"  {lbl:14s} " + " ".join(parts) +
                  f"  full:{annualized_sharpe(net):6.2f} ret:{net.mean()*3*365*100:7.1f}% to={to.mean():.3f}")
        for h, m in mom.items():
            net, to = run(close, fund, elig, m)
            parts = [f"{q}:{annualized_sharpe(net[mm]):6.2f}" for q, mm in qs.items()]
            print(f"  mom{h:11s} " + " ".join(parts) + f"  full:{annualized_sharpe(net):6.2f}")


if __name__ == "__main__":
    main()
