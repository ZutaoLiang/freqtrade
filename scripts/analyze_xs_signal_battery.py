#!/usr/bin/env python3
"""Battery scan of cross-sectional signals over both universes, 4 half-year splits.

Acceptance rule (pre-registered): a signal only advances to a freqtrade
backtest if its net Sharpe is positive in ALL FOUR splits (25H1/25H2 on the
2025 rotating pool, 26H1/26H2 on the 2026 all-perps set).  Everything else is
recorded as falsified.  Uses the loaders from analyze_smallcap_carry_xs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_smallcap_carry_xs import load_universe, annualized_sharpe, FEE_SIDE, MIN_QVOL


def portfolio_net(close, fund, elig, score, k=5, reb_ev=3):
    ret_fwd = close.shift(-1) / close - 1.0
    fund_fwd = fund.shift(-1)
    s = score.where(elig)
    ranks = s.rank(axis=1)
    nn = s.notna().sum(axis=1)
    w = pd.DataFrame(0.0, index=s.index, columns=s.columns)
    w[ranks.le(k, axis=0).values & s.notna().values] = 1.0 / k
    w[ranks.ge(nn - k + 1, axis=0).values & s.notna().values] = -1.0 / k
    w[nn < 2 * k + 2] = 0.0
    if reb_ev > 1:
        keep = np.arange(len(s.index)) % reb_ev == 0
        w = w.where(pd.Series(keep, index=s.index), np.nan).ffill().fillna(0.0)
    w = w.where(close.notna(), 0.0)
    pnl = (w * ret_fwd).sum(axis=1) + (w * -fund_fwd).sum(axis=1)
    to = (w - w.shift(1)).abs().sum(axis=1)
    return pnl - to * FEE_SIDE, to.mean()


def signals(close, fund, qvol):
    ret1 = close / close.shift(3) - 1.0
    vol30 = ret1.rolling(90, min_periods=45).std()
    high30 = close.rolling(90, min_periods=45).max()
    out = {
        # momentum family: long winners => score = -mom (low rank = long)
        "mom1d": -(close / close.shift(3) - 1.0),
        "mom3d": -(close / close.shift(9) - 1.0),
        "mom14d": -(close / close.shift(42) - 1.0),
        "mom30d": -(close / close.shift(90) - 1.0),
        # low-vol: long low vol
        "lowvol30d": vol30,
        # distance from 30d high: long near-high (score = distance)
        "nearhigh30d": (high30 - close) / high30,
        # reversal: long losers
        "rev3d": close / close.shift(9) - 1.0,
        # carry: long low funding
        "carry7d": fund.rolling(21, min_periods=9).mean(),
        # volume growth: long rising volume
        "volgrow": -(qvol / qvol.shift(21) - 1.0),
        # funding momentum blend: long winners whose funding is still low
        "mom14d_x_carry": (-(close / close.shift(42) - 1.0)).rank(axis=1, pct=True) * 0.5
                          + fund.rolling(21, min_periods=9).mean().rank(axis=1, pct=True) * 0.5,
    }
    return out


def main() -> None:
    uni = {
        "2025": ("user_data/data/binance-2025/futures", "2025-02-01", "2026-01-31",
                 {"25H1": ("2025-02-01", "2025-08-01"), "25H2": ("2025-08-01", "2026-02-01")}),
        "2026": ("user_data/data/binance/futures", "2026-01-15", "2026-07-31",
                 {"26H1": ("2026-01-15", "2026-05-01"), "26H2": ("2026-05-01", "2026-08-01")}),
    }
    results: dict[str, dict[str, float]] = {}
    for _, (pdir, a, b, splits_def) in uni.items():
        close, qvol, fund = load_universe(pdir, a, b)
        elig = qvol > MIN_QVOL
        idx = close.index
        splits = {lbl: (idx >= pd.Timestamp(x, tz="UTC")) & (idx < pd.Timestamp(y, tz="UTC"))
                  for lbl, (x, y) in splits_def.items()}
        for name, score in signals(close, fund, qvol).items():
            net, to = portfolio_net(close, fund, elig, score)
            for lbl, mask in splits.items():
                results.setdefault(name, {})[lbl] = annualized_sharpe(net[mask])
            results[name]["to_" + lbl[:2]] = to
    cols = ["25H1", "25H2", "26H1", "26H2"]
    print(f"{'signal':>16s} " + " ".join(f"{c:>7s}" for c in cols) + "  all+")
    for name, r in results.items():
        vals = [r.get(c, float('nan')) for c in cols]
        ok = all(v > 0 for v in vals if not np.isnan(v)) and not any(np.isnan(v) for v in vals)
        print(f"{name:>16s} " + " ".join(f"{v:7.2f}" for v in vals) + ("   <<< " if ok else ""))


if __name__ == "__main__":
    main()
