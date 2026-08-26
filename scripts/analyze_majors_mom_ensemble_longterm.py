#!/usr/bin/env python3
"""Does the XsMomEnsembleV1 multi-horizon trick help on the majors, 2018-2026?

The long-term regime study showed single-horizon 30d XS momentum on the 20
majors decayed to negative in 2024-2026.  This tests whether the ensemble
construction that survived on smallcaps (equal-weight rank blend of three
horizons) changes that picture on the majors' 8-year daily sample:

  XS single horizons: 3d, 7d, 14d, 30d, 90d
  XS ensembles: (3,14,30)d  — the smallcap recipe verbatim
                (7,30,90)d  — slower variant scaled to majors
  TSMOM single 90d vs ensemble sign(30)+sign(90)+sign(180)

All k=5 long/short (XS), daily signals, weekly rebalance, net 0.07%/side.
Per-year and full-sample Sharpe.  No parameter search beyond this list.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_regime_indicator_longterm import load, xs_weights, weekly, net_pnl


def yearly_report(name: str, pnl: pd.Series) -> None:
    cells = []
    for y in range(2019, 2027):
        seg = pnl[pnl.index.year == y].dropna()
        if len(seg) < 60 or seg.std() == 0:
            cells.append(f"{str(y)[2:]}:  nan")
            continue
        cells.append(f"{str(y)[2:]}:{seg.mean()/seg.std()*np.sqrt(365):+5.1f}")
    full = pnl.dropna()
    fs = full.mean() / full.std() * np.sqrt(365)
    ann = full.mean() * 365 * 100
    print(f"  {name:16s} " + " ".join(cells) + f" | full shp {fs:+.2f} ann {ann:+6.1f}%")


def main() -> None:
    close = load()
    fwd = close.shift(-1) / close - 1.0

    print("=== XS momentum on 20 majors, k=5, weekly rebalance, net fees ===")
    singles = {}
    for d in (3, 7, 14, 30, 90):
        score = -(close / close.shift(d) - 1.0)
        pnl = net_pnl(weekly(xs_weights(score, 5)), fwd)
        singles[d] = score
        yearly_report(f"mom{d}d", pnl)

    for combo in [(3, 14, 30), (7, 30, 90)]:
        ens = sum(singles[d].rank(axis=1, pct=True) for d in combo) / len(combo)
        pnl = net_pnl(weekly(xs_weights(ens, 5)), fwd)
        yearly_report(f"ens{combo}", pnl)

    print("\n=== TSMOM on 20 majors, equal weight, weekly rebalance ===")
    def tsmom(sig):
        n_act = sig.notna().sum(axis=1).replace(0, np.nan)
        return net_pnl(weekly(sig.div(n_act, axis=0).fillna(0.0)), fwd)
    yearly_report("tsm90d", tsmom(np.sign(close / close.shift(90) - 1.0)))
    ens_sig = sum(np.sign(close / close.shift(d) - 1.0) for d in (30, 90, 180)) / 3
    yearly_report("tsm(30,90,180)", tsmom(ens_sig))


if __name__ == "__main__":
    main()
