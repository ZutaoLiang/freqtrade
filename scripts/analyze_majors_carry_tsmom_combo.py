#!/usr/bin/env python3
"""Half-year-split check of the two surviving majors signals and their combo.

Survivors of the earlier scans (positive in BOTH full years, low turnover):

* XS funding carry, 14d trailing, banded rank (enter extreme 4, exit outside
  extreme 7), rebalanced every 3d
* per-pair time-series momentum, 20d lookback, vol-scaled, rebalanced 1d
* tsmom ensemble 10/20/60d

This script reports the four half-year net Sharpes for each and for equal-vol
combos, plus the correlation matrix of the streams.  Universe: 15 majors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_funding_carry_event_study import PAIRS, load_pair, FEE_SIDE, annualized_sharpe
from analyze_funding_carry_grid import weights_banded


def main() -> None:
    data = {p: load_pair(p) for p in PAIRS}
    idx = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in data.values()])))
    close = pd.DataFrame({p: data[p]["close"] for p in PAIRS}, index=idx)
    fund = pd.DataFrame({p: data[p]["funding"] for p in PAIRS}, index=idx)
    ret_fwd = close.shift(-1) / close - 1.0
    fund_fwd = fund.shift(-1)

    splits = {
        "25H1": (idx >= pd.Timestamp("2025-01-01", tz="UTC")) & (idx < pd.Timestamp("2025-07-01", tz="UTC")),
        "25H2": (idx >= pd.Timestamp("2025-07-01", tz="UTC")) & (idx < pd.Timestamp("2026-01-01", tz="UTC")),
        "26H1": (idx >= pd.Timestamp("2026-01-01", tz="UTC")) & (idx < pd.Timestamp("2026-05-01", tz="UTC")),
        "26H2": (idx >= pd.Timestamp("2026-05-01", tz="UTC")) & (idx < pd.Timestamp("2026-08-01", tz="UTC")),
    }

    def net_of(w: pd.DataFrame, reb_ev: int) -> tuple[pd.Series, float]:
        if reb_ev > 1:
            keep = np.arange(len(idx)) % reb_ev == 0
            w = w.where(pd.Series(keep, index=idx), np.nan).ffill().fillna(0.0)
        pnl = (w * ret_fwd).sum(axis=1) + (w * -fund_fwd).sum(axis=1)
        to = (w - w.shift(1)).abs().sum(axis=1)
        return pnl - to * FEE_SIDE, float(to.mean())

    def report(name: str, net: pd.Series, to: float) -> None:
        parts = [f"{lbl}:{annualized_sharpe(net[m], 3*365):6.2f}" for lbl, m in splits.items()]
        full = annualized_sharpe(net, 3*365)
        print(f"  {name:22s} " + " ".join(parts) + f"  full:{full:6.2f}  to={to:.3f}")

    trail14 = fund.rolling(42, min_periods=21).mean()
    net_carry, to_c = net_of(weights_banded(trail14, 4, 7), 9)
    report("carry14d band 3d", net_carry, to_c)

    ret1 = close / close.shift(3) - 1.0
    vol = ret1.rolling(90, min_periods=45).std() * np.sqrt(3 * 365)
    nets_ts = {}
    for look, lbl in [(30, "10d"), (60, "20d"), (180, "60d")]:
        sig = np.sign(close / close.shift(look) - 1.0)
        w = (sig * (0.15 / vol)).clip(-0.3, 0.3).fillna(0.0) / len(PAIRS) * 4
        nets_ts[lbl], to_t = net_of(w, 3)
        report(f"tsmom {lbl}", nets_ts[lbl], to_t)
    ens = (nets_ts["10d"] + nets_ts["20d"] + nets_ts["60d"]) / 3
    report("tsmom ens", ens, float("nan"))

    # equal-vol combo: scale each stream to same realized vol over full sample
    sc = net_carry / net_carry.std()
    st = ens / ens.std()
    for wc in [0.5, 0.6, 0.7]:
        combo = wc * sc + (1 - wc) * st
        report(f"combo carry{wc:.0%}", combo, float("nan"))

    print("\ncorrelations:")
    streams = pd.DataFrame({"carry": net_carry, "tsmom": ens}).dropna()
    print(streams.corr().round(3))


if __name__ == "__main__":
    main()
