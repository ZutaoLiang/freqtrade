#!/usr/bin/env python3
"""Cross-sectional funding-carry / momentum study on the high-volume universe.

Universes (both with funding + 1h OHLCV on disk):

* 2025: the 72 rotating-pool pairs in ``user_data/data/binance-2025/futures``
  (each file covers its pool window plus warmup/carry).
* 2026: all pairs in ``user_data/data/binance/futures`` with 1h OHLCV +
  funding, 2026-01-01 .. 2026-07-31.

Point-in-time liquidity filter: a pair is in the universe at event t only if
its trailing 7d mean 24h quote volume exceeds MIN_QVOL.  Signals are ranked
inside that universe only.

Signals: trailing 7d mean funding (carry, short expensive names) and 7d price
momentum.  Portfolios: rank k per side, daily rebalance, net of 0.07%/side.
2026 is additionally split into H1 (Jan-Apr) and H2 (May-Jul) for a
sample-split check.  Runs in under a minute, well inside the memory budget.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

FEE_SIDE = 0.0007
MIN_QVOL = 30e6  # 24h quote volume floor, USD


def load_universe(price_dir: str, start: str, end: str):
    files = sorted(glob.glob(f"{price_dir}/*-1h-futures.feather"))
    closes, qvols, funds = {}, {}, {}
    for f in files:
        pair = os.path.basename(f).split("-1h-")[0]
        ff = f"{price_dir}/{pair}-1h-funding_rate.feather"
        if not os.path.exists(ff):
            continue
        try:
            px = pd.read_feather(f, columns=["date", "close", "volume"]).set_index("date").sort_index()
            fr = pd.read_feather(ff, columns=["date", "open"]).set_index("date").sort_index()
        except Exception:
            continue
        if px.empty or fr.empty:
            continue
        closes[pair] = px["close"]
        qvols[pair] = px["close"] * px["volume"]
        funds[pair] = fr["open"]
    idx = pd.date_range(start, end, freq="8h", tz="UTC")
    close = pd.DataFrame({p: s for p, s in closes.items()}).sort_index()
    qvol24 = pd.DataFrame({p: s for p, s in qvols.items()}).sort_index().rolling(24, min_periods=12).sum()
    fund = pd.DataFrame({p: s for p, s in funds.items()}).sort_index()
    # align to 8h event grid
    close8 = close.reindex(idx)
    qvol8 = qvol24.reindex(idx).rolling(21, min_periods=9).mean()  # trailing 7d mean of 24h qvol
    fund8 = fund.reindex(idx)
    return close8, qvol8, fund8


def annualized_sharpe(rets: pd.Series, ppy: float = 3 * 365) -> float:
    r = rets.dropna()
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ppy))


def run_portfolio(close, fund, elig, score, k, reb_ev, label, splits):
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
    # a pair leaving the universe (data gap) forces liquidation at last close
    w = w.where(close.notna(), 0.0)
    pnl = (w * ret_fwd).sum(axis=1) + (w * -fund_fwd).sum(axis=1)
    to = (w - w.shift(1)).abs().sum(axis=1)
    net = pnl - to * FEE_SIDE
    parts = []
    for lbl, mask in splits.items():
        seg = net[mask]
        parts.append(f"{lbl}: {seg.mean()*3*365*100:7.2f}% shp={annualized_sharpe(seg):6.2f}")
    print(f"  {label:24s} " + " | ".join(parts) + f" | to={to.mean():.3f}")
    return net


def study(name, price_dir, start, end, splits_def):
    print(f"\n### {name} ({price_dir})")
    close, qvol, fund = load_universe(price_dir, start, end)
    print(f"pairs loaded: {close.shape[1]}")
    elig = qvol > MIN_QVOL
    print(f"universe size per event: median={elig.sum(axis=1).median():.0f}, "
          f"min={elig.sum(axis=1).min()}, max={elig.sum(axis=1).max()}")
    idx = close.index
    splits = {lbl: (idx >= pd.Timestamp(a, tz="UTC")) & (idx < pd.Timestamp(b, tz="UTC"))
              for lbl, (a, b) in splits_def.items()}
    trail = fund.rolling(21, min_periods=9).mean()
    mom7 = close / close.shift(21) - 1.0
    run_portfolio(close, fund, elig, trail, 5, 3, "carry7d k=5 1d", splits)
    run_portfolio(close, fund, elig, trail, 3, 3, "carry7d k=3 1d", splits)
    run_portfolio(close, fund, elig, fund.rolling(9, min_periods=5).mean(), 5, 3, "carry3d k=5 1d", splits)
    run_portfolio(close, fund, elig, -mom7, 5, 3, "mom7d k=5 1d", splits)
    blend = trail.rank(axis=1, pct=True) * 0.5 + (-mom7).rank(axis=1, pct=True) * 0.5
    run_portfolio(close, fund, elig, blend, 5, 3, "blend50/50 k=5 1d", splits)


def main() -> None:
    study("2025 rotating pool", "user_data/data/binance-2025/futures",
          "2025-02-01", "2026-01-31",
          {"25H1(Feb-Jul)": ("2025-02-01", "2025-08-01"),
           "25H2(Aug-Jan)": ("2025-08-01", "2026-02-01")})
    study("2026 all-perps", "user_data/data/binance/futures",
          "2026-01-15", "2026-07-31",
          {"26H1(Jan-Apr)": ("2026-01-15", "2026-05-01"),
           "26H2(May-Jul)": ("2026-05-01", "2026-08-01")})


if __name__ == "__main__":
    main()
