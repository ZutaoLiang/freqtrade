#!/usr/bin/env python3
"""Third battery: funding cost of shorting young listings, majors pair
mean-reversion, Amihud illiquidity factor, MAX lottery factor.

Same acceptance discipline as before.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from analyze_smallcap_carry_xs import MIN_QVOL
from analyze_funding_carry_event_study import PAIRS as MAJORS, load_pair, annualized_sharpe

FEE_SIDE = 0.0007


def load_1h(pdir, a, b):
    closes, qvols = {}, {}
    for f in sorted(glob.glob(f"{pdir}/*-1h-futures.feather")):
        p = os.path.basename(f).split("-1h-")[0]
        try:
            px = pd.read_feather(f, columns=["date", "close", "volume"]).set_index("date").sort_index()
        except Exception:
            continue
        if px.empty:
            continue
        px = px[~px.index.duplicated(keep="last")]
        closes[p] = px["close"]
        qvols[p] = px["close"] * px["volume"]
    close = pd.DataFrame(closes).sort_index()
    close = close[(close.index >= pd.Timestamp(a, tz="UTC")) & (close.index <= pd.Timestamp(b, tz="UTC"))]
    qvol = pd.DataFrame(qvols).sort_index().reindex(close.index)
    return close, qvol


def young_listing_funding(pdir, a, b):
    """Mean daily funding a SHORT would receive (+) or pay (-) by listing age."""
    rows = []
    for f in sorted(glob.glob(f"{pdir}/*-1h-funding_rate.feather")):
        p = os.path.basename(f).split("-1h-")[0]
        pf = f"{pdir}/{p}-1h-futures.feather"
        if not os.path.exists(pf):
            continue
        try:
            fr = pd.read_feather(f, columns=["date", "open"]).set_index("date").sort_index()
            px = pd.read_feather(pf, columns=["date"])
        except Exception:
            continue
        if fr.empty or px.empty:
            continue
        first = px["date"].min()
        fr = fr[(fr.index >= pd.Timestamp(a, tz="UTC")) & (fr.index <= pd.Timestamp(b, tz="UTC"))]
        if fr.empty:
            continue
        age = (fr.index - first).days
        rows.append(pd.DataFrame({"age": age, "funding": fr["open"].to_numpy()}))
    allf = pd.concat(rows)
    print("  funding received by SHORT, %/day, by listing age:")
    for lo, hi in [(4, 7), (8, 14), (15, 30), (31, 10_000)]:
        seg = allf[(allf.age >= lo) & (allf.age <= hi)]["funding"]
        print(f"    d{lo:3d}-{hi:5d}: n={len(seg):6d} {seg.mean()*3*100:+.3f}%/day")


def majors_pair_reversion():
    """Cross-major spread mean reversion vs BTC, z-score of 30d beta-hedged spread."""
    data = {p: load_pair(p) for p in MAJORS}
    idx = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in data.values()])))
    close = pd.DataFrame({p: data[p]["close"] for p in MAJORS}, index=idx)
    fund = pd.DataFrame({p: data[p]["funding"] for p in MAJORS}, index=idx)
    lg = np.log(close)
    btc = lg["BTC_USDT_USDT"]
    ret = lg.diff()
    bret = btc.diff()
    beta = ret.rolling(90).cov(bret).div(bret.rolling(90).var(), axis=0)
    spread = lg.sub(beta.mul(btc, axis=0))
    z = (spread - spread.rolling(90).mean()) / spread.rolling(90).std()
    fwd = ret.shift(-1)
    fund_fwd = fund.shift(-1)
    years = {"25H1": (idx.year == 2025) & (idx.month <= 6), "25H2": (idx.year == 2025) & (idx.month > 6),
             "26H1": (idx.year == 2026) & (idx.month <= 4), "26H2": (idx.year == 2026) & (idx.month > 4)}
    # portfolio: long z<-1.5, short z>1.5, exit at |z|<0.5 (stateful), 8h events
    pos = pd.DataFrame(0.0, index=idx, columns=close.columns)
    state = {c: 0 for c in close.columns}
    zv = z.to_numpy()
    for i in range(len(idx)):
        for j, c in enumerate(close.columns):
            v = zv[i, j]
            if np.isnan(v):
                state[c] = 0
            elif state[c] == 0:
                state[c] = -1 if v > 1.5 else (1 if v < -1.5 else 0)
            elif abs(v) < 0.5:
                state[c] = 0
            pos.iloc[i, j] = state[c]
    n_active = pos.abs().sum(axis=1).replace(0, np.nan)
    w = pos.div(n_active, axis=0).fillna(0.0)
    pnl = (w * fwd).sum(axis=1) + (w * -fund_fwd).sum(axis=1)
    to = (w - w.shift(1)).abs().sum(axis=1)
    net = pnl - to * FEE_SIDE
    print("  majors beta-hedged spread reversion (enter |z|>1.5 exit <0.5):")
    for lbl, m in years.items():
        s = net[m].dropna()
        print(f"    {lbl}: ann={s.mean()*3*365*100:+7.1f}% shp={annualized_sharpe(s, 3*365):5.2f}", end="")
    print(f"  to/ev={to.mean():.3f}")


def xs_factor(close, qvol, elig, score_builder, name, spl):
    """Generic k=5 XS portfolio at daily rebalance on 1h data, price-only pnl."""
    score = score_builder(close, qvol)
    s = score.where(elig)
    ranks = s.rank(axis=1)
    nn = s.notna().sum(axis=1)
    k = 5
    w = pd.DataFrame(0.0, index=s.index, columns=s.columns)
    w[ranks.le(k, axis=0).values & s.notna().values] = 1.0 / k
    w[ranks.ge(nn - k + 1, axis=0).values & s.notna().values] = -1.0 / k
    w[nn < 12] = 0.0
    daily = s.index.hour == 0
    w = w.where(pd.Series(daily, index=s.index), np.nan).ffill().fillna(0.0)
    w = w.where(close.notna(), 0.0)
    fwd = close.shift(-1) / close - 1.0
    pnl = (w * fwd).sum(axis=1)
    to = (w - w.shift(1)).abs().sum(axis=1)
    net = pnl - to * FEE_SIDE
    cells = []
    for lbl, m in spl.items():
        seg = net[m].dropna()
        cells.append(f"{lbl}: {seg.mean()*24*365*100:+7.1f}% shp={annualized_sharpe(seg, 24*365):5.2f}")
    print(f"  {name:14s} " + " | ".join(cells) + f" to/h={to.mean():.3f}")


def main() -> None:
    print("### funding cost of shorting young listings, 2026 all-perps")
    young_listing_funding("user_data/data/binance/futures", "2026-01-01", "2026-08-13")

    print("\n### majors pair reversion (15 majors, 2025-01..2026-07)")
    majors_pair_reversion()

    amihud = lambda c, q: -( (c.pct_change().abs() / q.replace(0, np.nan)).rolling(720, min_periods=360).mean() )
    # long illiquid = high |ret|/vol -> score ascending small = short liquid... sign: long high amihud
    maxfac = lambda c, q: (c.pct_change(24).rolling(720, min_periods=360).max())
    # short high MAX (lottery): score ascending -> long low MAX

    for tag, pdir, a, b, cuts in [
        ("2025", "user_data/data/binance-2025/futures", "2025-02-01", "2026-01-31",
         [("25H1", "2025-02-01", "2025-08-01"), ("25H2", "2025-08-01", "2026-02-01")]),
        ("2026", "user_data/data/binance/futures", "2026-01-15", "2026-08-13",
         [("26H1", "2026-01-15", "2026-05-01"), ("26H2", "2026-05-01", "2026-08-14")]),
    ]:
        close, qvol = load_1h(pdir, a, b)
        qtrail = qvol.rolling(24, min_periods=12).sum().rolling(168, min_periods=84).mean()
        elig = qtrail > MIN_QVOL
        spl = {lbl: (close.index >= pd.Timestamp(x, tz="UTC")) & (close.index < pd.Timestamp(y, tz="UTC"))
               for lbl, x, y in cuts}
        print(f"\n### {tag} XS factors (long low score / short high score, k=5, 1d rebalance)")
        xs_factor(close, qvol, elig, amihud, "amihud(longIll)", spl)
        xs_factor(close, qvol, elig, maxfac, "MAX(shortLotto)", spl)


if __name__ == "__main__":
    main()
