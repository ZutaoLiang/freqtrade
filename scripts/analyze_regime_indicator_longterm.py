#!/usr/bin/env python3
"""Long-sample (2018-2026) test: does trailing trendiness predict the
momentum-vs-reversion spread one month ahead?

Extends analyze_regime_indicator.py from 16 months / one regime flip to
~90 months spanning the 2019 chop, 2020-21 bull, 2022 bear, 2023 chop,
2024 bull and the 2025/2026 flip, using daily spot data for 20 majors
(scripts/download_majors_spot_daily.py).

Proxy strategy streams (daily, net 0.07%/side):
  MOM  cross-sectional 30d momentum, long top-5 / short bottom-5, weekly
  REV  cross-sectional 7d reversal, long losers / short winners, weekly
  TSM  time-series sign of 90d return per coin, equal weight, weekly

PRE-REGISTERED primary test: Spearman of index ER(90d, ex-ante at month
start) vs next-month (MOM - REV) spread.  Everything else descriptive.
Also: metric self-persistence, and a walk-forward switch (mom if trailing
ER above its own expanding median, else rev) vs static 50/50.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

FEE_SIDE = 0.0007
DATA = "user_data/data/binance-spot-daily"


def load() -> pd.DataFrame:
    closes = {}
    for f in sorted(glob.glob(f"{DATA}/*-1d.feather")):
        sym = os.path.basename(f).split("-")[0]
        df = pd.read_feather(f).set_index("date").sort_index()
        closes[sym] = df["close"][~df.index.duplicated(keep="last")]
    return pd.DataFrame(closes).sort_index()


def xs_weights(score: pd.DataFrame, k: int) -> pd.DataFrame:
    ranks = score.rank(axis=1)
    nn = score.notna().sum(axis=1)
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    w[ranks.le(k, axis=0).values & score.notna().values] = 1.0 / k
    w[ranks.ge(nn - k + 1, axis=0).values & score.notna().values] = -1.0 / k
    w[nn < 2 * k + 2] = 0.0
    return w


def weekly(w: pd.DataFrame) -> pd.DataFrame:
    keep = w.index.dayofweek == 0
    return w.where(pd.Series(keep, index=w.index), np.nan).ffill().fillna(0.0)


def net_pnl(w: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    pnl = (w * fwd).sum(axis=1)
    to = (w - w.shift(1)).abs().sum(axis=1)
    return pnl - to * FEE_SIDE


def main() -> None:
    close = load()
    fwd = close.shift(-1) / close - 1.0

    mom = net_pnl(weekly(xs_weights(-(close / close.shift(30) - 1.0), 5)), fwd)
    rev = net_pnl(weekly(xs_weights(close / close.shift(7) - 1.0, 5)), fwd)
    sig = np.sign(close / close.shift(90) - 1.0)
    n_act = sig.notna().sum(axis=1).replace(0, np.nan)
    tsm = net_pnl(weekly(sig.div(n_act, axis=0).fillna(0.0)), fwd)

    lg_index = np.log(close).mean(axis=1)  # equal-weight log index (available names)
    index_px = np.exp(lg_index)

    def metrics_at(m0, win=90):
        px = index_px[index_px.index < m0].tail(win)
        lr = np.log(px)
        dr = lr.diff().dropna()
        if len(dr) < win // 2:
            return {}
        er = abs(lr.iloc[-1] - lr.iloc[0]) / dr.abs().sum() if dr.abs().sum() > 0 else np.nan
        t = np.arange(len(lr))
        r2 = np.corrcoef(t, lr)[0, 1] ** 2
        d30 = close[close.index < m0].tail(31)
        disp = (d30.iloc[-1] / d30.iloc[0] - 1.0).std() if len(d30) > 30 else np.nan
        return {"ER": er, "R2": r2, "AC1": dr.autocorr(1),
                "VOL": dr.std() * np.sqrt(365), "DISP": disp}

    months = pd.date_range("2018-10-01", "2026-08-01", freq="MS", tz="UTC")
    rows = []
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        met = metrics_at(m0)
        if not met:
            continue
        sel = (mom.index >= m0) & (mom.index < m1)
        rows.append({"m": m0, **met,
                     "mom": mom[sel].sum(), "rev": rev[sel].sum(), "tsm": tsm[sel].sum()})
    df = pd.DataFrame(rows).set_index("m").dropna()
    df["spread"] = df["mom"] - df["rev"]
    n = len(df)

    print(f"months = {n} ({df.index[0].date()} .. {df.index[-1].date()})")
    print("\nannualized proxy Sharpe by calendar year:")
    for y in sorted(set(df.index.year)):
        seg = df[df.index.year == y]
        f = lambda s: f"{s.mean() / s.std() * np.sqrt(12):+.1f}" if s.std() > 0 else "nan"
        print(f"  {y}: mom {f(seg['mom'])}  rev {f(seg['rev'])}  tsm {f(seg['tsm'])}")

    def spear(a, b):
        return a.rank().corr(b.rank())

    print("\nPRIMARY: Spearman ER(ex-ante) vs next-month (MOM-REV) spread:")
    r = spear(df["ER"], df["spread"])
    # permutation p-value
    rng = np.random.default_rng(7)
    perm = np.array([spear(df["ER"], df["spread"].sample(frac=1, random_state=i).reset_index(drop=True).set_axis(df.index))
                     for i in range(2000)])
    p = (np.abs(perm) >= abs(r)).mean()
    print(f"  rho = {r:+.3f}  perm p = {p:.3f}  (n={n})")

    print("\nsecondary metrics vs spread / vs tsm:")
    for c in ["ER", "R2", "AC1", "VOL", "DISP"]:
        print(f"  {c:5s}: vs spread {spear(df[c], df['spread']):+.2f}   vs tsm {spear(df[c], df['tsm']):+.2f}")

    print("\nmetric self-persistence (Spearman month t vs t+1):")
    for c in ["ER", "R2", "AC1", "VOL", "DISP"]:
        print(f"  {c:5s}: {spear(df[c].iloc[:-1].reset_index(drop=True), df[c].iloc[1:].reset_index(drop=True)):+.2f}")

    print("\nwalk-forward switch (mom if ER > expanding median else rev), start after 12 months:")
    sw = []
    for i in range(12, n):
        hist = df["ER"].iloc[:i]
        pick = "mom" if df["ER"].iloc[i] > hist.median() else "rev"
        sw.append(df[pick].iloc[i])
    sw = pd.Series(sw, index=df.index[12:])
    half = (df["mom"] + df["rev"]).iloc[12:] / 2
    for nm, s in [("switch", sw), ("static50/50", half),
                  ("mom only", df["mom"].iloc[12:]), ("rev only", df["rev"].iloc[12:])]:
        print(f"  {nm:12s}: ann={s.mean()*12*100:+7.1f}%  shp={s.mean()/s.std()*np.sqrt(12):+.2f}")


if __name__ == "__main__":
    main()
