#!/usr/bin/env python3
"""Can a trailing daily/weekly trend gauge tell 2025-chop from 2026-trend?

Caveat first: 2025->2026 is ONE regime transition, so any indicator that
"calls" it has an effective sample of one.  This study only answers:

(a) do standard trendiness metrics, computed EX-ANTE on a trailing 90d
    window, actually separate the two years?
(b) at monthly granularity, how well does each metric's value at month
    start rank-correlate with that month's momentum-minus-reversion
    relative P&L (the thing a switcher would need to predict)?

Metrics (trailing 90d, on an equal-weight log index of the 15 full-span
majors, and on BTC alone):
  ER    Kaufman efficiency ratio |net|/sum|daily|
  R2    R^2 of log price on time
  AC1   lag-1 autocorrelation of daily returns
  VOL   annualized daily vol
  DISP  cross-sectional std of the majors' trailing 30d returns

Strategy streams re-simulated: XS momentum ensemble (2025 pool then 2026
all-perps universes) and majors beta-hedged spread reversion.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from analyze_funding_carry_event_study import PAIRS as MAJORS, load_pair
from analyze_smallcap_carry_xs import MIN_QVOL

FEE_SIDE = 0.0007


# ---------------------------------------------------------------- strategies

def mom_ensemble_daily(pdir, a, b):
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
    elig = qvol.rolling(24, min_periods=12).sum().rolling(168, min_periods=84).mean() > MIN_QVOL
    ens = sum((-(close / close.shift(n) - 1.0)).where(elig).rank(axis=1, pct=True)
              for n in (72, 336, 720)) / 3
    ranks = ens.rank(axis=1)
    nn = ens.notna().sum(axis=1)
    k = 5
    w = pd.DataFrame(0.0, index=ens.index, columns=ens.columns)
    w[ranks.le(k, axis=0).values & ens.notna().values] = 1.0 / k
    w[ranks.ge(nn - k + 1, axis=0).values & ens.notna().values] = -1.0 / k
    w[nn < 12] = 0.0
    daily_mask = pd.Series(ens.index.hour == 0, index=ens.index)
    w = w.where(daily_mask, np.nan).ffill().fillna(0.0).where(close.notna(), 0.0)
    fwd = close.shift(-1) / close - 1.0
    net = (w * fwd).sum(axis=1) - (w - w.shift(1)).abs().sum(axis=1) * FEE_SIDE
    return net.resample("1D").sum()


def majors_reversion_daily():
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
    w = pos.div(pos.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    net = (w * ret.shift(-1)).sum(axis=1) + (w * -fund.shift(-1)).sum(axis=1) \
        - (w - w.shift(1)).abs().sum(axis=1) * FEE_SIDE
    return net.resample("1D").sum()


# ------------------------------------------------------------------ metrics

def metrics_at(series_daily: pd.Series, when: pd.Timestamp, win: int = 90) -> dict:
    px = series_daily[series_daily.index < when].tail(win)
    if len(px) < win // 2:
        return {}
    lr = np.log(px)
    dr = lr.diff().dropna()
    er = abs(lr.iloc[-1] - lr.iloc[0]) / dr.abs().sum() if dr.abs().sum() > 0 else np.nan
    t = np.arange(len(lr))
    r2 = np.corrcoef(t, lr)[0, 1] ** 2
    ac1 = dr.autocorr(1)
    vol = dr.std() * np.sqrt(365)
    return {"ER": er, "R2": r2, "AC1": ac1, "VOL": vol}


def main() -> None:
    print("building strategy streams...")
    mom25 = mom_ensemble_daily("user_data/data/binance-2025/futures", "2025-02-01", "2026-01-31")
    mom26 = mom_ensemble_daily("user_data/data/binance/futures", "2026-01-15", "2026-08-13")
    mom = pd.concat([mom25[mom25.index < pd.Timestamp("2026-02-01", tz="UTC")], mom26]).sort_index()
    rev = majors_reversion_daily()

    data = {p: load_pair(p) for p in MAJORS}
    idx = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in data.values()])))
    close = pd.DataFrame({p: data[p]["close"] for p in MAJORS}, index=idx)
    index_daily = np.exp(np.log(close).mean(axis=1)).resample("1D").last().dropna()
    btc_daily = close["BTC_USDT_USDT"].resample("1D").last().dropna()
    ret30 = close.resample("1D").last().pct_change(30)

    months = pd.date_range("2025-05-01", "2026-08-01", freq="MS", tz="UTC")
    rows = []
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        met = metrics_at(index_daily, m0)
        if not met:
            continue
        met_btc = metrics_at(btc_daily, m0)
        met["ER_btc"] = met_btc.get("ER", np.nan)
        d = ret30[ret30.index < m0]
        met["DISP"] = d.iloc[-1].std() if len(d) else np.nan
        mm = mom[(mom.index >= m0) & (mom.index < m1)].sum()
        rr = rev[(rev.index >= m0) & (rev.index < m1)].sum()
        rows.append({"month": m0.strftime("%Y-%m"), **{k: round(v, 3) for k, v in met.items()},
                     "momP%": round(mm * 100, 1), "revP%": round(rr * 100, 1),
                     "spread%": round((mm - rr) * 100, 1)})
    df = pd.DataFrame(rows).set_index("month")
    print(df.to_string())
    print("\nSpearman corr of ex-ante metric vs same-month (mom - rev) spread:")
    for c in ["ER", "R2", "AC1", "VOL", "ER_btc", "DISP"]:
        a = df[c].rank(); b = df['spread%'].rank()
        print(f"  {c:6s}: {a.corr(b):+.2f}")
    print(f"\nn months = {len(df)} — one regime transition in sample; treat as descriptive only.")


if __name__ == "__main__":
    main()
