#!/usr/bin/env python3
"""Event study: does the perp funding rate predict anything tradable?

Universe: the 15 Binance USDT perps that have both 5m/30m OHLCV and 1h
funding-rate history covering 2025-01-01 .. 2026-07-31.  Funding events are
8-hourly (00/08/16 UTC); everything below is aligned to those timestamps.

Questions, per calendar year (2025 vs 2026 must agree for a signal to count):

1. Time-series: conditioned on the trailing 3-day mean funding of a pair,
   what is the forward carry P&L of the *fade* position (short when funding
   is high — collect funding, pay price drift; long when negative)?
2. Cross-section: rank the 15 pairs by trailing funding each event, hold
   long-bottom-k / short-top-k, rebalance per event.  Report gross and
   net-of-fee Sharpe plus turnover.
3. Controls: cross-sectional 7d momentum and 1d reversal portfolios built
   identically, so carry is compared against the obvious alternatives.

No backtest framework involved; runs in seconds, one pair at a time.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

DATA = "user_data/data/binance/futures"
FEE_SIDE = 0.0007  # taker fee per side, same convention as the backtests
START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2026-07-31", tz="UTC")

PAIRS = [
    "BCH_USDT_USDT", "BNB_USDT_USDT", "BTC_USDT_USDT", "DOGE_USDT_USDT",
    "ETH_USDT_USDT", "HBAR_USDT_USDT", "LINK_USDT_USDT", "LTC_USDT_USDT",
    "SOL_USDT_USDT", "SUI_USDT_USDT", "TRX_USDT_USDT", "XLM_USDT_USDT",
    "XMR_USDT_USDT", "XRP_USDT_USDT", "ZEC_USDT_USDT",
]


def load_pair(pair: str) -> pd.DataFrame:
    """8h-aligned close + funding paid at each event, indexed by event time."""
    px = pd.read_feather(f"{DATA}/{pair}-30m-futures.feather")[["date", "close"]]
    px = px.set_index("date").sort_index()
    fr = pd.read_feather(f"{DATA}/{pair}-1h-funding_rate.feather")[["date", "open"]]
    fr = fr.rename(columns={"open": "funding"}).set_index("date").sort_index()
    # funding events: keep only 8h timestamps present in the funding file
    ev = fr[(fr.index >= START) & (fr.index <= END)]
    out = pd.DataFrame(index=ev.index)
    out["funding"] = ev["funding"]
    out["close"] = px["close"].reindex(out.index)
    out = out.dropna()
    return out


def annualized_sharpe(rets: pd.Series, periods_per_year: float) -> float:
    if len(rets) < 10 or rets.std() == 0:
        return float("nan")
    return float(rets.mean() / rets.std() * np.sqrt(periods_per_year))


def main() -> None:
    data = {p: load_pair(p) for p in PAIRS}
    for p, df in data.items():
        print(f"{p:22s} events={len(df)} {df.index.min().date()} .. {df.index.max().date()}")

    # wide frames on the union of event timestamps
    idx = sorted(set().union(*[set(df.index) for df in data.values()]))
    idx = pd.DatetimeIndex(idx)
    close = pd.DataFrame({p: data[p]["close"] for p in PAIRS}, index=idx)
    fund = pd.DataFrame({p: data[p]["funding"] for p in PAIRS}, index=idx)

    ret_fwd = close.shift(-1) / close - 1.0          # price return event t -> t+1
    fund_fwd = fund.shift(-1)                         # funding paid at t+1 (longs pay when >0)
    trail = fund.rolling(9, min_periods=6).mean()     # 3-day trailing mean funding
    ann_trail = trail * 3 * 365                       # annualized for readability

    years = {"2025": idx.year == 2025, "2026": idx.year == 2026}

    print("\n=== 1. time-series: forward 8h fade P&L by trailing-funding bucket ===")
    print("fade = short if trailing funding > 0 else long; pnl = -sign*price_ret + sign*funding")
    stack = pd.DataFrame({
        "ann_trail": ann_trail.stack(),
        "ret_fwd": ret_fwd.stack(),
        "fund_fwd": fund_fwd.stack(),
    }).dropna()
    stack["year"] = stack.index.get_level_values(0).year.astype(str)
    sign = -np.sign(stack["ann_trail"])              # fade direction: short positive funding
    stack["fade_pnl"] = sign * stack["ret_fwd"] - sign * stack["fund_fwd"] * -1.0
    # careful: short position RECEIVES positive funding: pnl_short = -ret + funding
    stack["fade_pnl"] = np.where(sign < 0, -stack["ret_fwd"] + stack["fund_fwd"],
                                 stack["ret_fwd"] - stack["fund_fwd"])
    edges = [-np.inf, -0.10, -0.02, 0.02, 0.10, 0.30, np.inf]
    labels = ["<-10%", "-10..-2%", "-2..2%", "2..10%", "10..30%", ">30%"]
    stack["bucket"] = pd.cut(stack["ann_trail"], edges, labels=labels)
    for yr in ["2025", "2026"]:
        sub = stack[stack["year"] == yr]
        g = sub.groupby("bucket", observed=True)["fade_pnl"]
        t = g.mean() / g.std() * np.sqrt(g.count())
        rep = pd.DataFrame({"n": g.count(), "mean_bp": g.mean() * 1e4,
                            "t": t, "hit%": g.apply(lambda s: (s > 0).mean() * 100)})
        print(f"\n{yr}:\n{rep.round(2)}")

    print("\n=== 2. cross-sectional portfolios (rebalance each 8h event) ===")

    def xs_portfolio(score: pd.DataFrame, k: int, name: str) -> None:
        # long lowest-k score, short highest-k; carry+price pnl; fee on turnover
        ranks = score.rank(axis=1)
        nn = score.notna().sum(axis=1)
        w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
        w[ranks.le(k, axis=0).values & score.notna().values] = 1.0 / k
        top = ranks.ge(nn - k + 1, axis=0)
        w[top.values & score.notna().values] = -1.0 / k
        pnl_price = (w * ret_fwd).sum(axis=1)
        pnl_fund = (w * -fund_fwd).sum(axis=1)       # long pays funding, short receives
        turnover = (w - w.shift(1)).abs().sum(axis=1)
        pnl_net = pnl_price + pnl_fund - turnover * FEE_SIDE
        for yr, mask in years.items():
            for nm, s in [("gross", (pnl_price + pnl_fund)[mask]), ("net", pnl_net[mask])]:
                sh = annualized_sharpe(s.dropna(), 3 * 365)
                print(f"  {name:16s} {yr} {nm:5s}: ann_ret={s.mean()*3*365*100:7.2f}%  "
                      f"sharpe={sh:6.2f}  turnover/ev={turnover[mask].mean():.3f}")

    xs_portfolio(trail, 3, "carry k=3")
    xs_portfolio(trail, 5, "carry k=5")
    mom7 = close / close.shift(21) - 1.0              # 7d momentum (21 events)
    xs_portfolio(-mom7, 3, "mom7d k=3")               # long winners: low score = winner
    rev1 = close / close.shift(3) - 1.0               # 1d reversal
    xs_portfolio(rev1, 3, "rev1d k=3")                # long losers

    print("\n=== 3. persistence: autocorrelation of trailing funding (signal stability) ===")
    for lag_ev, lbl in [(3, "1d"), (9, "3d"), (21, "7d")]:
        acs = [trail[p].autocorr(lag_ev) for p in PAIRS]
        print(f"  lag {lbl}: median autocorr = {np.nanmedian(acs):.3f}")


if __name__ == "__main__":
    main()
