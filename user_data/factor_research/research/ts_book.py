"""Per-pair absolute-signal trade simulation -- the freqtrade-shaped evaluation.

Entry when the factor's trailing 720-bar percentile rank is >= 0.90 (long) or <= 0.10
(short) on bar close; fill at the next bar's open; hold exactly `h` bars; exit at the open
of bar t+1+h; one open trade per pair. Cost 5 bps per side (10 stress) plus funding carried
pro rata. Returns per-trade records and the equal-weight open-book daily P&L for a
Newey-West t-stat. Rules fixed in reports/PREREGISTRATION_ts_screen.md.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ic as IC  # noqa: E402
import operators as ops  # noqa: E402

RANK_WIN = 720
Q_HI, Q_LO = 0.90, 0.10
COST_BPS = 5.0
COST_STRESS_BPS = 10.0


def entries(factor, rank_win=RANK_WIN):
    r = ops.ts_rank(factor, int(rank_win))
    return (r >= Q_HI), (r <= Q_LO)


def _trades_one_symbol(idx, h, T):
    """Greedy non-overlapping entries: signal bar t -> entry t+1, exit t+1+h. Loop runs per trade."""
    out = []
    k = 0
    n = len(idx)
    while k < n:
        t = int(idx[k])
        x = t + 1 + h
        if x >= T:
            break
        out.append(t)
        k = int(np.searchsorted(idx, x, side="left"))   # next signal at or after the exit bar
    return out


def simulate_from_signal(sig, side, open_, fund_per_bar, h, sl):
    """sig: (T,N) bool signal on bar close; only signals inside `sl` are eligible.
    Entry at open[t+1], exit at open[t+1+h]. Returns trade frame and per-bar equal-weight book P&L."""
    sgn = 1.0 if side == "long" else -1.0
    T, N = sig.shape
    cols = np.flatnonzero(sig[sl].any(axis=0))
    syms, ts = [], []
    for j in cols:
        idx = np.flatnonzero(sig[sl.start:sl.stop, j]) + sl.start
        tt = _trades_one_symbol(idx, h, T)
        syms.extend([j] * len(tt)); ts.extend(tt)
    if not ts:
        return pd.DataFrame(), None
    sym = np.asarray(syms); t = np.asarray(ts); e = t + 1; x = e + h
    pe, px = open_[e, sym], open_[x, sym]
    ok = np.isfinite(pe) & np.isfinite(px) & (pe > 0)
    sym, t, e, x, pe, px = sym[ok], t[ok], e[ok], x[ok], pe[ok], px[ok]
    cf = np.nancumsum(fund_per_bar, axis=0)
    fund = cf[x - 1, sym] - cf[e - 1, sym]        # bars e..x-1
    price = sgn * (px / pe - 1.0)
    df = pd.DataFrame({"sym": sym, "t": t, "e": e, "x": x, "price": price, "funding": -sgn * fund})
    df["net5"] = df.price + df.funding - 2 * COST_BPS * 1e-4
    df["net10"] = df.price + df.funding - 2 * COST_STRESS_BPS * 1e-4
    # per-bar equal-weight open-book return: bar b in [e, x) earns open[b+1]/open[b]-1 on that trade
    r1 = np.full_like(open_, np.nan)
    r1[:-1] = open_[1:] / open_[:-1] - 1.0
    # vectorised: every trade covers bars e..x-1 (h of them); flatten (trade, bar) pairs
    n = len(sym)
    bars = (e[:, None] + np.arange(h)[None, :]).ravel()
    symf = np.repeat(sym, h)
    diff = np.zeros(T + 1); np.add.at(diff, e, 1); np.add.at(diff, x, -1)
    n_open = np.cumsum(diff)[:T]
    rb = sgn * r1[bars, symf] - sgn * fund_per_bar[bars, symf]
    rb = np.where(np.isfinite(rb), rb, 0.0)
    rb = rb.reshape(n, h); rb[:, -1] -= 2 * COST_BPS * 1e-4; rb = rb.ravel()
    ret_bar = np.zeros(T)
    np.add.at(ret_bar, bars, rb / np.maximum(n_open[bars], 1))
    return df, ret_bar


def summarize(df, ret_bar, index, bars_per_day):
    if df.empty or len(df) < 5:
        return {"n_trades": int(len(df))}
    daily = pd.Series(ret_bar, index=index).resample("1D").sum()
    daily = daily[daily != 0]
    band = IC.andrews_band(daily.values) if len(daily) > 30 else 0
    _, _, t, _ = IC.newey_west(daily.values, band=band) if len(daily) > 30 else (np.nan, np.nan, np.nan, np.nan)
    q = pd.Series(df.net5.values, index=index[df.t.values]).groupby(pd.Grouper(freq="QE")).mean()
    yr = pd.Series(df.net5.values, index=index[df.t.values]).groupby(pd.Grouper(freq="YE")).mean()
    return {
        "n_trades": int(len(df)), "n_symbols": int(df.sym.nunique()),
        "mean_net5_bps": float(df.net5.mean() * 1e4), "median_net5_bps": float(df.net5.median() * 1e4),
        "mean_net10_bps": float(df.net10.mean() * 1e4), "win": float((df.net5 > 0).mean()),
        "funding_bps": float(df.funding.mean() * 1e4), "price_bps": float(df.price.mean() * 1e4),
        "nw_t_daily": float(t), "daily_mean_bps": float(daily.mean() * 1e4), "n_days": int(len(daily)),
        "quarters_pos": int((q > 0).sum()), "quarters": int(q.notna().sum()),
        "q_str": " ".join(f"{k.strftime('%yQ%q') if False else str(k.year)[2:]+'Q'+str(k.quarter)}:{v*1e4:+.0f}" for k, v in q.items()),
        "y_str": " ".join(f"{k.year}:{v*1e4:+.0f}" for k, v in yr.items()),
        "max_concurrent": int(np.max(np.bincount(np.repeat(df.e.values, (df.x - df.e).values) + np.concatenate([np.arange(k) for k in (df.x - df.e).values])))) if len(df) else 0,
    }
