"""Can a causal BTC/ETH daily regime classifier make the momentum book tradeable?

Round 2 of the ts x screen study showed momentum's sign follows the market: long momentum earned
+186 bps/trade in 2023-24 and -173 in 2025. The obvious response is "detect the regime and flip
the side". This tests whether that works with information available at the time.

Three questions, in order:
  1. How well does each causal classifier label the regime at all (accuracy on forward BTC returns,
     lag behind the actual turn, number of flips)?
  2. Does switching the momentum book by regime beat always-long and always-short?
  3. Does it beat simply holding BTC long/short on the same regime signal? If not, the factor adds
     nothing and what is being tested is a BTC timing model, not a strategy.

Everything is evaluated on the 1d_long panel (2022-11..2026-08). Signals use bar-close data,
positions run from the next bar's open. Costs 5 bps per side on turnover.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import operators as ops  # noqa: E402
import screens as S  # noqa: E402
from panel import CACHE, Panel  # noqa: E402

TF = "1d_long"
COST = 5e-4          # per side, applied to turnover
SPLITS = {"train 22-11..24-12": ("2022-11-01", "2025-01-01"),
          "valid 2025": ("2025-01-01", "2026-01-01"),
          "holdout 2026": ("2026-01-01", "2026-08-17")}


def regimes(close, volume, idx, symbols):
    """Causal bull/bear labels from BTC and ETH daily bars. True = bull."""
    b = pd.Series(close[:, symbols.index("BTCUSDT")], index=idx).ffill()
    e = pd.Series(close[:, symbols.index("ETHUSDT")], index=idx).ffill()
    bv = pd.Series(volume[:, symbols.index("BTCUSDT")], index=idx).ffill()
    out = {}
    out["btc>ma50"] = b > b.rolling(50).mean()
    out["btc>ma200"] = b > b.rolling(200).mean()
    out["btc ma20>ma50"] = b.rolling(20).mean() > b.rolling(50).mean()
    out["btc ret60>0"] = b.pct_change(60) > 0
    out["btc dd90>-10%"] = b / b.rolling(90).max() - 1 > -0.10
    out["btc&eth>ma50"] = (b > b.rolling(50).mean()) & (e > e.rolling(50).mean())
    # price-volume: up-days' volume share over 30d (a crude OBV slope)
    upvol = (bv * (b.diff() > 0)).rolling(30).sum() / bv.rolling(30).sum()
    out["btc upvol30>0.5"] = upvol > 0.5
    out["btc>ma50 & upvol"] = out["btc>ma50"] & out["btc upvol30>0.5"]
    # ETH/BTC ratio trend: classic "altseason" proxy
    r = e / b
    out["eth/btc>ma50"] = r > r.rolling(50).mean()
    return {k: v.fillna(False).to_numpy() for k, v in out.items()}


def book_returns(sig, open_, mask, hold):
    """Equal-weight book: a signal at t puts 1/hold of the book into that pair for `hold` bars.
    Returns per-bar portfolio return and per-bar turnover."""
    T, N = sig.shape
    s = (sig & mask).astype("float64")
    # weight at bar t = fraction of the last `hold` signals, normalised across names
    cs = np.cumsum(s, axis=0)
    acc = cs.copy()
    acc[hold:] -= cs[:-hold]
    w = acc / hold
    gross = w.sum(axis=1, keepdims=True)
    w = np.divide(w, gross, out=np.zeros_like(w), where=gross > 0)
    r1 = np.full_like(open_, np.nan)
    r1[:-1] = open_[1:] / open_[:-1] - 1.0
    r1 = np.where(np.isfinite(r1), r1, 0.0)
    # weights known at close of t are held over bar t+1 -> shift
    wl = np.zeros_like(w)
    wl[1:] = w[:-1]
    ret = (wl * r1).sum(axis=1)
    turn = np.abs(np.diff(wl, axis=0, prepend=0)).sum(axis=1)
    return ret, turn


def stats(ret, turn, idx, lo, hi):
    m = (idx >= lo) & (idx < hi)
    r = ret[m] - COST * turn[m]
    if r.size < 20:
        return {}
    ann = 365
    return {"total%": (np.prod(1 + r) - 1) * 100, "ann%": (r.mean() * ann) * 100,
            "sharpe": r.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else np.nan,
            "maxdd%": (1 - (1 + r).cumprod() / np.maximum.accumulate((1 + r).cumprod())).max() * 100}


def main():
    p = Panel(TF)
    idx, syms = p.index, p.symbols
    close = np.asarray(p["close"], dtype="float64")
    open_ = np.asarray(p["open"], dtype="float64")
    volume = np.asarray(p["volume"], dtype="float64")
    scr = S.load(TF)
    mask = np.asarray(scr["top100"])
    pd.set_option("display.width", 250)

    reg = regimes(close, volume, idx, syms)
    btc = pd.Series(close[:, syms.index("BTCUSDT")], index=idx).ffill()
    fwd30 = (btc.shift(-30) / btc - 1).to_numpy()
    fwd7 = (btc.shift(-7) / btc - 1).to_numpy()

    print("=== 1. classifier quality (causal labels vs forward BTC return)")
    rows = []
    for k, v in reg.items():
        ok = np.isfinite(fwd30)
        acc30 = ((v[ok] & (fwd30[ok] > 0)) | (~v[ok] & (fwd30[ok] <= 0))).mean()
        acc7 = ((v[ok] & (fwd7[ok] > 0)) | (~v[ok] & (fwd7[ok] <= 0))).mean()
        flips = int(np.sum(v[1:] != v[:-1]))
        rows.append({"regime": k, "bull_frac": v.mean(), "acc_fwd7": acc7, "acc_fwd30": acc30,
                     "flips": flips, "days_per_state": len(v) / max(flips, 1),
                     "mean_fwd30_bull%": fwd30[ok & v].mean() * 100, "mean_fwd30_bear%": fwd30[ok & ~v].mean() * 100})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # the factor book: the round-2 entry rule, momentum, top100
    f = ops.ts_rank(close, 90)
    hi_sig, lo_sig = f >= 0.90, f <= 0.10
    hold = 7
    ret_L, tn_L = book_returns(hi_sig, open_, mask, hold)       # long the high decile (momentum long)
    ret_S, tn_S = book_returns(hi_sig, open_, mask, hold)       # same names, short
    ret_S = -ret_S
    ret_eq, tn_eq = book_returns(mask, open_, mask, 1)          # equal-weight all top100 = market beta
    btc_r = btc.pct_change().shift(-1).fillna(0).to_numpy()     # BTC return over the next bar

    print(f"\n=== 2. momentum book (ts_rank(close,90d) top decile, top100, hold {hold}d) under each regime switch")
    print("   switched = long-momentum when bull, short-momentum when bear")
    rows = []
    for name, arms in [("always long-mom", None), ("always short-mom", "short"), ("perfect foresight", "oracle")] + [(k, k) for k in reg]:
        if arms is None:
            r, t = ret_L, tn_L
        elif arms == "short":
            r, t = ret_S, tn_S
        elif arms == "oracle":
            bull = np.zeros(len(idx), dtype=bool)
            bull[:-30] = fwd30[:-30] > 0
            r = np.where(bull, ret_L, ret_S); t = np.where(bull, tn_L, tn_S) + np.abs(np.diff(bull.astype(float), prepend=0)) * 2
        else:
            bull = reg[arms]
            r = np.where(bull, ret_L, ret_S); t = np.where(bull, tn_L, tn_S) + np.abs(np.diff(bull.astype(float), prepend=0)) * 2
        row = {"arm": name}
        for sname, (lo, hi) in SPLITS.items():
            s = stats(r, t, idx, pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC"))
            row[f"{sname} ann%"] = s.get("ann%", np.nan)
            row[f"{sname} sh"] = s.get("sharpe", np.nan)
        rows.append(row)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n=== 3. the control: same regime signal applied to BTC alone (no factor at all)")
    rows = []
    for name in ["perfect foresight"] + list(reg):
        if name == "perfect foresight":
            bull = np.zeros(len(idx), dtype=bool); bull[:-30] = fwd30[:-30] > 0
        else:
            bull = reg[name]
        pos = np.where(bull, 1.0, -1.0)
        pos_l = np.zeros_like(pos); pos_l[1:] = pos[:-1]
        r = pos_l * btc_r
        t = np.abs(np.diff(pos_l, prepend=0))
        row = {"arm": f"BTC only: {name}"}
        for sname, (lo, hi) in SPLITS.items():
            s = stats(r, t, idx, pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC"))
            row[f"{sname} ann%"] = s.get("ann%", np.nan)
            row[f"{sname} sh"] = s.get("sharpe", np.nan)
        rows.append(row)
    # and buy-and-hold BTC for reference
    row = {"arm": "BTC buy & hold"}
    for sname, (lo, hi) in SPLITS.items():
        s = stats(btc_r, np.zeros_like(btc_r), idx, pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC"))
        row[f"{sname} ann%"] = s.get("ann%", np.nan); row[f"{sname} sh"] = s.get("sharpe", np.nan)
    rows.append(row)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n=== 4. does the factor add anything on top of the regime call?")
    print("   (switched momentum book) minus (BTC-only same regime), annualised %")
    rows = []
    for name in list(reg):
        bull = reg[name]
        r_f = np.where(bull, ret_L, ret_S) - COST * (np.where(bull, tn_L, tn_S) + np.abs(np.diff(bull.astype(float), prepend=0)) * 2)
        pos_l = np.zeros(len(idx)); pos_l[1:] = np.where(bull, 1.0, -1.0)[:-1]
        r_b = pos_l * btc_r - COST * np.abs(np.diff(pos_l, prepend=0))
        row = {"regime": name}
        for sname, (lo, hi) in SPLITS.items():
            m = (idx >= pd.Timestamp(lo, tz="UTC")) & (idx < pd.Timestamp(hi, tz="UTC"))
            row[sname] = (r_f[m].mean() - r_b[m].mean()) * 365 * 100
        rows.append(row)
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
