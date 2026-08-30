"""Event study on the niche 1m perp band: overshoot reversion vs BTC lead-lag.

Overshoot: idiosyncratic 1m move of |z| >= m_sigma with a volume spike while BTC is
quiet -> forward return measured in the reversion direction, entry at next-bar open.
Lead-lag: lagged correlation of the pair's 1m return against BTC's.

Output: per-pair rows plus pooled summaries, both for train and holdout windows.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

DATA = "/root/freqtrade/user_data/data/binance_public/freqtrade/futures"
SCRATCH = "/root/freqtrade/user_data/niche_work"
TRAIN = ("2025-11-01", "2026-05-31")
HOLD = ("2026-06-01", "2026-08-16")
SIGMA_WIN = 240
VOL_WIN = 240
HORIZONS = (1, 3, 5, 10, 20)
M_SIGMAS = (4.0, 5.0, 6.0)
VOL_Z_MIN = 2.0
BTC_VETO = 2.0


def load(sym):
    df = pd.read_feather(f"{DATA}/{sym}-1m-futures.feather").set_index("date")
    return df[~df.index.duplicated()]


def btc_frame():
    b = load("BTC_USDT_USDT")
    r = np.log(b["close"]).diff()
    return pd.DataFrame({"btc_z": r / r.rolling(SIGMA_WIN).std()})


def study(sym, btc):
    try:
        df = load(sym)
    except Exception:
        return []
    df = df.join(btc, how="left")
    r = np.log(df["close"]).diff()
    sigma = r.rolling(SIGMA_WIN).std()
    move_z = r / sigma
    v = df["volume"]
    vol_z = (v - v.rolling(VOL_WIN).mean()) / v.rolling(VOL_WIN).std()
    nxt_open = df["open"].shift(-1)
    fwd = {h: np.log(df["close"].shift(-h) / nxt_open) for h in HORIZONS}

    rows = []
    for label, (a, b) in (("train", TRAIN), ("hold", HOLD)):
        win = (df.index >= a) & (df.index <= b)
        for m in M_SIGMAS:
            base = win & (vol_z > VOL_Z_MIN) & (df["btc_z"].abs() < BTC_VETO) & sigma.notna()
            up = base & (move_z > m)      # overshot up -> short
            dn = base & (move_z < -m)     # overshot down -> long
            n = int(up.sum() + dn.sum())
            if n < 20:
                continue
            row = dict(sym=sym, window=label, m_sigma=m, n=n)
            for h in HORIZONS:
                pnl = pd.concat([-fwd[h][up], fwd[h][dn]]).dropna()
                if len(pnl) == 0:
                    continue
                row[f"bps_{h}"] = 1e4 * pnl.mean()
                row[f"med_{h}"] = 1e4 * pnl.median()
                row[f"hit_{h}"] = (pnl > 0).mean()
            rows.append(row)
    return rows


def leadlag(sym, btc_ret):
    try:
        df = load(sym)
    except Exception:
        return None
    r = np.log(df["close"]).diff()
    a, b = TRAIN
    r = r[(r.index >= a) & (r.index <= b)]
    j = pd.concat([r.rename("t"), btc_ret.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 50_000:
        return None
    out = dict(sym=sym, n=len(j))
    for k in range(0, 4):
        out[f"lag{k}"] = j["t"].corr(j["b"].shift(k))
    return out


def _work(sym):
    btc = _work.btc
    return study(sym, btc), leadlag(sym, _work.btc_ret)


def _init():
    b = load("BTC_USDT_USDT")
    _work.btc = pd.DataFrame({"btc_z": np.log(b["close"]).diff()
                              / np.log(b["close"]).diff().rolling(SIGMA_WIN).std()})
    _work.btc_ret = np.log(b["close"]).diff()


def main():
    u = pd.read_csv(f"{SCRATCH}/universe.csv")
    band = u[(u.med_qv > 3e6) & (u.med_qv < 6e7) & (u.days >= 250)]
    syms = band.sym.tolist()
    print(f"universe in band: {len(syms)}", flush=True)

    ev, ll = [], []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16,
                             initializer=_init) as ex:
        for e, l in ex.map(_work, syms, chunksize=2):
            ev.extend(e)
            if l:
                ll.append(l)

    ev = pd.DataFrame(ev)
    ll = pd.DataFrame(ll)
    ev.to_csv(f"{SCRATCH}/overshoot_events.csv", index=False)
    ll.to_csv(f"{SCRATCH}/leadlag.csv", index=False)

    print("\n=== overshoot pooled (weighted by n) ===")
    for (w, m), g in ev.groupby(["window", "m_sigma"]):
        tot = g.n.sum()
        line = [f"{w:5s} m={m} pairs={len(g):3d} n={tot:6d}"]
        for h in HORIZONS:
            c = f"bps_{h}"
            if c in g:
                wm = (g[c] * g.n).sum() / tot
                hm = (g[f"hit_{h}"] * g.n).sum() / tot
                line.append(f"{h}m:{wm:+6.1f}bps hit{hm:.3f}")
        print("  ".join(line))

    print("\n=== lead-lag (BTC leader) pooled ===")
    print(ll[[f"lag{k}" for k in range(4)]].describe().round(4).to_string())
    print("\ntop lag1 pairs:")
    print(ll.sort_values("lag1", ascending=False).head(10).round(4).to_string())


if __name__ == "__main__":
    main()
