"""R49 Cross-Sectional Funding Rate Dispersion Shock (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U162
Cadence: 8h settlements.
Logic:
  Cross-sectional standard deviation of funding rates across U162 at settlement T.
  When dispersion is extreme (std >= 0.04%):
    Short the top 5 highest funding coins (frenzy peak).
    Long the bottom 5 lowest funding coins (panic trough).
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

B = "/root/freqtrade/user_data/data/binance_public"
U = pd.read_csv("universe_rank.csv")
U162 = U[U.med_qv >= 1e7].base.tolist()

START = "2025-01-01"
END = "2025-10-01"

GRID = [480, 960]


def load_funding_matrix():
    funding_dict = {}
    for b in U162:
        ff = f"{B}/funding/{b}USDT.parquet"
        if not os.path.exists(ff):
            continue
        df = pd.read_parquet(ff, columns=["date", "funding_rate"]).set_index("date")
        df = df[(df.index >= "2024-12-15") & (df.index < END)]
        funding_dict[b] = df["funding_rate"]
    f_mat = pd.DataFrame(funding_dict).sort_index()
    f_mat = f_mat[(f_mat.index >= START) & (f_mat.index < END)]
    return f_mat


def one(base: str, f_mat: pd.DataFrame, high_disp_settles: set, top_coins_per_t: dict):
    kf = f"{B}/klines_1m/{base}USDT.parquet"
    if not os.path.exists(kf):
        return base, {}
        
    df = pd.read_parquet(kf, columns=["date", "open", "high", "low", "close", "volume"])
    df = df[(df.date >= START) & (df.date < END)]
    if len(df) < 50000:
        return base, {}
        
    df5 = df.set_index("date").resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    v = df5.volume.to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    # Map high dispersion settlements where this coin is in top 5
    sig_indices = []
    for t in high_disp_settles:
        if base in top_coins_per_t.get(t, []):
            if t in df5.index:
                idx = df5.index.get_loc(t)
                sig_indices.append(idx)
                
    if len(sig_indices) == 0:
        return base, {h: pd.DataFrame() for h in GRID}
        
    sig_idx = np.array(sig_indices, dtype=np.int64)
    sig_side = -np.ones(len(sig_idx), dtype=np.int8) # Short the top funding coins
    
    order = np.argsort(sig_idx)
    sig_idx = sig_idx[order]
    sig_side = sig_side[order]
    
    out = {}
    for hold_m in GRID:
        hold_bars = hold_m // 5
        tr = L.H.run(d5, sig_idx, sig_side, hold=hold_bars, cost_bps=cost)
        out[hold_m] = tr
        
    return base, out


def _worker(args):
    return one(args[0], args[1], args[2], args[3])


if __name__ == "__main__":
    f_mat = load_funding_matrix()
    disp = f_mat.std(axis=1)
    high_disp_settles = set(disp[disp >= 0.0004].index)
    
    top_coins_per_t = {}
    for t in high_disp_settles:
        row = f_mat.loc[t].dropna()
        if len(row) >= 30:
            top5 = row.nlargest(5).index.tolist()
            top_coins_per_t[t] = top5
            
    res = {}
    with ProcessPoolExecutor(16) as ex:
        args = [(b, f_mat, high_disp_settles, top_coins_per_t) for b in U162]
        for base, out in ex.map(_worker, args):
            if out:
                res[base] = out
                
    rows = []
    days = 273
    for hold_m in GRID:
        s = L.summarize({b: res[b][hold_m] for b in res if hold_m in res[b]})
        rows.append({
            "hold_m": hold_m,
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r49_train.csv", index=False)
