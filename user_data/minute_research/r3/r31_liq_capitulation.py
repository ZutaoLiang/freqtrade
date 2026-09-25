"""R31 Liquidation Cascade Panic Capitulation (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U162
Timeframe: 15m
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

# Grid: OI_thr in {0.02, 0.04} x P_thr in {0.02, 0.03} x HOLD in {60, 120, 240} x Arm in {long_bounce, short_exhaust, both}
GRID = list(itertools.product([0.02, 0.04], [0.02, 0.03], [60, 120, 240], ["long_bounce", "short_exhaust", "both"]))


def one(base: str):
    mf = f"{B}/metrics/{base}USDT.parquet"
    kf = f"{B}/klines_1m/{base}USDT.parquet"
    if not os.path.exists(mf) or not os.path.exists(kf):
        return base, {}
        
    df_k = pd.read_parquet(kf, columns=["date", "open", "high", "low", "close", "volume"])
    df_k = df_k[(df_k.date >= "2024-12-30") & (df_k.date < END)]
    if len(df_k) < 50000:
        return base, {}
        
    df15 = df_k.set_index("date").resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    # 15m metrics
    df_m = pd.read_parquet(mf, columns=["date", "sum_open_interest_value"])
    df_m = df_m[(df_m.date >= "2024-12-30") & (df_m.date < END)]
    if len(df_m) < 10000:
        return base, {}
    m15 = df_m.set_index("date").resample("15min").last()["sum_open_interest_value"].dropna()
    
    df15["oi"] = m15.reindex(df15.index).ffill()
    df15 = df15[(df15.index >= START) & (df15.index < END)].dropna()
    if len(df15) < 1000:
        return base, {}
        
    c = df15.close.to_numpy()
    o = df15.open.to_numpy()
    h = df15.high.to_numpy()
    l = df15.low.to_numpy()
    v = df15.volume.to_numpy()
    oi = df15.oi.to_numpy()
    
    # 15m deltas
    doi = np.diff(oi, prepend=oi[0]) / np.maximum(oi, 1e-9)
    dp = (c / o) - 1.0
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for oi_thr, p_thr, hold_m, arm in GRID:
        hold_bars = hold_m // 15
        
        # Long capitulation bounce: OI plunged and price plunged
        l_cond = (doi <= -oi_thr) & (dp <= -p_thr)
        # Short squeeze exhaustion: OI plunged and price spiked
        s_cond = (doi <= -oi_thr) & (dp >= p_thr)
        
        l_idx = np.where(l_cond)[0]
        s_idx = np.where(s_cond)[0]
        
        if arm == "long_bounce":
            idx = l_idx
            side = np.ones(len(idx), dtype=np.int8)
        elif arm == "short_exhaust":
            idx = s_idx
            side = -np.ones(len(idx), dtype=np.int8)
        else:
            idx = np.concatenate([l_idx, s_idx])
            side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
            
        if len(idx) == 0:
            out[(oi_thr, p_thr, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[(oi_thr, p_thr, hold_m, arm)] = tr
        
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, U162):
            if out:
                res[base] = out
                
    rows = []
    days = 273
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res if cell in res[b]})
        rows.append({
            "oi_thr": cell[0], "p_thr": cell[1], "hold_m": cell[2], "arm": cell[3],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r31_train.csv", index=False)
