"""R41 Daily Volatility Compression Breakout (NR7 on 15m) (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

U60 = L.U60
GRID = list(itertools.product([120, 240, 480], ["both", "long", "short"]))


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    # Daily bars for NR7
    df1d = df.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    df1d["range"] = df1d["high"] - df1d["low"]
    # NR7: today's range is the smallest of the past 7 days (shifted 1 bar so yesterday's range was NR7)
    min_range_7 = df1d["range"].rolling(7, min_periods=7).min()
    is_nr7 = (df1d["range"] == min_range_7).shift(1).fillna(False)
    
    # 15m bars
    df15 = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    df15["is_nr7"] = is_nr7.reindex(df15.index, method="ffill").fillna(False)
    
    c = df15.close.to_numpy()
    o = df15.open.to_numpy()
    h = df15.high.to_numpy()
    l = df15.low.to_numpy()
    v = df15.volume.to_numpy()
    nr7 = df15.is_nr7.to_numpy()
    
    # First 4h range of the day (16 bars of 15m)
    hh16 = pd.Series(h).rolling(16, min_periods=16).max().shift(1).to_numpy()
    ll16 = pd.Series(l).rolling(16, min_periods=16).min().shift(1).to_numpy()
    v_sma = pd.Series(v).rolling(20, min_periods=20).mean().to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for hold_m, arm in GRID:
        hold_bars = hold_m // 15
        
        long_cond = nr7 & (c > hh16) & (v >= 2.0 * v_sma)
        short_cond = nr7 & (c < ll16) & (v >= 2.0 * v_sma)
        
        l_idx = np.where(long_cond)[0]
        s_idx = np.where(short_cond)[0]
        
        if arm == "long":
            idx = l_idx
            side = np.ones(len(idx), dtype=np.int8)
        elif arm == "short":
            idx = s_idx
            side = -np.ones(len(idx), dtype=np.int8)
        else:
            idx = np.concatenate([l_idx, s_idx])
            side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
            
        if len(idx) == 0:
            out[(hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[(hold_m, arm)] = tr
        
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, U60):
            if out:
                res[base] = out
                
    rows = []
    days = 273
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res if cell in res[b]})
        rows.append({
            "hold_m": cell[0], "arm": cell[1],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r41_train.csv", index=False)
