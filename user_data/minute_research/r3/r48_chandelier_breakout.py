"""R48 Chandelier ATR Breakout (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60
Timeframe: 15m
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

U60 = L.U60
GRID = [60, 120, 240]


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df15 = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    c = df15.close.to_numpy()
    o = df15.open.to_numpy()
    h = df15.high.to_numpy()
    l = df15.low.to_numpy()
    v = df15.volume.to_numpy()
    
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr22 = pd.Series(tr).rolling(22, min_periods=22).mean().to_numpy()
    
    hh22 = pd.Series(h).rolling(22, min_periods=22).max().shift(1).to_numpy()
    ll22 = pd.Series(l).rolling(22, min_periods=22).min().shift(1).to_numpy()
    v_sma = pd.Series(v).rolling(20, min_periods=20).mean().to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for hold_m in GRID:
        hold_bars = hold_m // 15
        
        long_cond = (c > hh22) & (v >= 2.0 * v_sma)
        short_cond = (c < ll22) & (v >= 2.0 * v_sma)
        
        l_idx = np.where(long_cond)[0]
        s_idx = np.where(short_cond)[0]
        idx = np.concatenate([l_idx, s_idx])
        side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
        
        if len(idx) == 0:
            out[hold_m] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[hold_m] = tr
        
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, U60):
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
    df.to_csv("r48_train.csv", index=False)
