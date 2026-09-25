"""R46 Keltner Channel Volatility Explosion (TRAIN only).

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
GRID = list(itertools.product([2.0, 2.5], [60, 120, 240]))


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
    atr20 = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    v_sma = pd.Series(v).rolling(20, min_periods=20).mean().to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for kc_mult, hold_m in GRID:
        hold_bars = hold_m // 15
        kc_up = ema20 + kc_mult * atr20
        kc_lo = ema20 - kc_mult * atr20
        
        long_cond = (c > kc_up) & (v >= 2.5 * v_sma)
        short_cond = (c < kc_lo) & (v >= 2.5 * v_sma)
        
        l_idx = np.where(long_cond)[0]
        s_idx = np.where(short_cond)[0]
        idx = np.concatenate([l_idx, s_idx])
        side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
        
        if len(idx) == 0:
            out[(kc_mult, hold_m)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[(kc_mult, hold_m)] = tr
        
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
            "kc_mult": cell[0], "hold_m": cell[1],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r46_train.csv", index=False)
