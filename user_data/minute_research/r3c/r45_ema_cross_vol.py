"""R45 5m EMA 9/21 Cross with Volatility Expansion Filter (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60
Timeframe: 5m
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

U60 = L.U60
GRID = list(itertools.product([1.2, 1.5], [30, 60, 120]))


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df5 = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    v = df5.volume.to_numpy()
    
    ema9 = pd.Series(c).ewm(span=9, adjust=False).mean().to_numpy()
    ema21 = pd.Series(c).ewm(span=21, adjust=False).mean().to_numpy()
    
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().to_numpy()
    atr_sma = pd.Series(atr14).rolling(20, min_periods=20).mean().to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    out = {}
    
    # Crossovers
    bull_cross = (ema9 > ema21) & (np.roll(ema9, 1) <= np.roll(ema21, 1))
    bear_cross = (ema9 < ema21) & (np.roll(ema9, 1) >= np.roll(ema21, 1))
    bull_cross[0] = False
    bear_cross[0] = False
    
    for atr_mult, hold_m in GRID:
        hold_bars = hold_m // 5
        
        long_cond = bull_cross & (atr14 >= atr_mult * atr_sma)
        short_cond = bear_cross & (atr14 >= atr_mult * atr_sma)
        
        l_idx = np.where(long_cond)[0]
        s_idx = np.where(short_cond)[0]
        idx = np.concatenate([l_idx, s_idx])
        side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
        
        if len(idx) == 0:
            out[(atr_mult, hold_m)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
        out[(atr_mult, hold_m)] = tr
        
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
            "atr_mult": cell[0], "hold_m": cell[1],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r45_train.csv", index=False)
