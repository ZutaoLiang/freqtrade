"""R37 Multi-Day High-Low Liquidity Sweep + Reversal (Turtle Soup / Judas Swing).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60
Timeframe: 15m
Logic:
  24h (96 bars) rolling High and Low.
  Swept High: bar High > HH, but bar Close < HH, and (High - HH) / HH <= max_sweep. -> SHORT.
  Swept Low:  bar Low < LL, but bar Close > LL, and (LL - Low) / LL <= max_sweep. -> LONG.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

U60 = L.U60
GRID = list(itertools.product([0.005, 0.010, 0.015], [60, 120, 240], ["both", "fade_high", "fade_low"]))


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
    
    hh96 = pd.Series(h).rolling(96, min_periods=96).max().shift(1).to_numpy()
    ll96 = pd.Series(l).rolling(96, min_periods=96).min().shift(1).to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for max_sweep, hold_m, arm in GRID:
        hold_bars = hold_m // 15
        
        # Fade high: probed above HH but closed back below HH
        s_cond = (h > hh96) & (c < hh96) & ((h - hh96) / hh96 <= max_sweep) & ((h - hh96) / hh96 > 0.001)
        # Fade low: probed below LL but closed back above LL
        l_cond = (l < ll96) & (c > ll96) & ((ll96 - l) / ll96 <= max_sweep) & ((ll96 - l) / ll96 > 0.001)
        
        s_idx = np.where(s_cond)[0]
        l_idx = np.where(l_cond)[0]
        
        if arm == "fade_high":
            idx = s_idx
            side = -np.ones(len(idx), dtype=np.int8)
        elif arm == "fade_low":
            idx = l_idx
            side = np.ones(len(idx), dtype=np.int8)
        else:
            idx = np.concatenate([l_idx, s_idx])
            side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
            
        if len(idx) == 0:
            out[(max_sweep, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[(max_sweep, hold_m, arm)] = tr
        
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
            "max_sweep": cell[0], "hold_m": cell[1], "arm": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r37_train.csv", index=False)
