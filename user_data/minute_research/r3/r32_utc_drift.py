"""R32 UTC 00:00 Daily Candle Close Drift Reversal (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60
Timeframe: 1h
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

U60 = L.U60
GRID = list(itertools.product([0.015, 0.025, 0.035], ["both", "long", "short"]))


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df1h = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    # 00:00 to 02:00 return
    # At 02:00 UTC bar close (which is 03:00 UTC start):
    # Check return from 00:00 open to 02:00 close
    df1h["hour"] = df1h.index.hour
    df1h["day"] = df1h.index.date
    
    # Rolling 2-hour return
    ret2h = (df1h["close"] / df1h["open"].shift(1)) - 1.0
    
    # Signal is at hour == 2 (i.e. bar 02:00-03:00 close)
    is_eval_time = (df1h["hour"] == 2).to_numpy()
    ret2_arr = ret2h.to_numpy()
    
    c = df1h.close.to_numpy()
    o = df1h.open.to_numpy()
    h = df1h.high.to_numpy()
    l = df1h.low.to_numpy()
    v = df1h.volume.to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d1h = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df1h.index.to_series()
    }
    
    out = {}
    hold_bars = 6 # 6 hours: 03:00 to 09:00
    
    for r_thr, arm in GRID:
        long_cond = is_eval_time & (ret2_arr <= -r_thr)
        short_cond = is_eval_time & (ret2_arr >= r_thr)
        
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
            out[(r_thr, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d1h, idx, side, hold=hold_bars, cost_bps=cost)
        out[(r_thr, arm)] = tr
        
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
            "r_thr": cell[0], "arm": cell[1],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r32_train.csv", index=False)
