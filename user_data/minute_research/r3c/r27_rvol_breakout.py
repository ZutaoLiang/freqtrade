"""R27 High Relative Volume (RVOL) Breakout with 1h Trend Alignment (TRAIN only).

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
GRID = list(itertools.product([2.5, 3.5], [30, 60, 120], ["both", "long", "short"]))


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    # 5m bars
    df5 = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    # 1h trend
    df1h = df.resample("1h").agg({"close": "last"}).dropna()
    df1h["ema50"] = df1h["close"].ewm(span=50, adjust=False).mean()
    df1h["trend_up"] = df1h["close"] > df1h["ema50"]
    
    # Reindex 1h trend to 5m forward-filled
    trend_series = df1h["trend_up"].reindex(df5.index, method="ffill").fillna(False).to_numpy()
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    v = df5.volume.to_numpy()
    
    # RVOL over 20 bars
    v_sma20 = pd.Series(v).rolling(20, min_periods=20).mean().to_numpy()
    rvol = np.where(v_sma20 > 0, v / v_sma20, 0.0)
    
    # Donchian 20
    hh20 = pd.Series(h).rolling(20, min_periods=20).max().shift(1).to_numpy()
    ll20 = pd.Series(l).rolling(20, min_periods=20).min().shift(1).to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    out = {}
    for rvol_thr, hold_m, arm in GRID:
        hold_bars = hold_m // 5
        
        long_cond = (rvol >= rvol_thr) & (c > hh20) & trend_series
        short_cond = (rvol >= rvol_thr) & (c < ll20) & (~trend_series)
        
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
            out[(rvol_thr, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
        out[(rvol_thr, hold_m, arm)] = tr
        
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
            "rvol_thr": cell[0], "hold_m": cell[1], "arm": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r27_train.csv", index=False)
