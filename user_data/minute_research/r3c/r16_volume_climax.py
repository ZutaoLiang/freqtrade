"""R16 Volume Climax / Panic Capitulation Snapback vs Continuation (TRAIN only).

Pre-registration:
  Timeframe: 15m resampled from 1m.
  ATR(14): Average True Range over 14 bars.
  Drop condition: (Open - Close) >= K * ATR(14) on a red candle.
  Volume condition: Volume >= V * SMA(Volume, 20).
  Trigger: Climax candle satisfies Drop and Volume.
  Arms:
    fade: Long on climax candle close (expecting snapback).
    follow: Short on climax candle close (expecting cascading continuation).
  Exit: Time exit HOLD minutes (60, 120, 240, 480).
  Grid: K in {2.0, 3.0} x V in {3.0, 5.0} x HOLD in {60, 120, 240, 480} x Arm in {fade, follow} = 32 cells (take subset of 16 key cells).
  Subset 16 cells:
    K {2.0, 3.0} x V {3.0, 5.0} x HOLD {120, 240} x Arm {fade, follow} = 16 cells.
  Universe: U60; Cost: 7.5/10 bp per side; TRAIN: 2025-01-01 .. 2025-10-01.
  Gate: net mean > 0, day t >= 1.5, >= 1 trade/day.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

GRID = list(itertools.product([2.0, 3.0], [3.0, 5.0], [120, 240], ["fade", "follow"]))


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
    atr14 = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    vol_sma = pd.Series(v).rolling(20, min_periods=10).mean().to_numpy()
    
    cost = L.cost_bps(base)
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for K, V, hold_m, arm in GRID:
        hold_bars = hold_m // 15
        # Large drop + surge volume
        drop = (o - c) >= K * atr14
        vol_surge = v >= V * vol_sma
        is_red = c < o
        
        climax = drop & vol_surge & is_red
        idx = np.where(climax)[0]
        
        if len(idx) == 0:
            out[(K, V, hold_m, arm)] = pd.DataFrame()
            continue
            
        side = 1 if arm == "fade" else -1
        tr_res = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[(K, V, hold_m, arm)] = tr_res
        
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, L.U60):
            res[base] = out
            
    rows = []
    days = 273
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res})
        rows.append({
            "K": cell[0], "V": cell[1], "hold_m": cell[2], "arm": cell[3],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r16_train.csv", index=False)
