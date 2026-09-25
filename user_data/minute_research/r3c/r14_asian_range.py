"""R14 Asian Range Liquidity Sweep and Breakout (TRAIN only).

Pre-registration:
  Session: Asian session 00:00 - 07:00 UTC defines Asian High and Asian Low.
  Trading Window: 07:00 - 12:00 UTC (London open window).
  Sweep (Fade): High > Asian High * (1 + buf) but Close < Asian High -> Short (bearish sweep).
                Low < Asian Low * (1 - buf) but Close > Asian Low -> Long (bullish sweep).
  Breakout (Follow): Close > Asian High and Vol > V * SMA(Vol, 20) -> Long.
                     Close < Asian Low and Vol > V * SMA(Vol, 20) -> Short.
  Exit: Time exit HOLD minutes (60, 120, 240). Max 1 trade per coin per day.
  Universe: U60; Cost: 7.5/10 bp per side; TRAIN: 2025-01-01 .. 2025-10-01.
  Gate: net mean > 0, day t >= 1.5, >= 1 trade/day.
"""
from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

# Grid: Arm x Param x Hold
# Sweep: buf in {0.001, 0.003} x hold in {60, 120, 240} -> 6 cells
# Breakout: V in {1.5, 2.5} x hold in {60, 120, 240} -> 6 cells
SWEEP_CELLS = list(itertools.product(["sweep"], [0.001, 0.003], [60, 120, 240]))
BREAK_CELLS = list(itertools.product(["break"], [1.5, 2.5], [60, 120, 240]))
GRID = SWEEP_CELLS + BREAK_CELLS


def one(base: str):
    d = L.load(base)
    # Resample to 15m for clean bars
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    # 15m resample
    df15 = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    dates = df15.index
    hours = dates.hour
    mins = dates.minute
    day_group = dates.floor("D")
    
    # Identify Asian session (00:00 - 06:45 inclusive)
    asian_mask = hours < 7
    # Trading window (07:00 - 11:45 inclusive)
    trade_mask = (hours >= 7) & (hours < 12)
    
    # Compute Asian High and Low per day
    asian_highs = df15["high"].where(asian_mask).groupby(day_group).transform("max")
    asian_lows = df15["low"].where(asian_mask).groupby(day_group).transform("min")
    
    # Volume 20-period moving average
    vol_sma = df15["volume"].rolling(20, min_periods=10).mean()
    
    c = L.cost_bps(base)
    out = {}
    
    # Map signals back to 1m index or run on 15m bars
    # Using 15m bar simulation with H.run conventions
    d15 = {
        "open": df15.open.to_numpy(np.float64),
        "high": df15.high.to_numpy(np.float64),
        "low": df15.low.to_numpy(np.float64),
        "close": df15.close.to_numpy(np.float64),
        "volume": df15.volume.to_numpy(np.float64),
        "date": df15.index.to_series()
    }
    
    for arm, param, hold_m in GRID:
        hold_bars = hold_m // 15
        if arm == "sweep":
            buf = param
            # Long sweep: Low pierced Asian Low but closed above
            long_cond = trade_mask & (df15["low"] < asian_lows * (1 - buf)) & (df15["close"] > asian_lows)
            # Short sweep: High pierced Asian High but closed below
            short_cond = trade_mask & (df15["high"] > asian_highs * (1 + buf)) & (df15["close"] < asian_highs)
        else:
            v_mult = param
            # Long breakout: Close > Asian High with high vol
            long_cond = trade_mask & (df15["close"] > asian_highs) & (df15["volume"] > v_mult * vol_sma)
            # Short breakout: Close < Asian Low with high vol
            short_cond = trade_mask & (df15["close"] < asian_lows) & (df15["volume"] > v_mult * vol_sma)
            
        long_idx = np.where(long_cond.to_numpy())[0]
        short_idx = np.where(short_cond.to_numpy())[0]
        
        all_idx = np.concatenate([long_idx, short_idx])
        all_side = np.concatenate([np.ones(len(long_idx), dtype=np.int8), -np.ones(len(short_idx), dtype=np.int8)])
        
        if len(all_idx) == 0:
            out[(arm, param, hold_m)] = pd.DataFrame()
            continue
            
        order = np.argsort(all_idx)
        all_idx = all_idx[order]
        all_side = all_side[order]
        
        tr = L.H.run(d15, all_idx, all_side, hold=hold_bars, cost_bps=c)
        out[(arm, param, hold_m)] = tr

    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, L.U60):
            res[base] = out
            
    rows = []
    days = 273 # TRAIN days
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res})
        rows.append({
            "arm": cell[0], "param": cell[1], "hold_m": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r14_train.csv", index=False)
