"""R23 Inside Bar / NR4 Volatility Compression Breakout (TRAIN only).

Pre-registration:
  Data: 1h bars constructed from 1m klines.
  Pattern at hour H:
    Inside Bar: High[H] < High[H-1] and Low[H] > Low[H-1].
    NR4: (High[H] - Low[H]) is min of last 4 hourly ranges.
  Breakout in subsequent hour (evaluated on 5m bars):
    Long: 5m Close > High[H] and Close[t-1] <= High[H].
    Short: 5m Close < Low[H] and Close[t-1] >= Low[H].
  Exit: Time exit HOLD minutes (60, 120, 240, 480).
  Grid: Pattern in {inside, nr4} x HOLD in {60, 120, 240, 480} x Arm in {both, long, short} = 24 cells.
  Universe: U60; Cost: 7.5/10 bp per side; TRAIN: 2025-01-01 .. 2025-10-01.
  Gate: net mean > 0, day t >= 1.5, >= 1 trade/day.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

GRID = list(itertools.product(["inside", "nr4"], [60, 120, 240, 480], ["both", "long", "short"]))


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    # 1h bars
    df1h = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()
    
    h1h = df1h.high.to_numpy()
    l1h = df1h.low.to_numpy()
    rng = h1h - l1h
    
    # Inside bar condition on 1h
    is_inside = (h1h < np.roll(h1h, 1)) & (l1h > np.roll(l1h, 1))
    is_inside[0] = False
    
    # NR4 condition on 1h
    min_rng4 = pd.Series(rng).rolling(4, min_periods=4).min().to_numpy()
    is_nr4 = (rng == min_rng4)
    
    df1h["inside"] = is_inside
    df1h["nr4"] = is_nr4
    df1h["box_high"] = h1h
    df1h["box_low"] = l1h
    
    # 5m bars
    df5 = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    # Map 1h pattern shifted by 1 hour to 5m
    df1h_shifted = df1h.shift(1) # condition formed on completed 1h bar
    df5["inside_h"] = df1h_shifted["box_high"].reindex(df5.index, method="ffill")
    df5["inside_l"] = df1h_shifted["box_low"].reindex(df5.index, method="ffill")
    df5["is_inside"] = df1h_shifted["inside"].reindex(df5.index, method="ffill").fillna(False).astype(bool)
    df5["is_nr4"] = df1h_shifted["nr4"].reindex(df5.index, method="ffill").fillna(False).astype(bool)
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    v = df5.volume.to_numpy()
    
    cost = L.cost_bps(base)
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    out = {}
    for pat, hold_m, arm in GRID:
        hold_bars = hold_m // 5
        pat_active = df5["is_inside"].to_numpy() if pat == "inside" else df5["is_nr4"].to_numpy()
        b_high = df5["inside_h"].to_numpy()
        b_low = df5["inside_l"].to_numpy()
        
        long_sig = pat_active & (c > b_high) & (pd.Series(c).shift(1).to_numpy() <= b_high)
        short_sig = pat_active & (c < b_low) & (pd.Series(c).shift(1).to_numpy() >= b_low)
        
        l_idx = np.where(long_sig)[0]
        s_idx = np.where(short_sig)[0]
        
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
            out[(pat, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
        out[(pat, hold_m, arm)] = tr
        
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, L.U60):
            res[base] = out
            
    rows = []
    days = 273
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res if cell in res[b]})
        rows.append({
            "pat": cell[0], "hold_m": cell[1], "arm": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r23_train.csv", index=False)
