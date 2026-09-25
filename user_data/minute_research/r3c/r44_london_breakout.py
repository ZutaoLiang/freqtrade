"""R44 Asian Session Low-Vol Range Breakout into London Open (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60
Timeframe: 15m
Logic:
  Asian range: 00:00 - 06:00 UTC.
  Compression: Range <= 3.0%.
  London trigger window: 07:00 - 09:00 UTC.
  Breakout above Asian High -> Long; Breakout below Asian Low -> Short.
  Exit: 12:00 UTC.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

U60 = L.U60
GRID = ["both", "long", "short"]


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df15 = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    df15["hour"] = df15.index.hour
    df15["date_only"] = df15.index.date
    
    # Asian session is hour 0 to 5 inclusive
    is_asian = (df15["hour"] >= 0) & (df15["hour"] < 6)
    
    # Calculate daily Asian high and low
    asian_df = df15[is_asian].groupby("date_only").agg(
        asian_h=("high", "max"),
        asian_l=("low", "min")
    )
    asian_df["asian_range"] = (asian_df["asian_h"] - asian_df["asian_l"]) / asian_df["asian_l"]
    asian_df["is_compressed"] = asian_df["asian_range"] <= 0.035
    
    # Map back to 15m
    df15 = df15.join(asian_df, on="date_only", how="left")
    
    # Trigger window: 07:00 to 08:45 UTC
    is_london = (df15["hour"] >= 7) & (df15["hour"] <= 8)
    
    c = df15.close.to_numpy()
    o = df15.open.to_numpy()
    h = df15.high.to_numpy()
    l = df15.low.to_numpy()
    v = df15.volume.to_numpy()
    ah = df15.asian_h.to_numpy()
    al = df15.asian_l.to_numpy()
    comp = df15.is_compressed.fillna(False).to_numpy()
    london = is_london.to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    hold_bars = 20 # 5 hours (20 bars of 15m)
    
    for arm in GRID:
        long_cond = london & comp & (c > ah)
        short_cond = london & comp & (c < al)
        
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
            out[arm] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[arm] = tr
        
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, U60):
            if out:
                res[base] = out
                
    rows = []
    days = 273
    for arm in GRID:
        s = L.summarize({b: res[b][arm] for b in res if arm in res[b]})
        rows.append({
            "arm": arm,
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r44_train.csv", index=False)
