"""R47 Multi-Timeframe RSI Trend Pullback (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60
Timeframe: 5m with 1h trend filter.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

U60 = L.U60
GRID = [30, 60, 120]


def calc_rsi(c: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    return 100.0 - (100.0 / (1.0 + rs))


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    # 1h RSI (shifted by 1 bar to eliminate lookahead!)
    df1h = df.resample("1h").agg({"close": "last"}).dropna()
    c1h = df1h.close.to_numpy()
    rsi1h = calc_rsi(c1h, 14)
    df1h["bull1h"] = pd.Series(rsi1h > 60, index=df1h.index).shift(1).fillna(False)
    
    # 5m bars
    df5 = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    df5["bull1h"] = df1h["bull1h"].reindex(df5.index, method="ffill").fillna(False)
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    v = df5.volume.to_numpy()
    bull1h = df5.bull1h.to_numpy()
    
    rsi5 = calc_rsi(c, 14)
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    out = {}
    for hold_m in GRID:
        hold_bars = hold_m // 5
        long_cond = bull1h & (rsi5 < 30)
        idx = np.where(long_cond)[0]
        
        if len(idx) == 0:
            out[hold_m] = pd.DataFrame()
            continue
            
        side = np.ones(len(idx), dtype=np.int8)
        tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
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
    df.to_csv("r47_train.csv", index=False)
