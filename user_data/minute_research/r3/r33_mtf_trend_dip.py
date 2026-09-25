"""R33 4h Multi-Timeframe Trend Dip Buying (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60
Timeframe: 15m with 4h trend filter.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

U60 = L.U60
GRID = list(itertools.product([0.02, 0.04], [25, 30], [60, 120, 240, 480]))


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
    
    df15 = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    # 4h Trend (Must shift by 1 to prevent lookahead bias!)
    df4h = df.resample("4h").agg({"close": "last"}).dropna()
    df4h["ema20"] = df4h["close"].ewm(span=20, adjust=False).mean()
    df4h["ema50"] = df4h["close"].ewm(span=50, adjust=False).mean()
    df4h["bull_trend"] = ((df4h["close"] > df4h["ema50"]) & (df4h["ema20"] > df4h["ema50"])).shift(1).fillna(False)
    
    # Forward-fill completed 4h trend into 15m
    trend_4h = df4h["bull_trend"].reindex(df15.index, method="ffill").fillna(False).to_numpy()
    
    c = df15.close.to_numpy()
    o = df15.open.to_numpy()
    h = df15.high.to_numpy()
    l = df15.low.to_numpy()
    v = df15.volume.to_numpy()
    
    ema50_15m = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    rsi15 = calc_rsi(c, 14)
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for offset, rsi_thr, hold_m in GRID:
        hold_bars = hold_m // 15
        
        # Long only: strong 4h uptrend, but 15m oversold dip
        long_cond = trend_4h & (rsi15 < rsi_thr) & (c < ema50_15m * (1.0 - offset))
        idx = np.where(long_cond)[0]
        
        if len(idx) == 0:
            out[(offset, rsi_thr, hold_m)] = pd.DataFrame()
            continue
            
        side = np.ones(len(idx), dtype=np.int8)
        tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[(offset, rsi_thr, hold_m)] = tr
        
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
            "offset": cell[0], "rsi_thr": cell[1], "hold_m": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r33_train.csv", index=False)
