"""R38 1h SuperTrend + ADX Trend Following (TRAIN only).

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
GRID = list(itertools.product([2.5, 3.0], [20, 25], [12, 24, 48]))


def calc_adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(c)
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = h[1:] - l[1:]
    tr2 = np.abs(h[1:] - c[:-1])
    tr3 = np.abs(l[1:] - c[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    p_dm = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    m_dm = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    
    p_di = np.where(atr > 0, 100.0 * p_dm / atr, 0.0)
    m_di = np.where(atr > 0, 100.0 * m_dm / atr, 0.0)
    
    dx = np.where((p_di + m_di) > 0, 100.0 * np.abs(p_di - m_di) / (p_di + m_di), 0.0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    return np.concatenate([[0.0], adx])


def calc_supertrend(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 10, mult: float = 3.0):
    n = len(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    
    hl2 = (h + l) / 2.0
    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    trend = np.ones(n, dtype=np.int8) # 1 = bull, -1 = bear
    
    upper_band[0] = upper_basic[0]
    lower_band[0] = lower_basic[0]
    
    for i in range(1, n):
        # Lower band
        if lower_basic[i] > lower_band[i-1] or c[i-1] < lower_band[i-1]:
            lower_band[i] = lower_basic[i]
        else:
            lower_band[i] = lower_band[i-1]
            
        # Upper band
        if upper_basic[i] < upper_band[i-1] or c[i-1] > upper_band[i-1]:
            upper_band[i] = upper_basic[i]
        else:
            upper_band[i] = upper_band[i-1]
            
        # Trend
        if trend[i-1] == 1:
            if c[i] < lower_band[i]:
                trend[i] = -1
            else:
                trend[i] = 1
        else:
            if c[i] > upper_band[i]:
                trend[i] = 1
            else:
                trend[i] = -1
                
    return trend


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df1h = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    c = df1h.close.to_numpy()
    o = df1h.open.to_numpy()
    h = df1h.high.to_numpy()
    l = df1h.low.to_numpy()
    v = df1h.volume.to_numpy()
    
    adx14 = calc_adx(h, l, c, 14)
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d1h = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df1h.index.to_series()
    }
    
    out = {}
    for st_mult, adx_thr, hold_h in GRID:
        trend = calc_supertrend(h, l, c, period=10, mult=st_mult)
        trend_prev = np.roll(trend, 1)
        trend_prev[0] = trend[0]
        
        # Bull flip: was bear, now bull, and ADX confirms
        bull_flip = (trend == 1) & (trend_prev == -1) & (adx14 >= adx_thr)
        bear_flip = (trend == -1) & (trend_prev == 1) & (adx14 >= adx_thr)
        
        l_idx = np.where(bull_flip)[0]
        s_idx = np.where(bear_flip)[0]
        idx = np.concatenate([l_idx, s_idx])
        side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
        
        if len(idx) == 0:
            out[(st_mult, adx_thr, hold_h)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d1h, idx, side, hold=hold_h, cost_bps=cost)
        out[(st_mult, adx_thr, hold_h)] = tr
        
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
            "st_mult": cell[0], "adx_thr": cell[1], "hold_h": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r38_train.csv", index=False)
