"""R21 EMA Offset Dip Buyer (NostalgiaForInfinity / ClucMay7 style mean reversion).

Pre-registration:
  Timeframe: 5m resampled from 1m.
  EMA: 50-period EMA of close.
  RSI: 14-period RSI.
  Conditions:
    Long: Close < EMA50 * (1 - Offset) and RSI14 < RSI_thr and Volume > 1.2 * SMA(Volume, 20).
    Short (symmetric): Close > EMA50 * (1 + Offset) and RSI14 > (100 - RSI_thr) and Volume > 1.2 * SMA(Volume, 20).
  Exit: Time exit HOLD minutes (60, 120, 240).
  Grid: Offset in {0.02, 0.035} x RSI_thr in {25, 30} x hold_m in {60, 120, 240} x Arm in {long_only, symmetric} = 24 cells.
  Universe: U60; Cost: 7.5/10 bp per side; TRAIN: 2025-01-01 .. 2025-10-01.
  Gate: net mean > 0, day t >= 1.5, >= 1 trade/day.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

GRID = list(itertools.product([0.02, 0.035], [25, 30], [60, 120, 240], ["long_only", "symmetric"]))


def calc_rsi(c: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df5 = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    v = df5.volume.to_numpy()
    
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    vol_sma = pd.Series(v).rolling(20, min_periods=10).mean().to_numpy()
    rsi14 = calc_rsi(c, 14)
    
    cost = L.cost_bps(base)
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    out = {}
    for offset, rsi_thr, hold_m, arm in GRID:
        hold_bars = hold_m // 5
        v_ok = v > 1.2 * vol_sma
        
        long_cond = (c < ema50 * (1.0 - offset)) & (rsi14 < rsi_thr) & v_ok
        short_cond = (c > ema50 * (1.0 + offset)) & (rsi14 > (100.0 - rsi_thr)) & v_ok
        
        l_idx = np.where(long_cond)[0]
        s_idx = np.where(short_cond)[0]
        
        if arm == "long_only":
            idx = l_idx
            side = np.ones(len(idx), dtype=np.int8)
        else:
            idx = np.concatenate([l_idx, s_idx])
            side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
            
        if len(idx) == 0:
            out[(offset, rsi_thr, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
        out[(offset, rsi_thr, hold_m, arm)] = tr
        
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
            "offset": cell[0], "rsi_thr": cell[1], "hold_m": cell[2], "arm": cell[3],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r21_train.csv", index=False)
