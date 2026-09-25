"""R25 Chande Momentum Oscillator (CMO) Volatility Trend Breakout (TRAIN only).

Pre-registration:
  Timeframe: 15m resampled from 1m.
  CMO(N): 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses) over N bars.
  Filter: Trend filter EMA50 and Volatility filter ATR(14) > SMA(ATR, 20).
  Signals:
    Long: CMO > +C_thr and Close > EMA50 and ATR14 > ATR_sma.
    Short: CMO < -C_thr and Close < EMA50 and ATR14 > ATR_sma.
  Exit: Time exit HOLD minutes (60, 120, 240).
  Grid: N in {9, 14} x C_thr in {40, 50} x HOLD in {60, 120, 240} x Arm in {both, long, short} = 24 cells (test key 12 cells).
  Universe: U60; Cost: 7.5/10 bp per side; TRAIN: 2025-01-01 .. 2025-10-01.
  Gate: net mean > 0, day t >= 1.5, >= 1 trade/day.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

GRID = list(itertools.product([14], [40, 50], [60, 120, 240], ["both", "long", "short"]))


def calc_cmo(c: np.ndarray, period: int = 14) -> np.ndarray:
    diff = np.diff(c, prepend=c[0])
    pos = np.where(diff > 0, diff, 0.0)
    neg = np.where(diff < 0, -diff, 0.0)
    
    pos_sum = pd.Series(pos).rolling(period, min_periods=period).sum().to_numpy()
    neg_sum = pd.Series(neg).rolling(period, min_periods=period).sum().to_numpy()
    
    tot = pos_sum + neg_sum
    cmo = np.where(tot > 0, 100.0 * (pos_sum - neg_sum) / tot, 0.0)
    return cmo


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
    
    cmo = calc_cmo(c, 14)
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr14 = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    atr_sma = pd.Series(atr14).rolling(20, min_periods=10).mean().to_numpy()
    
    cost = L.cost_bps(base)
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for period, C_thr, hold_m, arm in GRID:
        hold_bars = hold_m // 15
        vol_ok = atr14 > atr_sma
        
        long_sig = (cmo > C_thr) & (c > ema50) & vol_ok
        short_sig = (cmo < -C_thr) & (c < ema50) & vol_ok
        
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
            out[(period, C_thr, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr_res = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[(period, C_thr, hold_m, arm)] = tr_res
        
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
            "period": cell[0], "C_thr": cell[1], "hold_m": cell[2], "arm": cell[3],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r25_train.csv", index=False)
