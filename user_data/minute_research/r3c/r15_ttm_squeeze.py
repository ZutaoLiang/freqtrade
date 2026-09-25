"""R15 TTM Volatility Squeeze into Expansion (TRAIN only).

Pre-registration:
  Timeframe: 15m resampled from 1m.
  Bollinger Bands: SMA(20), 2.0 std.
  Keltner Channel: EMA(20), K * ATR(20).
  Squeeze Condition: BB_upper < KC_upper and BB_lower > KC_lower (compression).
  Trigger: Squeeze was ON for >= S bars, and on the current bar Squeeze fires OFF (BB expands).
  Direction: Momentum = LinReg slope of (Close - (SMA(20) + (High+Low)/2)/2).
             If Momentum > 0 -> Long; If Momentum < 0 -> Short.
  Exit: Time exit HOLD minutes (60, 120, 240, 480).
  Grid: K in {1.0, 1.5} x S in {4, 8} x HOLD in {60, 120, 240, 480} = 16 cells.
  Universe: U60; Cost: 7.5/10 bp per side; TRAIN: 2025-01-01 .. 2025-10-01.
  Gate: net mean > 0, day t >= 1.5, >= 1 trade/day.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

GRID = list(itertools.product([1.0, 1.5], [4, 8], [60, 120, 240, 480]))


def calc_squeeze(df: pd.DataFrame, k_mult: float, min_s: int):
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    
    # 20 SMA & std
    sma20 = pd.Series(c).rolling(20, min_periods=20).mean().to_numpy()
    std20 = pd.Series(c).rolling(20, min_periods=20).std().to_numpy()
    bb_u = sma20 + 2.0 * std20
    bb_l = sma20 - 2.0 * std20
    
    # EMA 20 & ATR 20
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr20 = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    kc_u = ema20 + k_mult * atr20
    kc_l = ema20 - k_mult * atr20
    
    # Squeeze is on when BB is inside KC
    squeeze_on = (bb_u < kc_u) & (bb_l > kc_l)
    
    # Squeeze duration (number of consecutive bars on)
    s_count = np.zeros(len(c), dtype=np.int32)
    for i in range(1, len(c)):
        if squeeze_on[i]:
            s_count[i] = s_count[i - 1] + 1
        else:
            s_count[i] = 0
            
    # Trigger: was >= min_s bars on previous bar, and now squeeze_on is False
    fired = (np.roll(s_count, 1) >= min_s) & (~squeeze_on)
    fired[0] = False
    
    # Momentum: price relative to mean
    mid = (sma20 + (pd.Series(h).rolling(20).max().to_numpy() + pd.Series(l).rolling(20).min().to_numpy()) / 2) / 2
    delta = c - mid
    # 20-period linear regression slope of delta
    x = np.arange(20)
    x_mean = 9.5
    denom = np.sum((x - x_mean) ** 2)
    
    # Fast rolling linreg
    mom = np.zeros(len(c), dtype=np.float64)
    for i in range(20, len(c)):
        y = delta[i - 19:i + 1]
        mom[i] = np.sum((x - x_mean) * (y - np.mean(y))) / denom
        
    return fired, mom


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df15 = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    c = L.cost_bps(base)
    d15 = {
        "open": df15.open.to_numpy(np.float64),
        "high": df15.high.to_numpy(np.float64),
        "low": df15.low.to_numpy(np.float64),
        "close": df15.close.to_numpy(np.float64),
        "volume": df15.volume.to_numpy(np.float64),
        "date": df15.index.to_series()
    }
    
    out = {}
    for k_mult in (1.0, 1.5):
        for min_s in (4, 8):
            fired, mom = calc_squeeze(df15, k_mult, min_s)
            long_idx = np.where(fired & (mom > 0))[0]
            short_idx = np.where(fired & (mom < 0))[0]
            
            all_idx = np.concatenate([long_idx, short_idx])
            all_side = np.concatenate([np.ones(len(long_idx), dtype=np.int8), -np.ones(len(short_idx), dtype=np.int8)])
            
            if len(all_idx) == 0:
                for hold_m in (60, 120, 240, 480):
                    out[(k_mult, min_s, hold_m)] = pd.DataFrame()
                continue
                
            order = np.argsort(all_idx)
            all_idx = all_idx[order]
            all_side = all_side[order]
            
            for hold_m in (60, 120, 240, 480):
                hold_bars = hold_m // 15
                tr = L.H.run(d15, all_idx, all_side, hold=hold_bars, cost_bps=c)
                out[(k_mult, min_s, hold_m)] = tr
                
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
            "K": cell[0], "S": cell[1], "hold_m": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r15_train.csv", index=False)
