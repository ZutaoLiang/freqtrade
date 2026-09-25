"""R29 Squeeze Momentum with Taker Order Flow Confirmation (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60
Timeframe: 15m
Logic:
  Bollinger Bands (20, 2.0) vs Keltner Channels (20, 1.5 ATR).
  Squeeze releases + Taker Buy Ratio confirmation.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

U60 = L.U60
GRID = list(itertools.product([0.55, 0.60, 0.65], [60, 120, 240], ["both", "long", "short"]))


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"],
        "qv": d["quote_volume"], "tbqv": d["taker_buy_quote_volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df15 = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "qv": "sum", "tbqv": "sum"
    }).dropna()
    
    c = df15.close.to_numpy()
    o = df15.open.to_numpy()
    h = df15.high.to_numpy()
    l = df15.low.to_numpy()
    v = df15.volume.to_numpy()
    qv = df15.qv.to_numpy()
    tbqv = df15.tbqv.to_numpy()
    
    # Taker buy ratio
    tb_ratio = np.where(qv > 0, tbqv / qv, 0.5)
    
    # Bollinger Bands (20, 2.0)
    sma20 = pd.Series(c).rolling(20, min_periods=20).mean().to_numpy()
    std20 = pd.Series(c).rolling(20, min_periods=20).std().to_numpy()
    bb_up = sma20 + 2.0 * std20
    bb_lo = sma20 - 2.0 * std20
    
    # Keltner Channel (20, 1.5 ATR)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr20 = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    kc_up = ema20 + 1.5 * atr20
    kc_lo = ema20 - 1.5 * atr20
    
    # Squeeze is ON when BB is inside KC
    squeeze_on = (bb_up < kc_up) & (bb_lo > kc_lo)
    squeeze_prev = pd.Series(squeeze_on).shift(1).fillna(False).to_numpy()
    # Squeeze releases when squeeze was ON and is now OFF
    squeeze_fire = squeeze_prev & (~squeeze_on)
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for tb_thr, hold_m, arm in GRID:
        hold_bars = hold_m // 15
        
        long_cond = squeeze_fire & (c > bb_up) & (tb_ratio >= tb_thr)
        short_cond = squeeze_fire & (c < bb_lo) & (tb_ratio <= (1.0 - tb_thr))
        
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
            out[(tb_thr, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[(tb_thr, hold_m, arm)] = tr
        
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
            "tb_thr": cell[0], "hold_m": cell[1], "arm": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r29_train.csv", index=False)
