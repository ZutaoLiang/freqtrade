"""R19 Anchored VWAP Mean Reversion during Asian Low-Volatility Window (TRAIN only).

Pre-registration:
  Timeframe: 5m resampled from 1m.
  Anchored VWAP: Anchored at 00:00 UTC daily.
  VWAP = sum(Close * Vol) / sum(Vol).
  Std = sqrt(sum((Close - VWAP)^2 * Vol) / sum(Vol)).
  Z = (Close - VWAP) / max(Std, 1e-6).
  Window: Active between 01:00 UTC and 07:00 UTC.
  Signal:
    Long: Z <= -Z_thr (oversold relative to VWAP).
    Short: Z >= +Z_thr (overbought relative to VWAP).
  Exit:
    Touch VWAP or Time exit HOLD minutes (60, 120, 240), or end of Asian session (08:00 UTC).
  Grid: Z_thr in {2.0, 2.5, 3.0} x HOLD in {60, 120, 240} x Arm in {symmetric, long_only, short_only}.
  Let's test: Z_thr {2.0, 2.5} x HOLD {60, 120, 240} x Arm {symmetric, long_only, short_only} = 18 cells.
  Universe: U60; Cost: 7.5/10 bp per side; TRAIN: 2025-01-01 .. 2025-10-01.
  Gate: net mean > 0, day t >= 1.5, >= 1 trade/day.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

GRID = list(itertools.product([2.0, 2.5], [60, 120, 240], ["symmetric", "long_only", "short_only"]))


def calc_vwap_z(df5: pd.DataFrame):
    dates = df5.index
    days = dates.floor("D")
    c = df5.close.to_numpy()
    v = df5.volume.to_numpy()
    
    cv = c * v
    # Cumulative sums within each day
    df5["cv"] = cv
    df5["v"] = v
    df5["day"] = days
    
    cum_cv = df5.groupby("day")["cv"].cumsum().to_numpy()
    cum_v = df5.groupby("day")["v"].cumsum().to_numpy()
    
    vwap = np.where(cum_v > 0, cum_cv / cum_v, c)
    
    # Cumulative variance
    dev2_v = (c - vwap) ** 2 * v
    df5["dev2_v"] = dev2_v
    cum_dev2_v = df5.groupby("day")["dev2_v"].cumsum().to_numpy()
    std = np.sqrt(np.where(cum_v > 0, cum_dev2_v / cum_v, 0.0))
    
    z = np.where(std > 1e-6, (c - vwap) / std, 0.0)
    return vwap, z


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df5 = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    vwap, z = calc_vwap_z(df5)
    
    hours = df5.index.hour
    active_mask = (hours >= 1) & (hours < 7)
    
    c = L.cost_bps(base)
    d5 = {
        "open": df5.open.to_numpy(np.float64),
        "high": df5.high.to_numpy(np.float64),
        "low": df5.low.to_numpy(np.float64),
        "close": df5.close.to_numpy(np.float64),
        "volume": df5.volume.to_numpy(np.float64),
        "date": df5.index.to_series()
    }
    
    out = {}
    for Z_thr, hold_m, arm in GRID:
        hold_bars = hold_m // 5
        
        long_cond = active_mask & (z <= -Z_thr)
        short_cond = active_mask & (z >= Z_thr)
        
        if arm == "long_only":
            idx = np.where(long_cond)[0]
            side = np.ones(len(idx), dtype=np.int8)
        elif arm == "short_only":
            idx = np.where(short_cond)[0]
            side = -np.ones(len(idx), dtype=np.int8)
        else:
            l_idx = np.where(long_cond)[0]
            s_idx = np.where(short_cond)[0]
            idx = np.concatenate([l_idx, s_idx])
            side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
            
        if len(idx) == 0:
            out[(Z_thr, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=c)
        out[(Z_thr, hold_m, arm)] = tr
        
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
            "Z_thr": cell[0], "hold_m": cell[1], "arm": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r19_train.csv", index=False)
