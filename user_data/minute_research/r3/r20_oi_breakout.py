"""R20 OI Buildup Coiled Spring Breakout (TRAIN only).

Hypothesis & Mechanism (derived from skills/fable/oi-buildup-research-20260906.md):
  Prior research proved that when OI surges while price is flat (OI buildup),
  the subsequent price movement is 2.1x baseline (349 bps on 4h).
  Predicting direction with top-trader ratio failed, but a price breakout
  following the coiling event indicates the true direction of institutional flow.

Pre-registration:
  Data: 5m resample of 1m klines + 5m metrics (sum_open_interest).
  Event at bar t:
    OI surge: OI[t] / OI[t - W] - 1 >= OI_thr.
    Price flat: |Close[t] / Close[t - W] - 1| <= 0.5 * sigma_4h.
  Range: Box_High = max(High[t-W..t]), Box_Low = min(Low[t-W..t]).
  Breakout Trigger (monitored for next L=24 bars / 2h):
    Long: Close > Box_High.
    Short: Close < Box_Low.
  Entry: Next bar open. One position per coin at a time, 4h cooldown.
  Exit: Time exit HOLD minutes (120, 240, 720).
  Grid: W in {12, 48} (1h, 4h) x OI_thr in {0.05, 0.10} x HOLD in {120, 240, 720} = 12 cells.
  Universe: U60; Cost: 7.5/10 bp per side; TRAIN: 2025-01-01 .. 2025-10-01.
  Gate: net mean > 0, day t >= 1.5, >= 1 trade/day.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

B = "/root/freqtrade/user_data/data/binance_public"
GRID = [
    (12, 0.05, 120), (12, 0.05, 240), (12, 0.05, 720),
    (12, 0.08, 120), (12, 0.08, 240), (12, 0.08, 720),
    (48, 0.08, 120), (48, 0.08, 240), (48, 0.08, 720),
    (48, 0.12, 120), (48, 0.12, 240), (48, 0.12, 720),
]


def one(base: str):
    mf = f"{B}/metrics/{base}USDT.parquet"
    if not os.path.exists(mf):
        return base, {}
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"], "qv": d["quote_volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df5 = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "qv": "sum"
    }).dropna()
    
    mt = pd.read_parquet(mf).set_index("date")[["sum_open_interest"]]
    df5 = df5.join(mt, how="left")
    df5["oi"] = df5.sum_open_interest.ffill(limit=3)
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    oi = df5.oi.to_numpy()
    
    r4h = pd.Series(c).pct_change(48).to_numpy()
    sig4h = pd.Series(r4h).shift(48).rolling(2016, min_periods=500).std().to_numpy()
    
    cost = L.cost_bps(base)
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": df5.volume.to_numpy(),
        "date": df5.index.to_series()
    }
    
    out = {}
    n_bars = len(c)
    
    for W, OI_thr, hold_m in GRID:
        hold_bars = hold_m // 5
        # Calculate coiling events
        oi_change = np.where(pd.Series(oi).shift(W) > 0, oi / pd.Series(oi).shift(W) - 1.0, 0.0)
        coiled = (oi_change >= OI_thr) & (np.abs(r4h) <= 0.5 * sig4h)
        
        box_high = pd.Series(h).rolling(W, min_periods=W).max().shift(1).to_numpy()
        box_low = pd.Series(l).rolling(W, min_periods=W).min().shift(1).to_numpy()
        
        # Breakout occurs when coiled was true within the last 12 bars (1 hour)
        recent_coiled = pd.Series(coiled).rolling(12, min_periods=1).max().to_numpy().astype(bool)
        
        long_sig = recent_coiled & (c > box_high) & (pd.Series(c).shift(1) <= pd.Series(box_high).shift(1))
        short_sig = recent_coiled & (c < box_low) & (pd.Series(c).shift(1) >= pd.Series(box_low).shift(1))
        
        l_idx = np.where(long_sig)[0]
        s_idx = np.where(short_sig)[0]
        
        all_idx = np.concatenate([l_idx, s_idx])
        all_side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
        
        if len(all_idx) == 0:
            out[(W, OI_thr, hold_m)] = pd.DataFrame()
            continue
            
        order = np.argsort(all_idx)
        all_idx = all_idx[order]
        all_side = all_side[order]
        
        tr = L.H.run(d5, all_idx, all_side, hold=hold_bars, cost_bps=cost)
        out[(W, OI_thr, hold_m)] = tr
        
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
            "W": cell[0], "OI_thr": cell[1], "hold_m": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r20_train.csv", index=False)
