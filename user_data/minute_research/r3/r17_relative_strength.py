"""R17 Cross-Sectional Relative Strength vs BTC Momentum (TRAIN only).

Pre-registration:
  Data: 1h resample of 1m klines.
  Ratio = Close_alt / Close_BTC.
  Channel: Donchian High / Low of Ratio over lookback LB bars (excluding current bar).
  Volume condition: Volume_alt > V * SMA(Volume_alt, 20).
  Signals:
    Long: Ratio > Donchian_High and Vol > V * SMA(Vol).
    Short: Ratio < Donchian_Low and Vol > V * SMA(Vol).
  Exit: Time exit HOLD hours (4h, 12h, 24h).
  Grid: LB in {24, 72} x V in {1.0, 1.5} x HOLD in {4, 12, 24} x Arm in {long, short} = 24 cells.
  Universe: U60 (excluding BTC, 59 alts); Cost: 10 bp per side; TRAIN: 2025-01-01 .. 2025-10-01.
  Gate: net mean > 0, day t >= 1.5, >= 1 trade/day.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

GRID = list(itertools.product([24, 72], [1.0, 1.5], [4, 12, 24], ["long", "short"]))


def get_btc_1h():
    d = L.load("BTC")
    df = pd.DataFrame({"close": d["close"]}, index=pd.DatetimeIndex(d["date"]))
    return df.resample("1h").agg({"close": "last"}).dropna()["close"]


def one(base: str, btc_1h: pd.Series):
    if base == "BTC":
        return base, {}
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df1h = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    # Align with BTC
    idx = df1h.index.intersection(btc_1h.index)
    df1h = df1h.loc[idx]
    btc_close = btc_1h.loc[idx]
    
    ratio = (df1h["close"] / btc_close).to_numpy()
    vol = df1h["volume"].to_numpy()
    vol_sma = pd.Series(vol).rolling(20, min_periods=10).mean().to_numpy()
    
    c = L.cost_bps(base)
    d1h = {
        "open": df1h.open.to_numpy(np.float64),
        "high": df1h.high.to_numpy(np.float64),
        "low": df1h.low.to_numpy(np.float64),
        "close": df1h.close.to_numpy(np.float64),
        "volume": df1h.volume.to_numpy(np.float64),
        "date": df1h.index.to_series()
    }
    
    out = {}
    for LB in (24, 72):
        r_series = pd.Series(ratio)
        r_high = r_series.rolling(LB, min_periods=LB).max().shift(1).to_numpy()
        r_low = r_series.rolling(LB, min_periods=LB).min().shift(1).to_numpy()
        
        for V in (1.0, 1.5):
            long_sig = (ratio > r_high) & (vol > V * vol_sma)
            short_sig = (ratio < r_low) & (vol > V * vol_sma)
            
            for hold_h in (4, 12, 24):
                # Long arm
                l_idx = np.where(long_sig)[0]
                if len(l_idx) > 0:
                    out[(LB, V, hold_h, "long")] = L.H.run(d1h, l_idx, 1, hold=hold_h, cost_bps=c)
                else:
                    out[(LB, V, hold_h, "long")] = pd.DataFrame()
                    
                # Short arm
                s_idx = np.where(short_sig)[0]
                if len(s_idx) > 0:
                    out[(LB, V, hold_h, "short")] = L.H.run(d1h, s_idx, -1, hold=hold_h, cost_bps=c)
                else:
                    out[(LB, V, hold_h, "short")] = pd.DataFrame()
                    
    return base, out


if __name__ == "__main__":
    btc_1h = get_btc_1h()
    res = {}
    with ProcessPoolExecutor(16) as ex:
        futures = [ex.submit(one, b, btc_1h) for b in L.U60]
        for f in futures:
            base, out = f.result()
            if base != "BTC":
                res[base] = out
                
    rows = []
    days = 273
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res if cell in res[b]})
        rows.append({
            "LB": cell[0], "V": cell[1], "hold_h": cell[2], "arm": cell[3],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r17_train.csv", index=False)
