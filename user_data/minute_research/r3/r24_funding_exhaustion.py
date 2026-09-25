"""R24 Multi-Settlement Funding Rate Exhaustion Fade (TRAIN only).

Pre-registration:
  Data: 5m klines + funding rates from binance_public.
  Condition:
    Longs exhausted: fr(T-16h) >= F and fr(T-8h) >= F and fr(T) >= F (3 consecutive settlements).
                     -> Short (fade the overcrowded longs).
    Shorts exhausted: fr(T-16h) <= -F and fr(T-8h) <= -F and fr(T) <= -F.
                     -> Long (fade the overcrowded shorts).
  Entry: T+5m open.
  Exit: Time exit HOLD minutes (480, 960, 1440) [8h, 16h, 24h].
  Grid: F in {0.0003, 0.0005, 0.0008} x HOLD in {480, 960, 1440} x Arm in {fade_both, short_only, long_only} = 27 cells (take key 12 cells).
  Universe: U162; Cost: 10 bp per side; TRAIN: 2025-01-01 .. 2025-10-01.
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
U = pd.read_csv("universe_rank.csv")
U162 = U[U.med_qv >= 1e7].base.tolist()

GRID = list(itertools.product([0.0003, 0.0005], [480, 960], ["fade_both", "short_only", "long_only"]))


def one(base: str):
    ff = f"{B}/funding/{base}USDT.parquet"
    kf = f"{B}/klines_1m/{base}USDT.parquet"
    if not os.path.exists(ff) or not os.path.exists(kf):
        return base, {}
    d = pd.read_parquet(kf, columns=["date", "open", "high", "low", "close", "volume"])
    d = d[(d.date >= "2025-01-01") & (d.date < "2025-10-01")]
    if len(d) < 50000:
        return base, {}
        
    df5 = d.set_index("date").resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    f_df = pd.read_parquet(ff).set_index("date")[["funding_rate"]]
    # Rolling 3 settlements
    f_series = f_df["funding_rate"].sort_index()
    f1 = f_series.shift(1)
    f2 = f_series.shift(2)
    
    f_df["f0"] = f_series
    f_df["f1"] = f1
    f_df["f2"] = f2
    
    df5 = df5.join(f_df, how="left")
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    v = df5.volume.to_numpy()
    f0 = df5.f0.to_numpy()
    f1 = df5.f1.to_numpy()
    f2 = df5.f2.to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    out = {}
    for F_thr, hold_m, arm in GRID:
        hold_bars = hold_m // 5
        
        long_exhaust = (f0 >= F_thr) & (f1 >= F_thr) & (f2 >= F_thr)
        short_exhaust = (f0 <= -F_thr) & (f1 <= -F_thr) & (f2 <= -F_thr)
        
        # When longs are exhausted, we fade by going SHORT
        # When shorts are exhausted, we fade by going LONG
        s_idx = np.where(long_exhaust)[0]
        l_idx = np.where(short_exhaust)[0]
        
        if arm == "short_only":
            idx = s_idx
            side = -np.ones(len(idx), dtype=np.int8)
        elif arm == "long_only":
            idx = l_idx
            side = np.ones(len(idx), dtype=np.int8)
        else:
            idx = np.concatenate([l_idx, s_idx])
            side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
            
        if len(idx) == 0:
            out[(F_thr, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
        out[(F_thr, hold_m, arm)] = tr
        
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, U162):
            if out:
                res[base] = out
                
    rows = []
    days = 273
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res if cell in res[b]})
        rows.append({
            "F_thr": cell[0], "hold_m": cell[1], "arm": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r24_train.csv", index=False)
