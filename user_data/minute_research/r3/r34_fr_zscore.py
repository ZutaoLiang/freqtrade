"""R34 Funding Rate Z-score Dislocation (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U162
Timeframe: 8h settlements.
Z-score = (funding_rate - mean_14d) / std_14d over past 42 settlements (14 days).
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

START = "2025-01-01"
END = "2025-10-01"

GRID = list(itertools.product([2.0, 2.5, 3.0], [480, 960], ["both", "short_only", "long_only"]))


def one(base: str):
    ff = f"{B}/funding/{base}USDT.parquet"
    kf = f"{B}/klines_1m/{base}USDT.parquet"
    if not os.path.exists(ff) or not os.path.exists(kf):
        return base, {}
        
    f_df = pd.read_parquet(ff, columns=["date", "funding_rate"]).set_index("date")
    f_df = f_df[(f_df.index >= "2024-12-01") & (f_df.index < END)].sort_index()
    if len(f_df) < 60:
        return base, {}
        
    # 14-day rolling mean and std (42 settlements of 8h)
    f_series = f_df["funding_rate"]
    roll_mean = f_series.rolling(42, min_periods=21).mean().shift(1) # shift 1 to prevent lookahead!
    roll_std = f_series.rolling(42, min_periods=21).std().shift(1)
    
    f_df["zscore"] = np.where(roll_std > 1e-6, (f_series - roll_mean) / roll_std, 0.0)
    
    # 5m klines
    k_df = pd.read_parquet(kf, columns=["date", "open", "high", "low", "close", "volume"])
    k_df = k_df[(k_df.date >= "2024-12-30") & (k_df.date < END)]
    if len(k_df) < 50000:
        return base, {}
        
    df5 = k_df.set_index("date").resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    df5 = df5.join(f_df[["zscore"]], how="left")
    df5 = df5[(df5.index >= START) & (df5.index < END)]
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    v = df5.volume.to_numpy()
    z = df5.zscore.to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    out = {}
    for z_thr, hold_m, arm in GRID:
        hold_bars = hold_m // 5
        
        # When z >= z_thr -> funding is abnormally positive -> fade by going SHORT
        # When z <= -z_thr -> funding is abnormally negative -> fade by going LONG
        s_cond = z >= z_thr
        l_cond = z <= -z_thr
        
        s_idx = np.where(s_cond)[0]
        l_idx = np.where(l_cond)[0]
        
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
            out[(z_thr, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
        out[(z_thr, hold_m, arm)] = tr
        
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
            "z_thr": cell[0], "hold_m": cell[1], "arm": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r34_train.csv", index=False)
