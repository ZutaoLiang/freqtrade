"""R39 Funding Rate Payer Momentum with T+1m Execution (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U162
Data: 1m klines + funding rates.
Execution:
  At settlement T (00:00, 08:00, 16:00 UTC), |FR(T)| >= F_thr.
  Consecutive settlements with same sign >= 2.
  Entry at T + 1 min open (00:01, 08:01, 16:01).
  Side: Payer side (Long if FR > 0, Short if FR < 0).
  Exit: Time exit 235 min (before next settlement, zero funding fee paid).
  Cost: 10 bp per side (7.5 bp for BTC/ETH).
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

# Grid: F_thr in {0.003, 0.004, 0.005} x min_streak in {1, 2} x hold_m in {120, 235}
GRID = list(itertools.product([0.003, 0.004, 0.005], [1, 2], [120, 235]))


def one(base: str):
    ff = f"{B}/funding/{base}USDT.parquet"
    kf = f"{B}/klines_1m/{base}USDT.parquet"
    if not os.path.exists(ff) or not os.path.exists(kf):
        return base, {}
        
    f_df = pd.read_parquet(ff, columns=["date", "funding_rate"]).set_index("date")
    f_df = f_df[(f_df.index >= "2024-12-01") & (f_df.index < END)].sort_index()
    if len(f_df) < 20:
        return base, {}
        
    f_series = f_df["funding_rate"]
    
    # 1m klines
    k_df = pd.read_parquet(kf, columns=["date", "open", "high", "low", "close", "volume"])
    k_df = k_df[(k_df.date >= START) & (k_df.date < END)].reset_index(drop=True)
    if len(k_df) < 50000:
        return base, {}
        
    # Map each settlement to the exact 1m bar index of T
    k_df["idx"] = np.arange(len(k_df))
    # Align dates
    settle_times = f_series.index
    
    # Join on settlement timestamps
    k_indexed = k_df.set_index("date")
    
    c = k_df.close.to_numpy()
    o = k_df.open.to_numpy()
    h = k_df.high.to_numpy()
    l = k_df.low.to_numpy()
    v = k_df.volume.to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d1 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": k_df["date"]
    }
    
    out = {}
    
    for f_thr, min_streak, hold_m in GRID:
        # Determine streaks
        signs = np.sign(f_series)
        is_extreme = f_series.abs() >= f_thr
        valid_signs = np.where(is_extreme, signs, 0)
        
        # Streak calculation
        streaks = []
        cur_streak = 0
        cur_sign = 0
        for s in valid_signs:
            if s != 0 and s == cur_sign:
                cur_streak += 1
            elif s != 0:
                cur_streak = 1
                cur_sign = s
            else:
                cur_streak = 0
                cur_sign = 0
            streaks.append((cur_streak, cur_sign))
            
        # Find matching settlements
        sig_indices = []
        sig_sides = []
        for t, (strk, sgn) in zip(settle_times, streaks):
            if strk >= min_streak and sgn != 0:
                # T is the settlement minute
                if t in k_indexed.index:
                    # In harness, entry is at bar i + 1
                    # So if signal is on bar at T (i.e. minute 00), entry will be at T+1m open (minute 01)!
                    i = k_indexed.loc[t, "idx"]
                    sig_indices.append(i)
                    sig_sides.append(int(sgn)) # Payer side: +1 for long if FR > 0, -1 for short if FR < 0
                    
        if len(sig_indices) == 0:
            out[(f_thr, min_streak, hold_m)] = pd.DataFrame()
            continue
            
        sig_idx = np.array(sig_indices, dtype=np.int64)
        sig_side = np.array(sig_sides, dtype=np.int8)
        
        order = np.argsort(sig_idx)
        sig_idx = sig_idx[order]
        sig_side = sig_side[order]
        
        tr = L.H.run(d1, sig_idx, sig_side, hold=hold_m, cost_bps=cost)
        out[(f_thr, min_streak, hold_m)] = tr
        
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
            "f_thr": cell[0], "min_streak": cell[1], "hold_m": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r39_train.csv", index=False)
