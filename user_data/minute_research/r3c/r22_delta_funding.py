"""R22 Funding Rate Acceleration / Delta Funding Shock (TRAIN only).

Pre-registration:
  Funding Rate at settlement T: fr(T).
  Delta Funding: delta_fr = fr(T) - fr(T-1).
  Signal at T (entered at T+5m open):
    Follow arm: Long if delta_fr >= +D_thr; Short if delta_fr <= -D_thr.
    Fade arm: Short if delta_fr >= +D_thr; Long if delta_fr <= -D_thr.
  Exit: Time exit HOLD minutes (60, 240, 480).
  Grid: D_thr in {0.0002, 0.0004} x HOLD in {60, 240, 480} x Arm in {follow, fade} = 12 cells.
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

GRID = list(itertools.product([0.0002, 0.0004], [60, 240, 480], ["follow", "fade"]))


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
    df5 = df5.join(f_df, how="left")
    
    # Delta funding when funding is published
    is_settle = df5["funding_rate"].notna()
    fr = df5["funding_rate"].dropna()
    d_fr = fr.diff()
    
    df5["d_fr"] = d_fr
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    v = df5.volume.to_numpy()
    delta_funding = df5.d_fr.to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    out = {}
    for D_thr, hold_m, arm in GRID:
        hold_bars = hold_m // 5
        
        if arm == "follow":
            long_sig = delta_funding >= D_thr
            short_sig = delta_funding <= -D_thr
        else:
            long_sig = delta_funding <= -D_thr
            short_sig = delta_funding >= D_thr
            
        l_idx = np.where(long_sig)[0]
        s_idx = np.where(short_sig)[0]
        
        all_idx = np.concatenate([l_idx, s_idx])
        all_side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
        
        if len(all_idx) == 0:
            out[(D_thr, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(all_idx)
        all_idx = all_idx[order]
        all_side = all_side[order]
        
        tr = L.H.run(d5, all_idx, all_side, hold=hold_bars, cost_bps=cost)
        out[(D_thr, hold_m, arm)] = tr
        
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
            "D_thr": cell[0], "hold_m": cell[1], "arm": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r22_train.csv", index=False)
