"""Inspect Long vs Short and expand to U162 for R20 OI Breakout."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

B = "/root/freqtrade/user_data/data/binance_public"
U = pd.read_csv("universe_rank.csv")
U162 = U[U.med_qv >= 1e7].base.tolist()

# Let's test W=12, OI_thr in {0.06, 0.08}, hold_m in {120, 240, 480}
# Separating Long and Short arms!
GRID = []
for oi in [0.06, 0.08]:
    for hold in [120, 240, 480]:
        for arm in ["both", "long", "short"]:
            GRID.append((12, oi, hold, arm))


def one(base: str):
    mf = f"{B}/metrics/{base}USDT.parquet"
    kf = f"{B}/klines_1m/{base}USDT.parquet"
    if not os.path.exists(mf) or not os.path.exists(kf):
        return base, {}
    d = pd.read_parquet(kf, columns=["date", "open", "high", "low", "close", "volume"])
    d = d[(d.date >= "2025-01-01") & (d.date < "2025-10-01")]
    if len(d) < 50000:
        return base, {}
        
    df5 = d.set_index("date").resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
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
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": df5.volume.to_numpy(),
        "date": df5.index.to_series()
    }
    
    out = {}
    W = 12
    oi_shift = pd.Series(oi).shift(W).to_numpy()
    oi_change = np.where(oi_shift > 0, oi / oi_shift - 1.0, 0.0)
    
    box_high = pd.Series(h).rolling(W, min_periods=W).max().shift(1).to_numpy()
    box_low = pd.Series(l).rolling(W, min_periods=W).min().shift(1).to_numpy()
    
    for _, OI_thr, hold_m, arm in GRID:
        hold_bars = hold_m // 5
        coiled = (oi_change >= OI_thr) & (np.abs(r4h) <= 0.5 * sig4h)
        recent_coiled = pd.Series(coiled).rolling(12, min_periods=1).max().to_numpy().astype(bool)
        
        long_sig = recent_coiled & (c > box_high) & (pd.Series(c).shift(1) <= pd.Series(box_high).shift(1))
        short_sig = recent_coiled & (c < box_low) & (pd.Series(c).shift(1) >= pd.Series(box_low).shift(1))
        
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
            out[(W, OI_thr, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
        out[(W, OI_thr, hold_m, arm)] = tr
        
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
            "OI_thr": cell[1], "hold_m": cell[2], "arm": cell[3],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r20b_u162_train.csv", index=False)
