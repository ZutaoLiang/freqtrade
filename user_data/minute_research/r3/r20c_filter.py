"""R20c: Test SL / TP / Volume Confirmation on R20 OI Buildup Breakout (TRAIN)."""
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

# Grid: SL x TP x V_mult x Hold
# sl in {0.0, 0.025, 0.035} x tp in {0.0, 0.05} x v_mult in {1.0, 1.5} x hold in {120, 240} = 24 cells
GRID = list(itertools.product([0.0, 0.025, 0.035], [0.0, 0.05], [1.0, 1.5], [120, 240]))


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
    v = df5.volume.to_numpy()
    oi = df5.oi.to_numpy()
    
    r4h = pd.Series(c).pct_change(48).to_numpy()
    sig4h = pd.Series(r4h).shift(48).rolling(2016, min_periods=500).std().to_numpy()
    vol_sma = pd.Series(v).rolling(20, min_periods=10).mean().to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    out = {}
    W = 12
    OI_thr = 0.08
    oi_shift = pd.Series(oi).shift(W).to_numpy()
    oi_change = np.where(oi_shift > 0, oi / oi_shift - 1.0, 0.0)
    
    box_high = pd.Series(h).rolling(W, min_periods=W).max().shift(1).to_numpy()
    box_low = pd.Series(l).rolling(W, min_periods=W).min().shift(1).to_numpy()
    
    coiled = (oi_change >= OI_thr) & (np.abs(r4h) <= 0.5 * sig4h)
    recent_coiled = pd.Series(coiled).rolling(12, min_periods=1).max().to_numpy().astype(bool)
    
    for sl, tp, v_mult, hold_m in GRID:
        hold_bars = hold_m // 5
        v_ok = v >= v_mult * vol_sma
        
        long_sig = recent_coiled & (c > box_high) & (pd.Series(c).shift(1) <= pd.Series(box_high).shift(1)) & v_ok
        short_sig = recent_coiled & (c < box_low) & (pd.Series(c).shift(1) >= pd.Series(box_low).shift(1)) & v_ok
        
        l_idx = np.where(long_sig)[0]
        s_idx = np.where(short_sig)[0]
        
        idx = np.concatenate([l_idx, s_idx])
        side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
        
        if len(idx) == 0:
            out[(sl, tp, v_mult, hold_m)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d5, idx, side, hold=hold_bars, sl=sl, tp=tp, cost_bps=cost)
        out[(sl, tp, v_mult, hold_m)] = tr
        
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
            "sl": cell[0], "tp": cell[1], "v_mult": cell[2], "hold_m": cell[3],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r20c_filter_train.csv", index=False)
