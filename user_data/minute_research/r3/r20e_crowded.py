"""R20e: Crowded Long Filter (fr >= 0.0001 and btc not pumping) on R20 Short Cascade."""
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

# Let's load BTC 4h return
def get_btc_filter():
    d = L.load("BTC")
    df = pd.DataFrame({"close": d["close"]}, index=pd.DatetimeIndex(d["date"]))
    df5 = df.resample("5min").agg({"close": "last"}).dropna()
    # 4h return of BTC
    r4h = df5.close.pct_change(48).shift(1) # shift 1 to prevent lookahead
    return r4h

BTC_R4H = get_btc_filter()

GRID = [
    # W, OI_thr, hold_m, fr_min, btc_max
    (24, 0.08, 240, 0.0000, 0.02),
    (24, 0.08, 240, 0.0001, 0.02),
    (24, 0.08, 240, 0.0001, 0.01),
    (24, 0.08, 360, 0.0001, 0.02),
    (24, 0.08, 360, 0.0001, 0.01),
    (12, 0.08, 240, 0.0000, 0.02),
    (12, 0.08, 240, 0.0001, 0.02),
    (12, 0.08, 240, 0.0001, 0.01),
    (12, 0.06, 240, 0.0001, 0.01),
]


def one(base: str):
    mf = f"{B}/metrics/{base}USDT.parquet"
    kf = f"{B}/klines_1m/{base}USDT.parquet"
    ff = f"{B}/funding/{base}USDT.parquet"
    if not os.path.exists(mf) or not os.path.exists(kf) or not os.path.exists(ff):
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
    
    # Funding rate
    f_df = pd.read_parquet(ff).set_index("date")[["funding_rate"]]
    df5 = df5.join(f_df, how="left")
    df5["fr"] = df5.funding_rate.ffill(limit=24) # ffill across 2 hours
    
    # Join BTC 4h
    df5["btc_r4h"] = BTC_R4H.reindex(df5.index).ffill()
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    oi = df5.oi.to_numpy()
    fr = df5.fr.to_numpy()
    btc_r = df5.btc_r4h.to_numpy()
    
    r4h = pd.Series(c).pct_change(48).to_numpy()
    sig4h = pd.Series(r4h).shift(48).rolling(2016, min_periods=500).std().to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": df5.volume.to_numpy(),
        "date": df5.index.to_series()
    }
    
    out = {}
    for W, OI_thr, hold_m, fr_min, btc_max in GRID:
        hold_bars = hold_m // 5
        oi_shift = pd.Series(oi).shift(W).to_numpy()
        oi_change = np.where(oi_shift > 0, oi / oi_shift - 1.0, 0.0)
        box_low = pd.Series(l).rolling(W, min_periods=W).min().shift(1).to_numpy()
        
        coiled = (oi_change >= OI_thr) & (np.abs(r4h) <= 0.5 * sig4h)
        recent_coiled = pd.Series(coiled).rolling(12, min_periods=1).max().to_numpy().astype(bool)
        
        # Crowded filter
        crowded = (fr >= fr_min) & (btc_r <= btc_max)
        
        short_sig = recent_coiled & (c < box_low) & (pd.Series(c).shift(1) >= pd.Series(box_low).shift(1)) & crowded
        idx = np.where(short_sig)[0]
        
        if len(idx) == 0:
            out[(W, OI_thr, hold_m, fr_min, btc_max)] = pd.DataFrame()
            continue
            
        side = -np.ones(len(idx), dtype=np.int8)
        tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
        out[(W, OI_thr, hold_m, fr_min, btc_max)] = tr
        
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
            "W": cell[0], "OI_thr": cell[1], "hold_m": cell[2], "fr_min": cell[3], "btc_max": cell[4],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r20e_train.csv", index=False)
