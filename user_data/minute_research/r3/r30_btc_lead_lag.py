"""R30 BTC Momentum Lead-Lag Altcoin Follow (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60
Timeframe: 15m

Hypothesis:
  When BTC moves strongly in 15m (|ret_btc| >= 1.0%):
  Arm 'catchup': Trade the laggards (|ret_alt| <= 0.3%) in the direction of BTC.
  Arm 'momentum': Trade the leaders (|ret_alt| >= 1.5%) in the direction of BTC.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

B = "/root/freqtrade/user_data/data/binance_public"
U60 = [b for b in L.U60 if b != "BTC"]

GRID = list(itertools.product([0.008, 0.012], [30, 60, 120], ["catchup", "momentum"]))


def load_btc_15m():
    kf = f"{B}/klines_1m/BTCUSDT.parquet"
    df = pd.read_parquet(kf, columns=["date", "open", "close"])
    df = df[(df.date >= "2024-12-30") & (df.date < "2025-10-01")]
    df15 = df.set_index("date").resample("15min").agg({"open": "first", "close": "last"}).dropna()
    df15["ret"] = (df15["close"] / df15["open"]) - 1.0
    return df15


def one(base: str, btc15: pd.DataFrame):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df15 = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    # Join BTC return
    df15["btc_ret"] = btc15["ret"].reindex(df15.index).fillna(0.0)
    df15["alt_ret"] = (df15["close"] / df15["open"]) - 1.0
    
    c = df15.close.to_numpy()
    o = df15.open.to_numpy()
    h = df15.high.to_numpy()
    l = df15.low.to_numpy()
    v = df15.volume.to_numpy()
    btc_r = df15.btc_ret.to_numpy()
    alt_r = df15.alt_ret.to_numpy()
    
    cost = 7.5 if base == "ETH" else 10.0
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for btc_thr, hold_m, arm in GRID:
        hold_bars = hold_m // 15
        
        if arm == "catchup":
            # BTC up strong, alt has barely moved -> long alt
            long_cond = (btc_r >= btc_thr) & (alt_r <= 0.003) & (alt_r >= -0.003)
            # BTC down strong, alt has barely moved -> short alt
            short_cond = (btc_r <= -btc_thr) & (alt_r <= 0.003) & (alt_r >= -0.003)
        else: # momentum
            # BTC up strong, alt is already pumping hard -> follow momentum
            long_cond = (btc_r >= btc_thr) & (alt_r >= 1.5 * btc_thr)
            # BTC down strong, alt is dumping hard -> follow short
            short_cond = (btc_r <= -btc_thr) & (alt_r <= -1.5 * btc_thr)
            
        l_idx = np.where(long_cond)[0]
        s_idx = np.where(short_cond)[0]
        
        idx = np.concatenate([l_idx, s_idx])
        side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
        
        if len(idx) == 0:
            out[(btc_thr, hold_m, arm)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[(btc_thr, hold_m, arm)] = tr
        
    return base, out


_BTC15 = None

def _init_worker(b15):
    global _BTC15
    _BTC15 = b15

def _worker(base):
    return one(base, _BTC15)

if __name__ == "__main__":
    btc15 = load_btc_15m()
    res = {}
    with ProcessPoolExecutor(16, initializer=_init_worker, initargs=(btc15,)) as ex:
        for base, out in ex.map(_worker, U60):
            if out:
                res[base] = out
                
    rows = []
    days = 273
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res if cell in res[b]})
        rows.append({
            "btc_thr": cell[0], "hold_m": cell[1], "arm": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r30_train.csv", index=False)
