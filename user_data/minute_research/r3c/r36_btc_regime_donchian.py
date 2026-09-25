"""R36 Macro BTC Regime-Gated Intraday Donchian Breakout (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60 (excluding BTC)
Timeframe: 15m with 4h BTC informative.
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

# Grid: LB in {48, 96} bars x Vol_mult in {1.5, 2.0} x HOLD in {120, 240, 480} min
GRID = list(itertools.product([48, 96], [1.5, 2.0], [120, 240, 480]))


def load_btc_regime():
    kf = f"{B}/klines_1m/BTCUSDT.parquet"
    df = pd.read_parquet(kf, columns=["date", "close"])
    df = df[(df.date >= "2024-11-01") & (df.date < "2025-10-01")]
    df4h = df.set_index("date").resample("4h").last().dropna()
    df4h["ema50"] = df4h["close"].ewm(span=50, adjust=False).mean()
    df4h["ema200"] = df4h["close"].ewm(span=200, adjust=False).mean()
    
    # Must shift by 1 to prevent lookahead!
    bull = ((df4h["close"] > df4h["ema200"]) & (df4h["ema50"] > df4h["ema200"])).shift(1).fillna(False)
    bear = ((df4h["close"] < df4h["ema200"]) & (df4h["ema50"] < df4h["ema200"])).shift(1).fillna(False)
    
    regime = pd.DataFrame({"bull": bull, "bear": bear})
    return regime


def one_pair(base: str, btc_reg: pd.DataFrame):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df15 = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    # Reindex BTC regime forward-filled
    df15["btc_bull"] = btc_reg["bull"].reindex(df15.index, method="ffill").fillna(False)
    df15["btc_bear"] = btc_reg["bear"].reindex(df15.index, method="ffill").fillna(False)
    
    c = df15.close.to_numpy()
    o = df15.open.to_numpy()
    h = df15.high.to_numpy()
    l = df15.low.to_numpy()
    v = df15.volume.to_numpy()
    btc_bull = df15.btc_bull.to_numpy()
    btc_bear = df15.btc_bear.to_numpy()
    
    v_sma = pd.Series(v).rolling(20, min_periods=20).mean().to_numpy()
    
    cost = 7.5 if base == "ETH" else 10.0
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for lb, v_mult, hold_m in GRID:
        hold_bars = hold_m // 15
        
        hh = pd.Series(h).rolling(lb, min_periods=lb).max().shift(1).to_numpy()
        ll = pd.Series(l).rolling(lb, min_periods=lb).min().shift(1).to_numpy()
        
        long_cond = btc_bull & (c > hh) & (v >= v_mult * v_sma)
        short_cond = btc_bear & (c < ll) & (v >= v_mult * v_sma)
        
        l_idx = np.where(long_cond)[0]
        s_idx = np.where(short_cond)[0]
        idx = np.concatenate([l_idx, s_idx])
        side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
        
        if len(idx) == 0:
            out[(lb, v_mult, hold_m)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[(lb, v_mult, hold_m)] = tr
        
    return base, out


_BTC_REG = None

def _init_w(reg):
    global _BTC_REG
    _BTC_REG = reg

def _w(b):
    return one_pair(b, _BTC_REG)


if __name__ == "__main__":
    reg = load_btc_regime()
    res = {}
    with ProcessPoolExecutor(16, initializer=_init_w, initargs=(reg,)) as ex:
        for base, out in ex.map(_w, U60):
            if out:
                res[base] = out
                
    rows = []
    days = 273
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res if cell in res[b]})
        rows.append({
            "lb": cell[0], "v_mult": cell[1], "hold_m": cell[2],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r36_train.csv", index=False)
