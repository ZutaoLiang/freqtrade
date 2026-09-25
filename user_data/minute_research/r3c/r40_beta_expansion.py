"""R40 Altcoin High-Beta Expansion with 1h BTC Lead (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60 (excluding BTC)
Timeframe: 15m
Logic:
  Calculate 24h rolling beta of each altcoin to BTC.
  BTC 1h trend:
    Bull: BTC 1h Close > EMA20 > EMA50 (shifted 1 bar).
    Bear: BTC 1h Close < EMA20 < EMA50 (shifted 1 bar).
  In Bull: Long top-beta alts (beta >= beta_thr) on 15m Donchian high breakout.
  In Bear: Short top-beta alts (beta >= beta_thr) on 15m Donchian low breakdown.
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

START = "2025-01-01"
END = "2025-10-01"

GRID = list(itertools.product([1.5, 2.0], [60, 120, 240]))


def load_btc_1h_and_15m():
    kf = f"{B}/klines_1m/BTCUSDT.parquet"
    df = pd.read_parquet(kf, columns=["date", "open", "close"])
    df = df[(df.date >= "2024-12-15") & (df.date < END)]
    
    # 1h trend
    df1h = df.set_index("date").resample("1h").last().dropna()
    df1h["ema20"] = df1h["close"].ewm(span=20, adjust=False).mean()
    df1h["ema50"] = df1h["close"].ewm(span=50, adjust=False).mean()
    bull = ((df1h["close"] > df1h["ema20"]) & (df1h["ema20"] > df1h["ema50"])).shift(1).fillna(False)
    bear = ((df1h["close"] < df1h["ema20"]) & (df1h["ema20"] < df1h["ema50"])).shift(1).fillna(False)
    
    # 15m returns for beta calculation
    df15 = df.set_index("date").resample("15min").agg({"open": "first", "close": "last"}).dropna()
    df15["ret"] = (df15["close"] / df15["open"]) - 1.0
    
    df15["bull"] = bull.reindex(df15.index, method="ffill").fillna(False)
    df15["bear"] = bear.reindex(df15.index, method="ffill").fillna(False)
    
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
    
    df15["btc_ret"] = btc15["ret"].reindex(df15.index).fillna(0.0)
    df15["btc_bull"] = btc15["bull"].reindex(df15.index).fillna(False)
    df15["btc_bear"] = btc15["bear"].reindex(df15.index).fillna(False)
    df15["alt_ret"] = (df15["close"] / df15["open"]) - 1.0
    
    # 24h rolling beta (96 bars of 15m)
    cov = df15["alt_ret"].rolling(96, min_periods=48).cov(df15["btc_ret"]).shift(1)
    var = df15["btc_ret"].rolling(96, min_periods=48).var().shift(1)
    df15["beta"] = np.where(var > 1e-8, cov / var, 1.0)
    
    c = df15.close.to_numpy()
    o = df15.open.to_numpy()
    h = df15.high.to_numpy()
    l = df15.low.to_numpy()
    v = df15.volume.to_numpy()
    beta = df15.beta.to_numpy()
    btc_bull = df15.btc_bull.to_numpy()
    btc_bear = df15.btc_bear.to_numpy()
    
    hh12 = pd.Series(h).rolling(12, min_periods=12).max().shift(1).to_numpy()
    ll12 = pd.Series(l).rolling(12, min_periods=12).min().shift(1).to_numpy()
    
    cost = 7.5 if base == "ETH" else 10.0
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for beta_thr, hold_m in GRID:
        hold_bars = hold_m // 15
        
        long_cond = btc_bull & (beta >= beta_thr) & (c > hh12)
        short_cond = btc_bear & (beta >= beta_thr) & (c < ll12)
        
        l_idx = np.where(long_cond)[0]
        s_idx = np.where(short_cond)[0]
        idx = np.concatenate([l_idx, s_idx])
        side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
        
        if len(idx) == 0:
            out[(beta_thr, hold_m)] = pd.DataFrame()
            continue
            
        order = np.argsort(idx)
        idx = idx[order]
        side = side[order]
        
        tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[(beta_thr, hold_m)] = tr
        
    return base, out


_BTC_15 = None

def _init_w(b15):
    global _BTC_15
    _BTC_15 = b15

def _w(b):
    return one(b, _BTC_15)


if __name__ == "__main__":
    b15 = load_btc_15m_and_1h = load_btc_1h_and_15m()
    res = {}
    with ProcessPoolExecutor(16, initializer=_init_w, initargs=(b15,)) as ex:
        for base, out in ex.map(_w, U60):
            if out:
                res[base] = out
                
    rows = []
    days = 273
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res if cell in res[b]})
        rows.append({
            "beta_thr": cell[0], "hold_m": cell[1],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r40_train.csv", index=False)
