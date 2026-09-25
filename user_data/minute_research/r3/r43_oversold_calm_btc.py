"""R43 Altcoin Extreme Oversold Bounce with BTC Low-Vol Filter (TRAIN only).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U60 (excluding BTC)
Timeframe: 15m
Logic:
  BTC 1h volatility: ATR(14) / Close < 0.005 (0.5% calm macro).
  Altcoin: 15m RSI < 20 AND 15m Close < SMA20 - 3.0 * StdDev (extreme 3-sigma oversold).
  Entry: Long only.
  Exit: Time exit HOLD {60, 120, 240} min.
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

GRID = [60, 120, 240]


def calc_rsi(c: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    return 100.0 - (100.0 / (1.0 + rs))


def load_btc_calm():
    kf = f"{B}/klines_1m/BTCUSDT.parquet"
    df = pd.read_parquet(kf, columns=["date", "open", "high", "low", "close"])
    df = df[(df.date >= "2024-12-20") & (df.date < "2025-10-01")]
    df1h = df.set_index("date").resample("1h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    
    tr = np.maximum(df1h.high - df1h.low, np.maximum(np.abs(df1h.high - df1h.close.shift(1)), np.abs(df1h.low - df1h.close.shift(1))))
    atr = tr.ewm(span=14, adjust=False).mean()
    rel_atr = (atr / df1h.close).shift(1).fillna(1.0) # shift 1 to prevent lookahead!
    is_calm = rel_atr < 0.005
    return is_calm


def one(base: str, btc_calm: pd.Series):
    d = L.load(base)
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    df15 = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    df15["btc_calm"] = btc_calm.reindex(df15.index, method="ffill").fillna(False)
    
    c = df15.close.to_numpy()
    o = df15.open.to_numpy()
    h = df15.high.to_numpy()
    l = df15.low.to_numpy()
    v = df15.volume.to_numpy()
    calm = df15.btc_calm.to_numpy()
    
    rsi15 = calc_rsi(c, 14)
    sma20 = pd.Series(c).rolling(20, min_periods=20).mean().to_numpy()
    std20 = pd.Series(c).rolling(20, min_periods=20).std().to_numpy()
    bb_lo_3 = sma20 - 3.0 * std20
    
    cost = 7.5 if base == "ETH" else 10.0
    d15 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df15.index.to_series()
    }
    
    out = {}
    for hold_m in GRID:
        hold_bars = hold_m // 15
        long_cond = calm & (rsi15 < 20) & (c < bb_lo_3)
        idx = np.where(long_cond)[0]
        
        if len(idx) == 0:
            out[hold_m] = pd.DataFrame()
            continue
            
        side = np.ones(len(idx), dtype=np.int8)
        tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
        out[hold_m] = tr
        
    return base, out


_BTC_CALM = None

def _init_w(calm):
    global _BTC_CALM
    _BTC_CALM = calm

def _w(b):
    return one(b, _BTC_CALM)


if __name__ == "__main__":
    calm_series = load_btc_calm()
    res = {}
    with ProcessPoolExecutor(16, initializer=_init_w, initargs=(calm_series,)) as ex:
        for base, out in ex.map(_w, U60):
            if out:
                res[base] = out
                
    rows = []
    days = 273
    for hold_m in GRID:
        s = L.summarize({b: res[b][hold_m] for b in res if hold_m in res[b]})
        rows.append({
            "hold_m": hold_m,
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r43_train.csv", index=False)
