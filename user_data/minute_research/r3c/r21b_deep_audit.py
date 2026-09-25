"""Deep Audit of R21 EMA Offset Dip Buyer: Market Hedge & VALID test."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

START_VALID = "2025-12-01"
END_VALID = "2026-03-01"


def calc_rsi(c: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    return 100.0 - (100.0 / (1.0 + rs))


def one_pair(base: str, seg: tuple[str, str]):
    d = L.load(base, end=seg[1])
    df = pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "volume": d["volume"]
    }, index=pd.DatetimeIndex(d["date"]))
    
    # Filter with 7 days warm up
    warm = (pd.Timestamp(seg[0], tz="UTC") - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    df = df[df.index >= warm]
    
    df5 = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    v = df5.volume.to_numpy()
    
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    vol_sma = pd.Series(v).rolling(20, min_periods=10).mean().to_numpy()
    rsi14 = calc_rsi(c, 14)
    
    cost = L.cost_bps(base)
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    offset = 0.035
    rsi_thr = 25
    hold_bars = 120 // 5
    
    long_cond = (c < ema50 * (1.0 - offset)) & (rsi14 < rsi_thr) & (v > 1.2 * vol_sma)
    in_seg = (df5.index >= seg[0]) & (df5.index < seg[1])
    long_cond = long_cond & np.asarray(in_seg, dtype=bool)
    
    idx = np.where(long_cond)[0]
    if len(idx) == 0:
        return base, pd.DataFrame()
        
    side = np.ones(len(idx), dtype=np.int8)
    tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
    return base, tr


if __name__ == "__main__":
    # 1. Check VALID
    res_valid = {}
    with ProcessPoolExecutor(16) as ex:
        futures = [ex.submit(one_pair, b, (START_VALID, END_VALID)) for b in L.U60]
        for f in futures:
            b, tr = f.result()
            if len(tr): res_valid[b] = tr
            
    s_val = L.summarize(res_valid)
    days_val = 151
    print("=== VALID (2025-10 .. 2026-03) ===")
    print("Trades:", s_val.get("n", 0), "Per day:", s_val.get("n", 0) / days_val)
    print("Net Mean (bp):", s_val.get("mean_bp"), "Median (bp):", s_val.get("med_bp"))
    print("Win Rate:", s_val.get("win"), "Profit Factor:", s_val.get("pf"))
    print("Trading Days:", s_val.get("days"), "Day Mean (bp):", s_val.get("day_mean_bp"), "Day t-stat:", s_val.get("day_t"))
    print("Days Pos Fraction:", s_val.get("days_pos"))
    
    all_val = pd.concat(list(res_valid.values())).reset_index(drop=True)
    all_val["month"] = pd.DatetimeIndex(all_val.t).to_period("M")
    m_val = all_val.groupby("month").apply(lambda g: pd.Series({
        "n": len(g), "mean_bp": g.ret.mean() * 1e4, "pf": g.ret[g.ret > 0].sum() / -g.ret[g.ret < 0].sum()
    }))
    print("\nVALID Monthly:")
    print(m_val.to_string())
