"""R30 BTC High-Beta Momentum Follow - VALID & TRAIN Audit.

Frozen Parameters:
  Timeframe: 15m
  btc_thr = 0.012 (1.2% move in BTC within 15m)
  mult = 1.5 (Alt move >= 1.5 * btc_thr in the same direction)
  hold_m = 90 min (6 bars of 15m)
  Universe: U60 (excluding BTC)
  Cost: 7.5 bp for ETH, 10.0 bp for alts
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

B = "/root/freqtrade/user_data/data/binance_public"
U60 = [b for b in L.U60 if b != "BTC"]

SEGMENTS = {
    "TRAIN": ("2025-01-01", "2025-10-01"),
    "VALID": ("2025-12-01", "2026-03-01"),
}

BTC_THR = 0.012
MULT = 1.5
HOLD_M = 90


def load_btc_segment(seg: str):
    start, end = SEGMENTS[seg]
    # Warm up 2 days
    warm = (pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    kf = f"{B}/klines_1m/BTCUSDT.parquet"
    df = pd.read_parquet(kf, columns=["date", "open", "close"])
    df = df[(df.date >= warm) & (df.date < end)]
    df15 = df.set_index("date").resample("15min").agg({"open": "first", "close": "last"}).dropna()
    df15["ret"] = (df15["close"] / df15["open"]) - 1.0
    return df15


def one_pair(base: str, seg: str, btc15: pd.DataFrame):
    start, end = SEGMENTS[seg]
    warm = (pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    kf = f"{B}/klines_1m/{base}USDT.parquet"
    if not os.path.exists(kf):
        return base, pd.DataFrame()
        
    df = pd.read_parquet(kf, columns=["date", "open", "high", "low", "close", "volume"])
    df = df[(df.date >= warm) & (df.date < end)]
    if len(df) < 5000:
        return base, pd.DataFrame()
        
    df15 = df.set_index("date").resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    # Filter to segment window
    df15["btc_ret"] = btc15["ret"].reindex(df15.index).fillna(0.0)
    df15["alt_ret"] = (df15["close"] / df15["open"]) - 1.0
    
    # Mask to keep signals within segment
    seg_mask = (df15.index >= pd.Timestamp(start, tz="UTC")) & (df15.index < pd.Timestamp(end, tz="UTC"))
    
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
    
    hold_bars = HOLD_M // 15
    long_cond = (btc_r >= BTC_THR) & (alt_r >= MULT * BTC_THR) & seg_mask
    short_cond = (btc_r <= -BTC_THR) & (alt_r <= -MULT * BTC_THR) & seg_mask
    
    l_idx = np.where(long_cond)[0]
    s_idx = np.where(short_cond)[0]
    idx = np.concatenate([l_idx, s_idx])
    side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
    
    if len(idx) == 0:
        return base, pd.DataFrame()
        
    order = np.argsort(idx)
    idx = idx[order]
    side = side[order]
    
    tr = L.H.run(d15, idx, side, hold=hold_bars, cost_bps=cost)
    if len(tr) > 0:
        tr["base"] = base
    return base, tr


def _worker_tuple(args):
    base, seg, btc15 = args
    return one_pair(base, seg, btc15)


def audit_segment(seg: str):
    btc15 = load_btc_segment(seg)
    trades = {}
    args = [(b, seg, btc15) for b in U60]
    with ProcessPoolExecutor(16) as ex:
        for base, tr in ex.map(_worker_tuple, args):
            if len(tr) > 0:
                trades[base] = tr
                
    s = L.summarize(trades)
    all_tr = pd.concat(trades.values(), ignore_index=True) if trades else pd.DataFrame()
    return s, all_tr


if __name__ == "__main__":
    for seg in ["TRAIN", "VALID"]:
        s, tr = audit_segment(seg)
        print(f"================ {seg} SUMMARY ================")
        for k, v in s.items():
            print(f"  {k}: {v}")
        if len(tr) > 0:
            tr["day"] = pd.to_datetime(tr.t).dt.floor("D")
            tr["month"] = pd.to_datetime(tr.t).dt.to_period("M")
            
            # Monthly breakdown
            m_stats = tr.groupby("month").agg(
                n=("ret", "count"),
                mean_bp=("ret", lambda x: np.mean(x) * 10000),
                sum_bp=("ret", lambda x: np.sum(x) * 10000),
                win=("ret", lambda x: np.mean(x > 0))
            )
            print("\n  Monthly Breakdown:")
            print(m_stats)
            
            # Concentration checks (Criterion E)
            coin_pnl = tr.groupby("base")["ret"].sum()
            tot_pnl = tr["ret"].sum()
            max_coin_share = coin_pnl.max() / tot_pnl if tot_pnl > 0 else 0
            top_coin = coin_pnl.idxmax()
            
            day_pnl = tr.groupby("day")["ret"].sum()
            max_day_share = day_pnl.max() / tot_pnl if tot_pnl > 0 else 0
            top_day = day_pnl.idxmax().date()
            
            print(f"\n  Criterion E Concentration:")
            print(f"    Max single coin share: {max_coin_share*100:.1f}% ({top_coin}) [Limit: <= 30%]")
            print(f"    Max single day share:  {max_day_share*100:.1f}% ({top_day}) [Limit: <= 20%]")
            
            # Cost stress test (Criterion F)
            # Add extra 10 bps round trip (15 bp fee instead of 10 bp)
            stressed_ret = tr["ret"] - 0.0010
            pos = stressed_ret[stressed_ret > 0].sum()
            neg = -stressed_ret[stressed_ret < 0].sum()
            stressed_pf = pos / neg if neg > 0 else 999.0
            print(f"\n  Criterion F Cost Stress (fee x 1.5):")
            print(f"    Stressed mean: {stressed_ret.mean()*10000:.2f} bp")
            print(f"    Stressed PF:   {stressed_pf:.3f} [Must be > 1.0]")
