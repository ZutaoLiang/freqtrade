"""R24 Multi-Settlement Funding Exhaustion Short - Audit on TRAIN and VALID.

Frozen parameters from TRAIN:
  F_thr = 0.0003 (0.03% per 8h settlement, 3 consecutive settlements >= 0.03%)
  Hold = 480 min (8 hours)
  Arm = short_only
  Universe: U162
  Cost: 10 bp per side (7.5 for BTC/ETH)
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

B = "/root/freqtrade/user_data/data/binance_public"
U = pd.read_csv("/root/freqtrade/user_data/minute_research/r3/universe_rank.csv")
U162 = U[U.med_qv >= 1e7].base.tolist()

SEGMENTS = {
    "TRAIN": ("2025-01-01", "2025-10-01"),
    "VALID": ("2025-12-01", "2026-03-01"),
}


def one_pair(base: str, seg: str):
    start, end = SEGMENTS[seg]
    ff = f"{B}/funding/{base}USDT.parquet"
    kf = f"{B}/klines_1m/{base}USDT.parquet"
    if not os.path.exists(ff) or not os.path.exists(kf):
        return base, pd.DataFrame()
        
    d = pd.read_parquet(kf, columns=["date", "open", "high", "low", "close", "volume"])
    d = d[(d.date >= start) & (d.date < end)]
    if len(d) < 10000:
        return base, pd.DataFrame()
        
    df5 = d.set_index("date").resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    f_df = pd.read_parquet(ff).set_index("date")[["funding_rate"]]
    f_series = f_df["funding_rate"].sort_index()
    f_df["f0"] = f_series
    f_df["f1"] = f_series.shift(1)
    f_df["f2"] = f_series.shift(2)
    
    df5 = df5.join(f_df, how="left")
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    v = df5.volume.to_numpy()
    f0 = df5.f0.to_numpy()
    f1 = df5.f1.to_numpy()
    f2 = df5.f2.to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    hold_bars = 480 // 5 # 96 bars
    F_thr = 0.0003
    
    # 3 consecutive settlements >= F_thr -> short
    long_exhaust = (f0 >= F_thr) & (f1 >= F_thr) & (f2 >= F_thr)
    idx = np.where(long_exhaust)[0]
    if len(idx) == 0:
        return base, pd.DataFrame()
        
    side = -np.ones(len(idx), dtype=np.int8)
    tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
    if len(tr) > 0:
        tr["base"] = base
    return base, tr


def worker(args):
    base, seg = args
    return one_pair(base, seg)


def audit_segment(seg: str):
    trades = {}
    args = [(b, seg) for b in U162]
    with ProcessPoolExecutor(16) as ex:
        for base, tr in ex.map(worker, args):
            if len(tr) > 0:
                trades[base] = tr
                
    s = L.summarize(trades)
    all_tr = pd.concat(trades.values(), ignore_index=True) if trades else pd.DataFrame()
    return s, all_tr


if __name__ == "__main__":
    for seg in ["TRAIN", "VALID"]:
        s, tr = audit_segment(seg)
        print(f"=== {seg} SUMMARY ===")
        for k, v in s.items():
            print(f"  {k}: {v}")
        if len(tr) > 0:
            print(f"  First 3 trades:\n{tr[['base', 't', 'ret', 'side']].head(3)}")
            print(f"  Monthly Breakdown in {seg}:")
            tr["month"] = pd.to_datetime(tr.t).dt.to_period("M")
            m_stats = tr.groupby("month").agg(
                n=("ret", "count"),
                mean_bp=("ret", lambda x: np.mean(x) * 10000),
                sum_bp=("ret", lambda x: np.sum(x) * 10000),
                win=("ret", lambda x: np.mean(x > 0))
            )
            print(m_stats)
