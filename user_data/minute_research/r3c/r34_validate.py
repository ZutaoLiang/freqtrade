"""R34 Funding Rate Z-score Dislocation - Audit on VALID and TRAIN.

Frozen Parameters:
  z_thr = 2.5
  hold_m = 960 min (16 hours)
  arm = short_only
  Universe: U162
  Cost: 10 bp/side (7.5 for BTC/ETH)
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

B = "/root/freqtrade/user_data/data/binance_public"
U = pd.read_csv("/root/freqtrade/user_data/minute_research/r3c/universe_rank.csv")
U162 = U[U.med_qv >= 1e7].base.tolist()

SEGMENTS = {
    "TRAIN": ("2025-01-01", "2025-10-01"),
    "VALID": ("2025-12-01", "2026-03-01"),
}

Z_THR = 2.5
HOLD_M = 960


def one_pair(base: str, seg: str):
    start, end = SEGMENTS[seg]
    # Warm up 30 days of funding
    warm_f = (pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    
    ff = f"{B}/funding/{base}USDT.parquet"
    kf = f"{B}/klines_1m/{base}USDT.parquet"
    if not os.path.exists(ff) or not os.path.exists(kf):
        return base, pd.DataFrame()
        
    f_df = pd.read_parquet(ff, columns=["date", "funding_rate"]).set_index("date")
    f_df = f_df[(f_df.index >= warm_f) & (f_df.index < end)].sort_index()
    if len(f_df) < 50:
        return base, pd.DataFrame()
        
    f_series = f_df["funding_rate"]
    roll_mean = f_series.rolling(42, min_periods=21).mean().shift(1)
    roll_std = f_series.rolling(42, min_periods=21).std().shift(1)
    f_df["zscore"] = np.where(roll_std > 1e-6, (f_series - roll_mean) / roll_std, 0.0)
    
    # 5m klines
    k_df = pd.read_parquet(kf, columns=["date", "open", "high", "low", "close", "volume"])
    k_df = k_df[(k_df.date >= start) & (k_df.date < end)]
    if len(k_df) < 10000:
        return base, pd.DataFrame()
        
    df5 = k_df.set_index("date").resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    df5 = df5.join(f_df[["zscore"]], how="left")
    
    c = df5.close.to_numpy()
    o = df5.open.to_numpy()
    h = df5.high.to_numpy()
    l = df5.low.to_numpy()
    v = df5.volume.to_numpy()
    z = df5.zscore.to_numpy()
    
    cost = 7.5 if base in ("BTC", "ETH") else 10.0
    d5 = {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "date": df5.index.to_series()
    }
    
    hold_bars = HOLD_M // 5
    s_cond = z >= Z_THR
    idx = np.where(s_cond)[0]
    if len(idx) == 0:
        return base, pd.DataFrame()
        
    side = -np.ones(len(idx), dtype=np.int8)
    tr = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
    if len(tr) > 0:
        tr["base"] = base
    return base, tr


def _worker(args):
    return one_pair(args[0], args[1])


def audit_segment(seg: str):
    trades = {}
    args = [(b, seg) for b in U162]
    with ProcessPoolExecutor(16) as ex:
        for base, tr in ex.map(_worker, args):
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
            
            stressed_ret = tr["ret"] - 0.0010
            pos = stressed_ret[stressed_ret > 0].sum()
            neg = -stressed_ret[stressed_ret < 0].sum()
            stressed_pf = pos / neg if neg > 0 else 999.0
            print(f"\n  Criterion F Cost Stress (fee x 1.5):")
            print(f"    Stressed mean: {stressed_ret.mean()*10000:.2f} bp")
            print(f"    Stressed PF:   {stressed_pf:.3f} [Must be > 1.0]")
