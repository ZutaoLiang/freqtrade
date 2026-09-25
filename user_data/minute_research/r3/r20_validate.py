"""R20 VALID Test: Evaluate frozen OI Buildup Short Cascade on VALID segment (2025-10-01 .. 2026-03-01)."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

B = "/root/freqtrade/user_data/data/binance_public"
U = pd.read_csv("/root/freqtrade/user_data/minute_research/r3/universe_rank.csv")
U162 = U[U.med_qv >= 1e7].base.tolist()

START = "2025-12-01"
END = "2026-03-01"


def get_btc_filter():
    d = L.load("BTC", end=END)
    df = pd.DataFrame({"close": d["close"]}, index=pd.DatetimeIndex(d["date"]))
    df5 = df.resample("5min").agg({"close": "last"}).dropna()
    r4h = df5.close.pct_change(48).shift(1)
    return r4h

BTC_R4H = get_btc_filter()


def one(base: str):
    mf = f"{B}/metrics/{base}USDT.parquet"
    kf = f"{B}/klines_1m/{base}USDT.parquet"
    ff = f"{B}/funding/{base}USDT.parquet"
    if not os.path.exists(mf) or not os.path.exists(kf) or not os.path.exists(ff):
        return base, {}
    d = pd.read_parquet(kf, columns=["date", "open", "high", "low", "close", "volume"])
    # Need 7 days before START for rolling sigma
    d = d[(d.date >= "2025-11-20") & (d.date < END)]
    if len(d) < 30000:
        return base, {}
        
    df5 = d.set_index("date").resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    mt = pd.read_parquet(mf).set_index("date")[["sum_open_interest"]]
    df5 = df5.join(mt, how="left")
    df5["oi"] = df5.sum_open_interest.ffill(limit=3)
    
    f_df = pd.read_parquet(ff).set_index("date")[["funding_rate"]]
    df5 = df5.join(f_df, how="left")
    df5["fr"] = df5.funding_rate.ffill(limit=24)
    
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
    
    W = 24
    OI_thr = 0.08
    hold_m = 240
    hold_bars = hold_m // 5
    
    oi_shift = pd.Series(oi).shift(W).to_numpy()
    oi_change = np.where(oi_shift > 0, oi / oi_shift - 1.0, 0.0)
    box_low = pd.Series(l).rolling(W, min_periods=W).min().shift(1).to_numpy()
    
    coiled = (oi_change >= OI_thr) & (np.abs(r4h) <= 0.5 * sig4h)
    recent_coiled = pd.Series(coiled).rolling(12, min_periods=1).max().to_numpy().astype(bool)
    
    crowded = (fr >= 0.0000) & (btc_r <= 0.02)
    
    short_sig = recent_coiled & (c < box_low) & (pd.Series(c).shift(1) >= pd.Series(box_low).shift(1)) & crowded
    
    # Filter only signals strictly within VALID segment
    valid_mask = (df5.index >= START) & (df5.index < END)
    short_sig = short_sig & np.asarray(valid_mask, dtype=bool)
    
    idx = np.where(short_sig)[0]
    if len(idx) == 0:
        return base, {"standard": pd.DataFrame(), "stress_cost": pd.DataFrame()}
        
    side = -np.ones(len(idx), dtype=np.int8)
    tr_std = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost)
    tr_stress = L.H.run(d5, idx, side, hold=hold_bars, cost_bps=cost * 1.5)
    
    return base, {"standard": tr_std, "stress_cost": tr_stress}


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, U162):
            if out:
                res[base] = out
                
    std_trades = {b: res[b]["standard"] for b in res if "standard" in res[b]}
    stress_trades = {b: res[b]["stress_cost"] for b in res if "stress_cost" in res[b]}
    
    s_std = L.summarize(std_trades)
    s_stress = L.summarize(stress_trades)
    
    days = 151 # Oct 1 to Mar 1 = 151 days
    print("=== VALID STANDARD ===")
    print("Trades:", s_std.get("n", 0), "Per day:", s_std.get("n", 0) / days)
    print("Net Mean (bp):", s_std.get("mean_bp"), "Median (bp):", s_std.get("med_bp"))
    print("Win Rate:", s_std.get("win"), "Profit Factor:", s_std.get("pf"))
    print("Trading Days:", s_std.get("days"), "Day Mean (bp):", s_std.get("day_mean_bp"), "Day t-stat:", s_std.get("day_t"))
    print("Days Pos Fraction:", s_std.get("days_pos"))
    
    print("\n=== VALID STRESS COST (Fee x 1.5) ===")
    print("Net Mean (bp):", s_stress.get("mean_bp"), "Profit Factor:", s_stress.get("pf"))
    
    # Monthly breakdown
    all_tr = pd.concat([t for t in std_trades.values() if len(t)]).reset_index(drop=True)
    if len(all_tr):
        all_tr["month"] = pd.DatetimeIndex(all_tr.t).to_period("M")
        m_perf = all_tr.groupby("month").apply(lambda g: pd.Series({
            "n": len(g), "mean_bp": g.ret.mean() * 1e4, "pf": g.ret[g.ret > 0].sum() / -g.ret[g.ret < 0].sum()
        }))
        print("\n=== MONTHLY BREAKDOWN ===")
        print(m_perf.to_string())
