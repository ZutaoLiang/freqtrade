"""R26 Cross-Sectional Momentum vs Reversal Factor (APFF Market-Neutral Portfolio).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U162
Cost: 10 bp per side per unit turnover.
"""
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

START = "2025-01-01"
END = "2025-10-01"

# Grid: LB in {4, 12, 24} x H in {4, 8} x N in {5, 10} x Arm in {momentum, reversal}
LBS = [4, 12, 24]
HS = [4, 8]
NS = [5, 10]
ARMS = ["momentum", "reversal"]
GRID = list(itertools.product(LBS, HS, NS, ARMS))


def load_hourly_prices():
    prices = {}
    print(f"Loading 1h prices for {len(U162)} coins...")
    for b in U162:
        kf = f"{B}/klines_1m/{b}USDT.parquet"
        if not os.path.exists(kf):
            continue
        df = pd.read_parquet(kf, columns=["date", "close"])
        df = df[(df.date >= "2024-12-30") & (df.date < END)]
        if len(df) < 50000:
            continue
        h = df.set_index("date").resample("1h").last()["close"].dropna()
        prices[b] = h
    pdf = pd.DataFrame(prices)
    pdf = pdf[(pdf.index >= START) & (pdf.index < END)]
    print(f"Loaded matrix: {pdf.shape[0]} hours x {pdf.shape[1]} coins")
    return pdf


def eval_cell(pdf: pd.DataFrame, lb: int, h_step: int, n: int, arm: str):
    # Log or simple return over lb hours
    ret_lb = (pdf / pdf.shift(lb)) - 1.0
    # Future return over next h_step hours
    fwd_ret = (pdf.shift(-h_step) / pdf) - 1.0
    
    # We step every h_step hours
    # To avoid offset dependency, we can test the primary 0-offset
    idx_steps = list(range(lb, len(pdf) - h_step, h_step))
    
    records = []
    prev_longs = set()
    prev_shorts = set()
    cost_per_side = 0.0010 # 10 bps
    
    for i in idx_steps:
        t = pdf.index[i]
        row_lb = ret_lb.iloc[i].dropna()
        row_fwd = fwd_ret.iloc[i].dropna()
        common = row_lb.index.intersection(row_fwd.index)
        if len(common) < 30:
            continue
            
        ranked = row_lb.loc[common].sort_values()
        bottom_n = ranked.index[:n].tolist()
        top_n = ranked.index[-n:].tolist()
        
        if arm == "momentum":
            curr_longs = set(top_n)
            curr_shorts = set(bottom_n)
        else: # reversal
            curr_longs = set(bottom_n)
            curr_shorts = set(top_n)
            
        # Turnover calculation
        # New longs entering / exiting
        l_enter = len(curr_longs - prev_longs) / n
        s_enter = len(curr_shorts - prev_shorts) / n
        turnover = (l_enter + s_enter) # fraction of portfolio turned over
        fee = turnover * cost_per_side
        
        # PnL: Long leg mean fwd_ret minus Short leg mean fwd_ret
        l_ret = row_fwd.loc[list(curr_longs)].mean()
        s_ret = row_fwd.loc[list(curr_shorts)].mean()
        
        gross = 0.5 * l_ret - 0.5 * s_ret
        net = gross - fee
        
        records.append({
            "t": t, "gross": gross, "net": net, "fee": fee,
            "l_ret": l_ret, "s_ret": s_ret
        })
        
        prev_longs = curr_longs
        prev_shorts = curr_shorts
        
    df_res = pd.DataFrame(records)
    if len(df_res) == 0:
        return {}
        
    net_bp = df_res.net * 10000
    df_res["day"] = pd.to_datetime(df_res.t).dt.floor("D")
    day_net = df_res.groupby("day")["net"].sum() * 10000
    
    n_days = len(day_net)
    day_mean = day_net.mean()
    day_std = day_net.std(ddof=1) if n_days > 1 else 1e-9
    day_t = (day_mean / (day_std / np.sqrt(n_days))) if day_std > 0 else 0.0
    
    pos_sum = df_res.net[df_res.net > 0].sum()
    neg_sum = -df_res.net[df_res.net < 0].sum()
    pf = (pos_sum / neg_sum) if neg_sum > 0 else 999.0
    
    sharpe = (df_res.net.mean() / df_res.net.std()) * np.sqrt(365 * 24 / h_step) if df_res.net.std() > 0 else 0.0
    
    return {
        "n_periods": len(df_res),
        "mean_bp": net_bp.mean(),
        "sum_bp": net_bp.sum(),
        "win": (df_res.net > 0).mean(),
        "pf": pf,
        "day_mean_bp": day_mean,
        "day_t": day_t,
        "days_pos": (day_net > 0).mean(),
        "sharpe": sharpe
    }


if __name__ == "__main__":
    pdf = load_hourly_prices()
    rows = []
    for lb, h, n, arm in GRID:
        res = eval_cell(pdf, lb, h, n, arm)
        rows.append({
            "lb": lb, "h": h, "n": n, "arm": arm,
            **res
        })
    res_df = pd.DataFrame(rows)
    print(res_df.to_string())
    res_df.to_csv("r26_train.csv", index=False)
