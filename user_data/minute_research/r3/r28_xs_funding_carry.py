"""R28 Cross-Sectional Funding Rate Carry Arbitrage (Dollar-Neutral Factor).

TRAIN: 2025-01-01 .. 2025-10-01.
Universe: U162
Cadence: Every 8h settlement (00:00, 08:00, 16:00 UTC).
Factor: Settled funding rate at T.
Strategy:
  Long N lowest-funding coins, Short N highest-funding coins.
  Hold 8 hours (across next settlement).
  Earn Price Spread + Funding Carry Spread - Turnover Fees (10 bps/side).
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

import r3lib as L

B = "/root/freqtrade/user_data/data/binance_public"
U = pd.read_csv("universe_rank.csv")
U162 = U[U.med_qv >= 1e7].base.tolist()

START = "2025-01-01 00:00:00+00:00"
END = "2025-10-01 00:00:00+00:00"

# Grid: N in {5, 10, 15, 20}
NS = [5, 10, 15, 20]


def load_funding_and_prices():
    funding_dict = {}
    price_dict = {}
    print(f"Loading 8h funding and prices for {len(U162)} coins...")
    for b in U162:
        ff = f"{B}/funding/{b}USDT.parquet"
        kf = f"{B}/klines_1m/{b}USDT.parquet"
        if not os.path.exists(ff) or not os.path.exists(kf):
            continue
        # Funding rate
        f_df = pd.read_parquet(ff, columns=["date", "funding_rate"]).set_index("date")
        f_df = f_df[(f_df.index >= "2024-12-31") & (f_df.index <= END)]
        
        # Prices at 8h mark (00:00, 08:00, 16:00)
        k_df = pd.read_parquet(kf, columns=["date", "close"])
        k_df = k_df[(k_df.date >= "2024-12-31") & (k_df.date <= END)]
        if len(k_df) < 50000:
            continue
        p_8h = k_df.set_index("date").resample("8h").first()["close"].dropna()
        
        funding_dict[b] = f_df["funding_rate"]
        price_dict[b] = p_8h
        
    f_mat = pd.DataFrame(funding_dict).sort_index()
    p_mat = pd.DataFrame(price_dict).sort_index()
    
    # Align to common 8h timestamps
    common_idx = f_mat.index.intersection(p_mat.index)
    f_mat = f_mat.loc[common_idx]
    p_mat = p_mat.loc[common_idx]
    
    f_mat = f_mat[(f_mat.index >= START) & (f_mat.index < END)]
    p_mat = p_mat[(p_mat.index >= START) & (p_mat.index <= END)]
    
    print(f"Aligned: {len(f_mat)} 8h settlements x {f_mat.shape[1]} coins")
    return f_mat, p_mat


def eval_carry(f_mat: pd.DataFrame, p_mat: pd.DataFrame, n: int):
    # Future 8h price return
    fwd_ret = (p_mat.shift(-1) / p_mat) - 1.0
    
    records = []
    prev_longs = set()
    prev_shorts = set()
    cost_per_side = 0.0010 # 10 bps
    
    # We step through each 8h settlement
    for i in range(len(f_mat) - 1):
        t = f_mat.index[i]
        t_next = f_mat.index[i+1]
        
        # Funding rate at current settlement t
        curr_fr = f_mat.iloc[i].dropna()
        curr_fwd_ret = fwd_ret.loc[t].dropna() if t in fwd_ret.index else pd.Series()
        # Next funding rate at t_next (which will be settled at end of period)
        next_fr = f_mat.loc[t_next].dropna() if t_next in f_mat.index else pd.Series()
        
        common = curr_fr.index.intersection(curr_fwd_ret.index).intersection(next_fr.index)
        if len(common) < 30:
            continue
            
        ranked = curr_fr.loc[common].sort_values()
        bottom_n = ranked.index[:n].tolist() # lowest funding (negative / low)
        top_n = ranked.index[-n:].tolist()    # highest funding (positive / high)
        
        curr_longs = set(bottom_n)
        curr_shorts = set(top_n)
        
        # Turnover calculation
        l_enter = len(curr_longs - prev_longs) / n
        s_enter = len(curr_shorts - prev_shorts) / n
        turnover = (l_enter + s_enter)
        fee = turnover * cost_per_side
        
        # 1. Price Return: Long leg - Short leg
        l_px_ret = curr_fwd_ret.loc[list(curr_longs)].mean()
        s_px_ret = curr_fwd_ret.loc[list(curr_shorts)].mean()
        price_spread = 0.5 * (l_px_ret - s_px_ret)
        
        # 2. Funding Carry:
        # Long position pays next_fr (so receives -next_fr)
        # Short position receives next_fr
        l_fr_rec = -next_fr.loc[list(curr_longs)].mean()
        s_fr_rec = next_fr.loc[list(curr_shorts)].mean()
        funding_spread = 0.5 * (l_fr_rec + s_fr_rec)
        
        net = price_spread + funding_spread - fee
        
        records.append({
            "t": t, "price_spread": price_spread, "funding_spread": funding_spread,
            "fee": fee, "net": net
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
    
    sharpe = (df_res.net.mean() / df_res.net.std()) * np.sqrt(365 * 3) if df_res.net.std() > 0 else 0.0
    
    return {
        "n_periods": len(df_res),
        "mean_bp": net_bp.mean(),
        "sum_bp": net_bp.sum(),
        "win": (df_res.net > 0).mean(),
        "pf": pf,
        "day_mean_bp": day_mean,
        "day_t": day_t,
        "days_pos": (day_net > 0).mean(),
        "sharpe": sharpe,
        "avg_funding_carry_bp": df_res.funding_spread.mean() * 10000,
        "avg_price_spread_bp": df_res.price_spread.mean() * 10000
    }


if __name__ == "__main__":
    f_mat, p_mat = load_funding_and_prices()
    rows = []
    for n in NS:
        res = eval_carry(f_mat, p_mat, n)
        rows.append({"N": n, **res})
    res_df = pd.DataFrame(rows)
    print(res_df.to_string())
    res_df.to_csv("r28_train.csv", index=False)
