"""R18 Cumulative Volume Delta (CVD) Absorption Divergence vs Momentum (TRAIN only).

Pre-registration:
  Data: 1m klines, aggregate to rolling W minutes (60m, 120m).
  Delta = 2 * taker_buy_quote_volume - quote_volume.
  Relative Delta RD = rolling_sum(Delta, W) / rolling_sum(quote_volume, W).
  Price Return Ret = Close / Close[t-W] - 1.
  Arms:
    1. Absorption (Divergence / Fade):
       Bullish: Ret <= -R_thr and RD >= +D_thr -> Long.
       Bearish: Ret >= +R_thr and RD <= -D_thr -> Short.
    2. Continuation (Concordance / Follow):
       Bullish: Ret >= +R_thr and RD >= +D_thr -> Long.
       Bearish: Ret <= -R_thr and RD <= -D_thr -> Short.
  Exit: Time exit HOLD minutes (60, 120, 240).
  Grid: W in {60, 120} x R_thr in {0.015, 0.025} x D_thr in {0.10, 0.20} x Arm in {divergence, continuation} = 16 cells (hold=120).
        Let's test W {60, 120} x R_thr {0.015, 0.025} x Arm {divergence, continuation} x HOLD {60, 120} = 16 cells (D_thr=0.10).
  Universe: U60; Cost: 7.5/10 bp per side; TRAIN: 2025-01-01 .. 2025-10-01.
  Gate: net mean > 0, day t >= 1.5, >= 1 trade/day.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

GRID = list(itertools.product([60, 120], [0.015, 0.025], ["divergence", "continuation"], [60, 120]))


def one(base: str):
    d = L.load(base)
    c = d["close"]
    qv = d["quote_volume"]
    tb = d["taker_buy_quote_volume"]
    delta = 2.0 * tb - qv
    
    cost = L.cost_bps(base)
    out = {}
    
    for W, R_thr, arm, hold in GRID:
        roll_qv = pd.Series(qv).rolling(W, min_periods=W).sum().to_numpy()
        roll_delta = pd.Series(delta).rolling(W, min_periods=W).sum().to_numpy()
        rd = np.where(roll_qv > 0, roll_delta / roll_qv, 0.0)
        
        c_prev = pd.Series(c).shift(W).to_numpy()
        ret = np.where(c_prev > 0, c / c_prev - 1.0, 0.0)
        
        D_thr = 0.10
        if arm == "divergence":
            long_sig = (ret <= -R_thr) & (rd >= D_thr)
            short_sig = (ret >= R_thr) & (rd <= -D_thr)
        else: # continuation
            long_sig = (ret >= R_thr) & (rd >= D_thr)
            short_sig = (ret <= -R_thr) & (rd <= -D_thr)
            
        l_idx = np.where(long_sig)[0]
        s_idx = np.where(short_sig)[0]
        
        all_idx = np.concatenate([l_idx, s_idx])
        all_side = np.concatenate([np.ones(len(l_idx), dtype=np.int8), -np.ones(len(s_idx), dtype=np.int8)])
        
        if len(all_idx) == 0:
            out[(W, R_thr, arm, hold)] = pd.DataFrame()
            continue
            
        order = np.argsort(all_idx)
        all_idx = all_idx[order]
        all_side = all_side[order]
        
        tr = L.H.run(d, all_idx, all_side, hold=hold, cost_bps=cost)
        out[(W, R_thr, arm, hold)] = tr
        
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, L.U60):
            res[base] = out
            
    rows = []
    days = 273
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res})
        rows.append({
            "W": cell[0], "R_thr": cell[1], "arm": cell[2], "hold": cell[3],
            **s, "per_day": s.get("n", 0) / days
        })
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("r18_train.csv", index=False)
