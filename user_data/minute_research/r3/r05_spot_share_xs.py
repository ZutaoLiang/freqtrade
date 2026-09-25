"""R05 cross-sectional spot-share tilt (TRAIN only).

Pre-registration:
  hourly panel over U60 coins that have a spot market.
  share_LB = spot quote volume / (spot + perp) over the last LB hours ; sig = share_LB / trailing 30d median hourly share
  every 24h at hour offset o (all 24 offsets evaluated, results averaged -> no rebalance-time luck):
     long_top     : long the N coins with highest sig, hold 24h
     short_bottom : short the N coins with lowest sig, hold 24h
  entry = perp open of the next hour, exit = perp open 24h later ; cost 2 x (7.5|10) bp per trade
  grid: LB {24, 72} x arm {long_top, short_bottom} x N {5, 10} = 8 cells
  reference: equal-weight all-coin long, same timing (to separate the tilt from market beta)
  gate: mean-over-offsets net > 0 and day-t (daily book return) >= 1.5 for the median offset
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L
import spotlib as S


def one(base: str):
    d = L.load(base)
    h = S.hourly_panel(base, d)
    if h is None:
        return base, None
    return base, h[["o", "qv", "sqv"]]


def main():
    hs = {}
    with ProcessPoolExecutor(16) as ex:
        for b, h in ex.map(one, L.U60):
            if h is not None:
                hs[b] = h
    O = pd.DataFrame({b: h.o for b, h in hs.items()})
    QV = pd.DataFrame({b: h.qv for b, h in hs.items()})
    SQV = pd.DataFrame({b: h.sqv for b, h in hs.items()})
    cost = pd.Series({b: 2 * L.cost_bps(b) / 1e4 for b in hs})
    fwd = O.shift(-25) / O.shift(-1) - 1          # decided at close of hour t -> enter next hour open, exit +24h
    share_h = SQV / (SQV + QV)
    base_med = share_h.rolling(720, min_periods=360).median().shift(1)
    rows = []
    for LB in (24, 72):
        sh = SQV.rolling(LB).sum() / (SQV.rolling(LB).sum() + QV.rolling(LB).sum())
        sig = (sh / base_med).where(SQV.rolling(LB).sum() > 0)
        rk = sig.rank(axis=1, ascending=False)
        rkb = sig.rank(axis=1, ascending=True)
        for arm, N in itertools.product(["long_top", "short_bottom", "ref_all_long"], [5, 10]):
            if arm == "long_top":
                w = (rk <= N).astype(float); side = 1
            elif arm == "short_bottom":
                w = (rkb <= N).astype(float); side = -1
            else:
                w = sig.notna().astype(float); side = 1
            book = ((side * fwd - cost) * w).sum(axis=1) / w.sum(axis=1).replace(0, np.nan)
            res = []
            for o in range(24):
                b = book[(book.index.hour == o)].dropna()
                res.append((b.mean() * 1e4, b.mean() / b.std() * np.sqrt(len(b)), len(b)))
            r = np.array(res)
            rows.append({"LB": LB, "arm": arm, "N": N, "mean_bp_avg": r[:, 0].mean(), "t_med": np.median(r[:, 1]),
                         "t_min": r[:, 1].min(), "t_max": r[:, 1].max(), "offsets_pos": int((r[:, 0] > 0).sum()),
                         "days": int(r[:, 2].mean())})
    df = pd.DataFrame(rows)
    print("coins:", len(hs))
    print(df.round(2).to_string())
    df.to_csv("r05_train.csv", index=False)


if __name__ == "__main__":
    main()
