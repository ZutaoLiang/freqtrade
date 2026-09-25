"""R10 cross-sectional depth imbalance, beta-neutral (pre-registered in LOG.md). TRAIN only."""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import depthlib as D
import r3lib as L


def one(base):
    dep = D.build(f"{base}USDT")
    d = L.load(base)
    idx = pd.DatetimeIndex(d["date"])
    f = D.features(dep, idx)
    f["o"] = d["open"]
    h = f.resample("1h").agg({"imb5": "mean", "o": "first"})
    return base, h


def main():
    hs = {}
    with ProcessPoolExecutor(12) as ex:
        for b, h in ex.map(one, [b for b in L.U60 if b != "BTC"]):
            hs[b] = h
    O = pd.DataFrame({b: h.o for b, h in hs.items()})
    I = pd.DataFrame({b: h.imb5 for b, h in hs.items()})
    rows = []
    for LB, N in itertools.product([24, 168], [5, 10]):
        sig = I.rolling(LB, min_periods=LB // 2).mean()
        res = []
        for o in range(24):
            t = sig.index[(sig.index.hour == o)]
            prev_w = pd.Series(0.0, index=O.columns)
            daily = []
            for a, b in zip(t[:-1], t[1:]):
                s = sig.loc[a].dropna()
                if len(s) < 2 * N:
                    continue
                w = pd.Series(0.0, index=O.columns)
                w[s.nsmallest(N).index] = 1.0 / N
                w[s.nlargest(N).index] = -1.0 / N
                # decided at close of hour a -> trade at open of hour a+1, held to open of hour b+1
                i0, i1 = sig.index.get_loc(a) + 1, sig.index.get_loc(b) + 1
                if i1 >= len(O):
                    break
                r = O.iloc[i1] / O.iloc[i0] - 1
                turn = (w - prev_w).abs().sum()
                daily.append(((w * r).sum() - turn * 10e-4, a))
                prev_w = w * (1 + r) / (1 + (w * r).sum())      # drifted weights
            dr = pd.Series([x for x, _ in daily])
            res.append((dr.mean() * 1e4, dr.mean() / dr.std() * np.sqrt(len(dr)), len(dr)))
        r = np.array(res)
        rows.append({"LB": LB, "N": N, "mean_bp_day": r[:, 0].mean(), "t_med": np.median(r[:, 1]),
                     "t_min": r[:, 1].min(), "t_max": r[:, 1].max(), "offsets_pos": int((r[:, 0] > 0).sum()), "days": int(r[:, 2].mean())})
    print(pd.DataFrame(rows).round(2).to_string())


if __name__ == "__main__":
    main()
