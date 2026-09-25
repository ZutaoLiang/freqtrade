"""R06 order-book depth imbalance (pre-registered in LOG.md). TRAIN only unless --seg given."""
from __future__ import annotations

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import depthlib as D
import r3lib as L

GRID = list(itertools.product([1, 5], [0.90, 0.95], [240, 1440], ["long_high", "short_low"]))


def one(base: str):
    dep = D.build(f"{base}USDT")
    if dep is None:
        return base, None
    d = L.load(base)
    idx = pd.DatetimeIndex(d["date"])
    f = D.features(dep, idx)
    f["i"] = np.arange(len(f))
    out = {}
    cov = {}
    for K in (1, 5):
        h = f[[f"imb{K}", "i"]].resample("1h").agg({f"imb{K}": "mean", "i": "last"}).dropna()
        pct = h[f"imb{K}"].rolling(168, min_periods=120).rank(pct=True)
        cov[K] = len(h)
        for P, hold, arm in itertools.product([0.90, 0.95], [240, 1440], ["long_high", "short_low"]):
            m = (pct >= P) if arm == "long_high" else (pct <= 1 - P)
            idx_ = h.i[m].to_numpy(np.int64)
            out[(K, P, hold, arm)] = L.H.run(d, idx_, 1 if arm == "long_high" else -1, hold=hold,
                                             cost_bps=L.cost_bps(base))
    return base, (out, cov)


if __name__ == "__main__":
    bases = [b for b in L.U60]
    res, cov = {}, {}
    with ProcessPoolExecutor(12) as ex:
        for b, r in ex.map(one, bases):
            if r is not None:
                res[b], cov[b] = r
    print("coins with depth:", len(res), "median hours covered:", int(np.median([c[1] for c in cov.values()])))
    rows = []
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res if b != "BTC"})
        rows.append({"K": cell[0], "P": cell[1], "hold": cell[2], "arm": cell[3], **s})
    df = pd.DataFrame(rows)
    df["per_day"] = df.n / 273
    print(df.to_string())
    df.to_csv("r06_train.csv", index=False)
