"""R07 liquidity-conditioned impact (pre-registered in LOG.md). TRAIN only."""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import depthlib as D
import r3lib as L

CELLS = list(itertools.product([0.02, 0.04], ["thin_fade", "thick_follow", "ref_fade", "ref_follow"], [240, 1440]))


def one(base: str):
    dep = D.build(f"{base}USDT")
    if dep is None:
        return base, None
    d = L.load(base)
    idx = pd.DatetimeIndex(d["date"])
    f = D.features(dep, idx)
    f["o"], f["c"], f["i"] = d["open"], d["close"], np.arange(len(f))
    h = f.resample("1h").agg({"o": "first", "c": "last", "tot2": "mean", "i": "last"}).dropna(subset=["o", "c"])
    pre = h.tot2.shift(1)                                           # depth in the hour before the event hour
    thin = pre / h.tot2.rolling(168, min_periods=120).median().shift(2)
    r1 = h.c / h.o - 1
    sg = np.sign(r1)
    out = {}
    for thr, kind, hold in CELLS:
        big = (r1.abs() >= thr) & thin.notna()
        if kind == "thin_fade":
            m, side = big & (thin <= 0.6), -sg
        elif kind == "thick_follow":
            m, side = big & (thin >= 1.2), sg
        elif kind == "ref_fade":
            m, side = big, -sg
        else:
            m, side = big, sg
        out[(thr, kind, hold)] = L.H.run(d, h.i[m].to_numpy(np.int64), side[m].to_numpy().astype(np.int8),
                                         hold=hold, cost_bps=L.cost_bps(base))
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(12) as ex:
        for b, r in ex.map(one, L.U60):
            if r is not None:
                res[b] = r
    print("coins with depth:", len(res))
    rows = []
    for cell in CELLS:
        s = L.summarize({b: res[b][cell] for b in res if b != "BTC"})
        rows.append({"thr": cell[0], "kind": cell[1], "hold": cell[2], **s})
    df = pd.DataFrame(rows)
    df["per_day"] = df.n / 273
    print(df.to_string())
    df.to_csv("r07_train.csv", index=False)
