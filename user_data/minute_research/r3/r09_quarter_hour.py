"""R09 quarter-hour opening imbalance (pre-registered in LOG.md). TRAIN only."""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

GRID = list(itertools.product(["open", "control"], [0.90, 0.95], [240, 720], ["long_high", "short_low"]))


def one(base: str):
    d = L.load(base)
    idx = pd.DatetimeIndex(d["date"])
    qv, tb = d["quote_volume"], d["taker_buy_quote_volume"]
    flow = pd.Series(2 * tb - qv, index=idx)
    vol = pd.Series(qv, index=idx)
    out = {}
    for kind, minute in (("open", 0), ("control", 7)):
        sel = (idx.minute % 15) == minute
        f = flow.where(sel, 0.0).rolling(240).sum()          # last 16 selected minutes live in the last 240 bars
        v = vol.where(sel, 0.0).rolling(240).sum()
        S = (f / v.replace(0, np.nan))
        h = pd.DataFrame({"S": S, "i": np.arange(len(S))}).iloc[59::60]   # hour closes (bar :59)
        h = h[idx[h.i].minute == 59]
        pct = h.S.rolling(720, min_periods=360).rank(pct=True)
        for P, hold, arm in itertools.product([0.90, 0.95], [240, 720], ["long_high", "short_low"]):
            m = (pct >= P) if arm == "long_high" else (pct <= 1 - P)
            out[(kind, P, hold, arm)] = L.H.run(d, h.i[m].to_numpy(np.int64), 1 if arm == "long_high" else -1,
                                                hold=hold, cost_bps=L.cost_bps(base))
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for b, r in ex.map(one, L.U60):
            res[b] = r
    rows = []
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res})
        rows.append({"kind": cell[0], "P": cell[1], "hold": cell[2], "arm": cell[3], **s})
    df = pd.DataFrame(rows)
    df["per_day"] = df.n / 273
    print(df.to_string())
    df.to_csv("r09_train.csv", index=False)
