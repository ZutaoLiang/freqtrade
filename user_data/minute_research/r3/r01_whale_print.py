"""R01 large-trade prints (TRAIN only).

Pre-registration (written before running):
  ats      = quote_volume / count on a 1m bar (average trade size, USDT)
  ats_r    = ats / trailing-1440-bar median ats;  qv_r = quote_volume / trailing-1440 median
  imb      = (2*taker_buy_quote - quote_volume) / quote_volume
  event    = ats_r >= A and qv_r >= V and |imb| >= 0.5
  side     = follow: sign(imb) ; fade: -sign(imb)
  exit     = time exit after HOLD bars, no stop (stop variants only after a TRAIN pass)
  grid     = A {3,5} x V {5,10} x HOLD {60,240} x arm {follow,fade} = 16 cells
  universe = U60 ; cost 7.5/10 bp per side ; TRAIN 2025-01..09
  gate     = net mean > 0, day t >= 1.5, >= 1 trade/day
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

GRID = list(itertools.product([3, 5], [5, 10], [60, 240], ["follow", "fade"]))


def one(base: str):
    d = L.load(base)
    qv, cnt, tb = d["quote_volume"], d["count"], d["taker_buy_quote_volume"]
    ats = np.where(cnt > 0, qv / np.maximum(cnt, 1), np.nan)
    ats_r = ats / L.roll_med(ats, 1440)
    qv_r = qv / L.roll_med(qv, 1440)
    imb = np.where(qv > 0, (2 * tb - qv) / np.maximum(qv, 1e-9), 0)
    out = {}
    for A, V, hold, arm in GRID:
        idx = np.where((ats_r >= A) & (qv_r >= V) & (np.abs(imb) >= 0.5))[0]
        side = np.sign(imb[idx]).astype(np.int8) * (1 if arm == "follow" else -1)
        out[(A, V, hold, arm)] = L.H.run(d, idx, side, hold=hold, cost_bps=L.cost_bps(base))
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, L.U60):
            res[base] = out
    rows = []
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res})
        rows.append({"A": cell[0], "V": cell[1], "hold": cell[2], "arm": cell[3], **s})
    df = pd.DataFrame(rows)
    df["per_day"] = df.n / 273
    print(df.to_string())
    df.to_csv("r01_train.csv", index=False)
