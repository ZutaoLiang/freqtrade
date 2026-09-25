"""R02 retail frenzy (TRAIN only).

Pre-registration:
  hourly aggregation of 1m bars (hour closes on the :59 bar; entry = next 1m open)
  cnt_r = hourly trade count / trailing 168h median ; ats_r = hourly avg trade size / trailing 168h median
  r1h   = hourly close / hourly open - 1
  event = cnt_r >= C and ats_r <= S and |r1h| >= 2%
  side  = fade: -sign(r1h) ; follow: +sign(r1h)
  grid  = C {3,5} x S {0.7, inf} x HOLD {240, 1440} min x arm {fade, follow} = 16
  gate  = net mean > 0, day t >= 1.5, >= 1 trade/day (TRAIN)
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

GRID = list(itertools.product([3, 5], [0.7, np.inf], [240, 1440], ["fade", "follow"]))


def one(base: str):
    d = L.load(base)
    df = pd.DataFrame({k: d[k] for k in ("open", "close", "quote_volume", "count")}, index=pd.DatetimeIndex(d["date"]))
    df["i"] = np.arange(len(df))
    h = df.resample("1h").agg({"open": "first", "close": "last", "quote_volume": "sum", "count": "sum", "i": "last"})
    h = h.dropna()
    ats = h.quote_volume / h["count"].clip(lower=1)
    cnt_r = h["count"] / h["count"].rolling(168, min_periods=84).median().shift(1)
    ats_r = ats / ats.rolling(168, min_periods=84).median().shift(1)
    r1 = h.close / h.open - 1
    out = {}
    for C, S, hold, arm in GRID:
        m = (cnt_r >= C) & (ats_r <= S) & (r1.abs() >= 0.02)
        idx = h.i[m].to_numpy(np.int64)
        side = (-np.sign(r1[m]) if arm == "fade" else np.sign(r1[m])).to_numpy().astype(np.int8)
        out[(C, S, hold, arm)] = L.H.run(d, idx, side, hold=hold, cost_bps=L.cost_bps(base))
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, L.U60):
            res[base] = out
    rows = []
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res})
        rows.append({"C": cell[0], "S": cell[1], "hold": cell[2], "arm": cell[3], **s})
    df = pd.DataFrame(rows)
    df["per_day"] = df.n / 273
    print(df.to_string())
    df.to_csv("r02_train.csv", index=False)
