"""R03 round-number crossings (Osler 2003: stops cluster just beyond round numbers -> crossings accelerate).

Pre-registration:
  m = 10^floor(log10(price)); round step = smallest of m*{1, 1/2, 1/5, 1/10, 1/20, 1/50} whose step/price >= MINSP
  (step recomputed daily from the previous day's close; levels = integer multiples of step)
  event: 1m close crosses a level (prev close < L <= close, or mirror), and that level was not crossed
         in the previous 1440 bars (fresh)
  side:  follow = direction of the crossing ; fade = opposite
  exit:  time HOLD bars, no stop
  grid:  MINSP {2.5%, 5%} x HOLD {60, 240} x arm {follow, fade} = 8 cells ; TRAIN only
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

GRID = list(itertools.product([0.025, 0.05], [60, 240], ["follow", "fade"]))
FRACS = np.array([1, 1 / 2, 1 / 5, 1 / 10, 1 / 20, 1 / 50])


def steps_for(close: np.ndarray, dates: pd.Series, minsp: float) -> np.ndarray:
    day = dates.dt.floor("D")
    prev = pd.Series(close).groupby(day.to_numpy()).transform("last").shift(1440).bfill().to_numpy()
    out = np.empty_like(prev)
    for i, p in enumerate(np.unique(prev)):
        m = 10 ** np.floor(np.log10(p))
        cand = m * FRACS
        ok = cand[cand / p >= minsp]
        out[prev == p] = ok.min() if len(ok) else m
    return out


def one(base: str):
    d = L.load(base)
    c = d["close"]
    out = {}
    for minsp in (0.025, 0.05):
        st = steps_for(c, d["date"], minsp)
        lv_now = np.floor(c / st)
        lv_prev = np.floor(np.r_[c[0], c[:-1]] / st)
        up = lv_now > lv_prev
        dn = lv_now < lv_prev
        lvl = np.where(up, lv_now, np.where(dn, lv_prev, np.nan))  # index of crossed level
        ev = np.where(up | dn)[0]
        fresh = []
        last = {}
        for i in ev:
            key = (round(st[i], 12), lvl[i])
            if i - last.get(key, -10**9) > 1440:
                fresh.append(i)
            last[key] = i
        fresh = np.array(fresh, np.int64)
        dirn = np.where(up[fresh], 1, -1).astype(np.int8)
        for hold in (60, 240):
            for arm in ("follow", "fade"):
                out[(minsp, hold, arm)] = L.H.run(d, fresh, dirn * (1 if arm == "follow" else -1), hold=hold,
                                                  cost_bps=L.cost_bps(base))
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, L.U60):
            res[base] = out
    rows = []
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res})
        rows.append({"minsp": cell[0], "hold": cell[1], "arm": cell[2], **s})
    df = pd.DataFrame(rows)
    df["per_day"] = df.n / 273
    print(df.to_string())
    df.to_csv("r03_train.csv", index=False)
