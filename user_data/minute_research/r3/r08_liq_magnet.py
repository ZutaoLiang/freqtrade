"""R08 estimated liquidation-cluster magnet (pre-registered in LOG.md). TRAIN only."""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from numba import njit

import r3lib as L

BIN = 0.0025
LEVS = np.array([10.0, 25.0, 50.0])
GRID = list(itertools.product([0.03, 0.06], [0.3, 0.6], [240, 1440], ["toward", "away"]))


@njit(cache=True)
def heat(lp, hi, lo, oi, b0, nb, levs, half_life_bars, hour_mask, W_list):
    """Returns imb[k, j] for each window k at each snapshot j (NaN where not an hour snapshot)."""
    longc = np.zeros(nb); shortc = np.zeros(nb)
    dec = 0.5 ** (1.0 / half_life_bars)
    n = lp.shape[0]
    out = np.full((W_list.shape[0], n), np.nan)
    for j in range(1, n):
        longc *= dec; shortc *= dec
        # a long-liq level at or above this bar's low has been reached; a short-liq level at or below the high too
        bl = int(np.ceil(np.log(lo[j]) / 0.0025)) - b0
        bh = int(np.floor(np.log(hi[j]) / 0.0025)) - b0
        for b in range(max(bl, 0), nb):
            longc[b] = 0.0
        for b in range(0, min(bh + 1, nb)):
            shortc[b] = 0.0
        d = oi[j] - oi[j - 1]
        if d > 0:
            for L_ in levs:
                bL = int(np.round((lp[j] + np.log(1.0 - 1.0 / L_)) / 0.0025)) - b0
                bS = int(np.round((lp[j] + np.log(1.0 + 1.0 / L_)) / 0.0025)) - b0
                if 0 <= bL < nb: longc[bL] += d / 2.0 / levs.shape[0]
                if 0 <= bS < nb: shortc[bS] += d / 2.0 / levs.shape[0]
        elif d < 0 and oi[j - 1] > 0:
            s = oi[j] / oi[j - 1]
            longc *= s; shortc *= s
        if hour_mask[j]:
            bp = int(np.round(lp[j] / 0.0025)) - b0
            for k in range(W_list.shape[0]):
                w = int(np.round(np.log(1.0 + W_list[k]) / 0.0025))
                ab = 0.0; be = 0.0
                for b in range(bp + 1, min(bp + w + 1, nb)):
                    ab += shortc[b]
                for b in range(max(bp - w, 0), bp):
                    be += longc[b]
                if ab + be > 0:
                    out[k, j] = (ab - be) / (ab + be)
    return out


def one(base: str):
    d = L.load(base)
    m = pd.read_parquet(f"{L.ROOT}/metrics/{base}USDT.parquet", columns=["date", "sum_open_interest_value"])
    m = m[m.date < L.TRAIN[1]].drop_duplicates("date").set_index("date").sort_index()
    k = pd.DataFrame({"h": d["high"], "l": d["low"], "c": d["close"], "i": np.arange(len(d["close"]))},
                     index=pd.DatetimeIndex(d["date"]))
    # 5m bars ending at snapshot time t: (t-5min, t] -> 1m bars with open in [t-5, t-1]
    k5 = k.resample("5min", label="right", closed="left").agg({"h": "max", "l": "min", "c": "last", "i": "last"})
    x = k5.join(m, how="inner").dropna()
    lp = np.log(x.c.to_numpy())
    b0 = int(np.floor(lp.min() / BIN)) - 200
    nb = int(np.ceil(lp.max() / BIN)) - b0 + 200
    hour = np.asarray(x.index.minute == 0)
    imb = heat(lp, x.h.to_numpy(), x.l.to_numpy(), x.sum_open_interest_value.to_numpy(), b0, nb, LEVS,
               7 * 288.0, hour, np.array([0.03, 0.06]))
    ent = x.i.to_numpy() + 5            # last 1m bar of the window is t-1min; +5 -> signal bar t+4min, fill t+5min
    out = {}
    for (W, T, hold, arm) in GRID:
        v = imb[0 if W == 0.03 else 1]
        sel = np.where(np.abs(v) >= T)[0]
        side = np.sign(v[sel]).astype(np.int8) * (1 if arm == "toward" else -1)
        out[(W, T, hold, arm)] = L.H.run(d, ent[sel], side, hold=hold, cost_bps=L.cost_bps(base))
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for b, r in ex.map(one, L.U60):
            res[b] = r
    rows = []
    for cell in GRID:
        s = L.summarize({b: res[b][cell] for b in res})
        rows.append({"W": cell[0], "T": cell[1], "hold": cell[2], "arm": cell[3], **s})
    df = pd.DataFrame(rows)
    df["per_day"] = df.n / 273
    print(df.to_string())
    df.to_csv("r08_train.csv", index=False)
