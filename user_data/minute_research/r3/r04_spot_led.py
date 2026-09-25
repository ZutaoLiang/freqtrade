"""R04 spot-led vs perp-led hourly moves (TRAIN only).

Pre-registration:
  hourly, per coin with a Binance spot market. r1 = perp 1h return.
  s_imb  = spot taker imbalance over the hour (2*taker_buy_quote - quote)/quote
  share_r = spot share of (spot+perp) quote volume in the hour / trailing 168h median share
  spot-led  : |r1| >= THR and s_imb*sign(r1) >= 0.10 and share_r >= 1.2  -> FOLLOW the move
  perp-led  : |r1| >= THR and s_imb*sign(r1) <= 0     and share_r <= 0.8  -> FADE the move
  grid: THR {1.5%, 3%} x type {spot-led follow, perp-led fade} x HOLD {240, 1440} min = 8 cells
  reference rows (not selectable): all moves >= THR, follow and fade
  gate: net mean > 0, day t >= 1.5, >= 1 trade/day
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L
import spotlib as S

CELLS = list(itertools.product([0.015, 0.03], ["spot_follow", "perp_fade", "ref_follow", "ref_fade"], [240, 1440]))


def one(base: str):
    d = L.load(base)
    h = S.hourly_panel(base, d)
    if h is None:
        return base, None
    r1 = h.c / h.o - 1
    simb = ((2 * h.stb - h.sqv) / h.sqv.where(h.sqv > 0)).fillna(0)
    share = h.sqv / (h.sqv + h.qv)
    share_r = share / share.rolling(168, min_periods=84).median().shift(1)
    sg = np.sign(r1)
    out = {}
    for thr, kind, hold in CELLS:
        big = r1.abs() >= thr
        if kind == "spot_follow":
            m, side = big & (simb * sg >= 0.10) & (share_r >= 1.2), sg
        elif kind == "perp_fade":
            m, side = big & (simb * sg <= 0) & (share_r <= 0.8), -sg
        elif kind == "ref_follow":
            m, side = big, sg
        else:
            m, side = big, -sg
        idx = h.i[m].to_numpy(np.int64)
        out[(thr, kind, hold)] = L.H.run(d, idx, side[m].to_numpy().astype(np.int8), hold=hold,
                                         cost_bps=L.cost_bps(base))
    return base, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(16) as ex:
        for base, out in ex.map(one, L.U60):
            if out is not None:
                res[base] = out
    print("coins with spot:", len(res), sorted(set(L.U60) - set(res)))
    rows = []
    for cell in CELLS:
        s = L.summarize({b: res[b][cell] for b in res})
        rows.append({"thr": cell[0], "kind": cell[1], "hold": cell[2], **s})
    df = pd.DataFrame(rows)
    df["per_day"] = df.n / 273
    print(df.to_string())
    df.to_csv("r04_train.csv", index=False)
