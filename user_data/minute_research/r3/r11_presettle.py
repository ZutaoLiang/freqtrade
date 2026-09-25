"""R11 pre-settlement receiver harvest (pre-registered in LOG.md). TRAIN only unless SEG env set."""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import r3lib as L

U = pd.read_csv("universe_rank.csv")
U162 = U[U.med_qv >= 1e7].base.tolist()
U60 = set(L.U60)
SEG = {"TRAIN": L.TRAIN, "VALID": L.VALID, "HOLD": L.HOLD}[os.environ.get("SEG", "TRAIN")]
CELLS = list(itertools.product([0.001, 0.003], [15, 60, 240], ["atT", "Tplus1"]))


def one(base: str):
    k = pd.read_parquet(f"{L.ROOT}/klines_1m/{base}USDT.parquet", columns=["date", "open"])
    k = k[(k.date >= pd.Timestamp(SEG[0], tz="UTC") - pd.Timedelta(days=2)) & (k.date < SEG[1])]
    op = pd.Series(k.open.to_numpy(), index=pd.DatetimeIndex(k.date))
    f = pd.read_parquet(f"{L.ROOT}/funding/{base}USDT.parquet")
    f["date"] = f.date.dt.round("min")
    f = f.drop_duplicates("date").set_index("date").sort_index()
    f["prev"] = f.funding_rate.shift(1)
    f = f[(f.index >= SEG[0]) & (f.index < SEG[1])].dropna(subset=["prev"])
    c = (10.0 if base in U60 else 15.0) / 1e4
    rows = []
    for T, r in f.iterrows():
        side = 1 if r.prev < 0 else -1
        for H in (15, 60, 240):
            t0 = T - pd.Timedelta(minutes=H)
            if t0 not in op.index or T not in op.index:
                continue
            e = op[t0]
            for ex, tx in (("atT", T), ("Tplus1", T + pd.Timedelta(minutes=1))):
                if tx not in op.index:
                    continue
                pr = side * (op[tx] / e - 1)
                fund = -side * r.funding_rate          # long receives when rate < 0
                rows.append((base, T, H, ex, abs(r.prev), side, pr, fund, pr + fund - 2 * c))
    return pd.DataFrame(rows, columns=["base", "T", "H", "exit", "absprev", "side", "price", "fund", "net"])


if __name__ == "__main__":
    with ProcessPoolExecutor(16) as ex:
        df = pd.concat(list(ex.map(one, U162)))
    df.to_parquet(f"r11_{os.environ.get('SEG','TRAIN')}.parquet")
    days = (pd.Timestamp(SEG[1]) - pd.Timestamp(SEG[0])).days
    out = []
    for F, H, ex in CELLS:
        s = df[(df.absprev >= F) & (df.H == H) & (df.exit == ex)]
        dm = s.groupby(s["T"].dt.floor("D")).net.mean()
        g, b = s.net[s.net > 0].sum(), -s.net[s.net < 0].sum()
        out.append({"F": F, "H": H, "exit": ex, "n": len(s), "price_bp": s.price.mean() * 1e4, "fund_bp": s.fund.mean() * 1e4,
                    "net_bp": s.net.mean() * 1e4, "pf": g / b if b else np.nan, "days": len(dm),
                    "day_t": dm.mean() / dm.std() * np.sqrt(len(dm)), "per_day": len(s) / days,
                    "top_coin_share": s.groupby("base").net.sum().max() / s.net.sum() if s.net.sum() > 0 else np.nan})
    print(pd.DataFrame(out).round(3).to_string())
