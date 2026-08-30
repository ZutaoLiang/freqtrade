"""Rank the 1m perp universe by median daily quote volume (from 1d candles)."""
import glob
import os

import pandas as pd

DATA = "/root/freqtrade/user_data/data/binance_public/freqtrade/futures"
OUT = "/root/freqtrade/user_data/niche_work/universe.csv"
START = "2025-11-01"

rows = []
for f in sorted(glob.glob(f"{DATA}/*-1d-futures.feather")):
    sym = os.path.basename(f).replace("-1d-futures.feather", "")
    if not sym.endswith("_USDT_USDT"):
        continue
    d = pd.read_feather(f).set_index("date")
    d = d.loc[START:]
    if len(d) < 120:
        continue
    qv = (d["close"] * d["volume"])
    rows.append(dict(sym=sym, days=len(d), med_qv=qv.median(), min_qv=qv.quantile(0.1)))

u = pd.DataFrame(rows).sort_values("med_qv", ascending=False)
u.to_csv(OUT, index=False)
print(len(u))
print(u.head(15).to_string())
print(u[(u.med_qv > 3e6) & (u.med_qv < 6e7)].shape, "in niche band 3M-60M")
