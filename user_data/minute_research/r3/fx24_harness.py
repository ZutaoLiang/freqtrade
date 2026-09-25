"""Frozen harness replica of R24 (session freqtrade-1c): 3 settlements >= F -> short at T+5m, hold H min. No funding."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import r3lib as L

B = "/root/freqtrade/user_data/data/binance_public"
U = pd.read_csv("universe_rank.csv"); U162 = U[U.med_qv >= 1e7].base.tolist()


def one(args):
    base, start, end, F, hold = args
    d = pd.read_parquet(f"{B}/klines_1m/{base}USDT.parquet", columns=["date", "open", "high", "low", "close", "volume"])
    d = d[(d.date >= start) & (d.date < end)]
    if len(d) < 1000:
        return base, None
    k = d.set_index("date").resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    f = pd.read_parquet(f"{B}/funding/{base}USDT.parquet"); f["date"] = f.date.dt.round("h")
    f = f.drop_duplicates("date", keep="last").set_index("date").funding_rate.sort_index()
    ok = (f >= F) & (f.shift(1) >= F) & (f.shift(2) >= F)
    T = f.index[ok]
    idx = k.index.get_indexer(T); idx = idx[idx >= 0]
    dd = {c: k[c].to_numpy() for c in ("open", "high", "low", "close", "volume")}; dd["date"] = k.index.to_series()
    tr = L.H.run(dd, idx, -1, hold=hold // 5, cost_bps=10.0)
    tr["base"] = base
    return base, tr


def run(start, end, F=0.0003, hold=480):
    with ProcessPoolExecutor(16) as ex:
        out = [t for _, t in ex.map(one, [(b, start, end, F, hold) for b in U162]) if t is not None and len(t)]
    return pd.concat(out, ignore_index=True)
