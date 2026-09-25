"""bookDepth daily zips -> per-minute depth features (parquet cache per symbol).

Each snapshot (~every 30-60 s) gives cumulative depth notional within +-1..5% of mid.
Per minute we keep the LAST snapshot whose timestamp falls inside that minute, stamped with the
minute's open time, so it is known at that 1m bar's close (same convention as klines).
"""
from __future__ import annotations

import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

RAW = "/root/freqtrade/user_data/data/r3raw/depth"
CACHE = "/root/freqtrade/user_data/data/r3raw/depth_parquet"


def _day(path: str) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(path) as z:
            raw = z.read(z.namelist()[0])
    except zipfile.BadZipFile:
        return None
    d = pd.read_csv(io.BytesIO(raw))
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    w = d.pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="last")
    w = w[[c for c in w.columns if float(c) in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)]]
    w.columns = [f"n{int(float(c)):+d}" for c in w.columns]
    w["minute"] = w.index.floor("min")
    w = w.groupby("minute").last()
    return w


def build(sym: str) -> pd.DataFrame | None:
    """sym like BTCUSDT. Rebuilds the cache when new zips exist."""
    files = sorted(glob.glob(f"{RAW}/{sym}/*.zip"))
    cp = f"{CACHE}/{sym}.parquet"
    if not files:
        return None
    if os.path.exists(cp) and os.path.getmtime(cp) > max(os.path.getmtime(f) for f in files):
        return pd.read_parquet(cp)
    parts = [x for x in (_day(f) for f in files) if x is not None]
    d = pd.concat(parts).sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d = d.astype("float32")
    os.makedirs(CACHE, exist_ok=True)
    d.to_parquet(cp)
    return d


def features(dep: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align to 1m bar index (forward-fill at most 5 minutes). imbK = (bid - ask)/(bid + ask) within K%."""
    x = dep.reindex(index, method="ffill", limit=5)
    f = pd.DataFrame(index=index)
    for k in (1, 2, 5):
        b, a = x[f"n-{k}"], x[f"n+{k}"]
        f[f"imb{k}"] = (b - a) / (b + a)
        f[f"tot{k}"] = b + a
    return f
