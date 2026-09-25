"""Spot 5m klines from Binance Vision zips -> parquet cache; hourly spot/perp flow panel."""
from __future__ import annotations

import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

RAW = "/root/freqtrade/user_data/data/r3raw/spot5"
CACHE = "/root/freqtrade/user_data/data/r3raw/spot5_parquet"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def spot_symbol(base: str) -> str:
    return (base[4:] if base.startswith("1000") else base) + "USDT"


def load_spot(base: str) -> pd.DataFrame | None:
    sym = spot_symbol(base)
    cp = f"{CACHE}/{sym}.parquet"
    if os.path.exists(cp):
        return pd.read_parquet(cp)
    files = sorted(glob.glob(f"{RAW}/{sym}/*.zip"))
    if not files:
        return None
    parts = []
    for f in files:
        with zipfile.ZipFile(f) as z:
            raw = z.read(z.namelist()[0])
        df = pd.read_csv(io.BytesIO(raw), header=0 if not raw[:1].isdigit() else None, names=COLS)
        parts.append(df)
    d = pd.concat(parts)
    t = d.open_time.astype("int64").to_numpy().copy()
    t[t > 10**14] //= 1000
    d["date"] = pd.to_datetime(t, unit="ms", utc=True)
    d = d.drop_duplicates("date").sort_values("date")[["date", "open", "close", "quote_volume", "count",
                                                          "taker_buy_quote_volume"]].reset_index(drop=True)
    os.makedirs(CACHE, exist_ok=True)
    d.to_parquet(cp)
    return d


def hourly_panel(base: str, perp: dict) -> pd.DataFrame | None:
    """Hourly frame indexed by hour start: perp o/c/qv/tb, spot qv/tb, and the last 1m index of each hour."""
    s = load_spot(base)
    if s is None:
        return None
    p = pd.DataFrame({"o": perp["open"], "c": perp["close"], "qv": perp["quote_volume"],
                      "tb": perp["taker_buy_quote_volume"], "i": np.arange(len(perp["open"]))},
                     index=pd.DatetimeIndex(perp["date"]))
    ph = p.resample("1h").agg({"o": "first", "c": "last", "qv": "sum", "tb": "sum", "i": "last"})
    sh = s.set_index("date").resample("1h").agg({"quote_volume": "sum", "taker_buy_quote_volume": "sum"})
    sh.columns = ["sqv", "stb"]
    h = ph.join(sh, how="left").dropna(subset=["o", "c"])
    h[["sqv", "stb"]] = h[["sqv", "stb"]].fillna(0.0)
    return h
