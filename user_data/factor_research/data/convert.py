"""Convert downloaded Binance 1m kline zips into one parquet per symbol.

Keeps every column Binance publishes, not just OHLCV: quote volume, trade
count and taker-buy volume are inputs for later factor work, and re-downloading
14 GiB to recover them later would be wasteful.
"""
import io
import os
import sys
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import MAX_WORKERS  # noqa: E402

RAW = "/root/freqtrade/user_data/data/binance_public/raw"
OUT = "/root/freqtrade/user_data/data/binance_public/klines_1m"
START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2026-08-17", tz="UTC")  # exclusive

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
KEEP = ["date", "open", "high", "low", "close", "volume",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    head = raw[:64].split(b"\n", 1)[0]
    skip = 1 if b"open_time" in head else 0
    return pd.read_csv(io.BytesIO(raw), header=None, names=COLS, skiprows=skip)


def to_utc(series):
    """Binance switched futures timestamps from ms to us partway through 2025.

    Decided per value rather than per file, so a symbol whose months straddle
    the switch still lands on the right instant.
    """
    v = series.astype("int64")
    v = np.where(v > 1e14, v // 1000, v)
    return pd.to_datetime(v, unit="ms", utc=True)


def convert(sym):
    files = sorted(f for f in os.listdir(os.path.join(RAW, sym)) if f.endswith(".zip"))
    if not files:
        return sym, 0, "no archives"
    try:
        df = pd.concat([read_zip(os.path.join(RAW, sym, f)) for f in files],
                       ignore_index=True)
        df["date"] = to_utc(df["open_time"])
        df = df[KEEP]
        df = df[(df["date"] >= START) & (df["date"] < END)]
        df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
        for c in KEEP[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        os.makedirs(OUT, exist_ok=True)
        df.to_parquet(os.path.join(OUT, f"{sym}.parquet"), compression="zstd", index=False)
        return sym, len(df), None
    except Exception as exc:
        return sym, 0, repr(exc)


if __name__ == "__main__":
    syms = sorted(d for d in os.listdir(RAW) if os.path.isdir(os.path.join(RAW, d)))
    print(f"converting {len(syms)} symbols ...", flush=True)
    ok = bad = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, (sym, n, err) in enumerate(ex.map(convert, syms), 1):
            if err:
                bad += 1
                sys.stderr.write(f"FAIL {sym}: {err}\n")
            else:
                ok += 1
            if i % 100 == 0:
                print(f"  {i}/{len(syms)}  ok={ok} bad={bad}", flush=True)
    print(f"done: ok={ok} bad={bad}")
