#!/usr/bin/env python3
"""Download long-history daily spot klines for the majors from Binance REST.

Purpose: extend the regime-indicator study back to 2018 (several full
bull/bear/chop cycles) — see analyze_regime_indicator_longterm.py.
Output: one feather per symbol under user_data/data/binance-spot-daily/.
Tiny volume (~3k rows/pair), serial requests, no API key needed.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path("user_data/data/binance-spot-daily")
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT",
    "DOGEUSDT", "TRXUSDT", "LTCUSDT", "LINKUSDT", "BCHUSDT", "XLMUSDT",
    "DOTUSDT", "AVAXUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT",
    "NEARUSDT", "HBARUSDT",
]
START_MS = int(pd.Timestamp("2018-01-01", tz="UTC").timestamp() * 1000)
END_MS = int(pd.Timestamp("2026-08-25", tz="UTC").timestamp() * 1000)


def fetch(symbol: str) -> pd.DataFrame:
    rows = []
    start = START_MS
    while start < END_MS:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "startTime": start,
                    "endTime": END_MS, "limit": 1000},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        start = batch[-1][0] + 86_400_000
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "tbb", "tbq", "ignore"])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = df[c].astype(float)
    return df[["date", "open", "high", "low", "close", "volume", "quote_volume"]]


def main() -> None:
    for sym in SYMBOLS:
        path = OUT / f"{sym}-1d.feather"
        if path.exists():
            print(f"{sym}: exists, skip")
            continue
        df = fetch(sym)
        if df.empty:
            print(f"{sym}: no data")
            continue
        df.reset_index(drop=True).to_feather(path)
        print(f"{sym}: {len(df)} rows {df['date'].min().date()}..{df['date'].max().date()}")


if __name__ == "__main__":
    main()
