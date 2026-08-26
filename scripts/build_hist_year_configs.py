#!/usr/bin/env python3
"""Generate per-year whitelists + configs for the 2023-2025 expanded backtests.

For each year: whitelist = pairs in user_data/data/binance-hist/futures whose
trailing 7d mean of 24h quote volume crossed MIN_QVOL inside the year window
(with warmup margin), that have funding+mark feathers, AND that still exist on
Binance futures today (delisted contracts lack market metadata and crash the
backtest -- a known, unavoidable exclusion; counts are printed and must be
disclosed with results).

Writes user_data/config_xs_mom_ensemble_v1_hist_<year>.json.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd
import requests

FUT = "user_data/data/binance-hist/futures"
MIN_QVOL = 30e6
BASE_CONFIG = "user_data/config_xs_mom_ensemble_v1_2026.json"
YEARS = {
    "2023": ("2022-11-20", "2024-01-01"),
    "2024": ("2023-11-20", "2025-01-01"),
    "2025": ("2024-11-20", "2026-01-01"),
}


def listed_symbols() -> set[str]:
    info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=30).json()
    return {s["symbol"] for s in info["symbols"]
            if s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL"}


def main() -> None:
    listed = listed_symbols()
    base = json.load(open(BASE_CONFIG))
    files = sorted(glob.glob(f"{FUT}/*-1h-futures.feather"))
    print(f"feathers: {len(files)}")
    per_year: dict[str, list[str]] = {y: [] for y in YEARS}
    excluded_delisted: dict[str, int] = {y: 0 for y in YEARS}
    for f in files:
        stem = os.path.basename(f).split("-1h-")[0]
        sym = stem.replace("_USDT_USDT", "") + "USDT"
        pair = stem.replace("_USDT_USDT", "") + "/USDT:USDT"
        has_aux = (os.path.exists(f"{FUT}/{stem}-1h-funding_rate.feather")
                   and os.path.exists(f"{FUT}/{stem}-1h-mark.feather"))
        if not has_aux:
            continue
        try:
            df = pd.read_feather(f, columns=["date", "close", "volume"])
        except Exception:
            continue
        if df.empty:
            continue
        qv = (df["close"] * df["volume"]).rolling(24, min_periods=12).sum()
        qtrail = qv.rolling(168, min_periods=84).mean()
        df = df.assign(qtrail=qtrail)
        for y, (a, b) in YEARS.items():
            seg = df[(df["date"] >= pd.Timestamp(a, tz="UTC"))
                     & (df["date"] < pd.Timestamp(b, tz="UTC"))]
            if seg.empty or not (seg["qtrail"] > MIN_QVOL).any():
                continue
            if sym not in listed:
                excluded_delisted[y] += 1
                continue
            per_year[y].append(pair)
    for y, (a, b) in YEARS.items():
        wl = sorted(per_year[y])
        c = dict(base)
        c["exchange"] = dict(base["exchange"], pair_whitelist=wl, pair_blacklist=[])
        c["bot_name"] = f"xs-mom-ensemble-v1-hist-{y}"
        fn = f"user_data/config_xs_mom_ensemble_v1_hist_{y}.json"
        json.dump(c, open(fn, "w"), indent=1)
        print(f"{y}: whitelist={len(wl)} excluded_delisted={excluded_delisted[y]} -> {fn}")


if __name__ == "__main__":
    main()
