#!/usr/bin/env python3
"""Backfill 2023-2025 1h history for every USDT perp that ever mattered.

Two phases:

* ``screen``: enumerate all USDT perpetuals (listed + delisted, same union
  as download_all_binance_usdt_perp_1m.py), pull daily klines from the fapi
  API (archive fallback for symbols the API no longer serves), and shortlist
  the symbols whose trailing 7d mean of 24h quote volume ever exceeded
  SCREEN_QVOL between 2023-01-01 and 2025-12-31.  Writes the shortlist and
  the per-symbol peak to user_data/data/binance-hist/screen.json.
* ``download``: for shortlisted symbols, fetch monthly archive zips of 1h
  klines, 1h mark-price klines and funding rates for 2022-11..2025-12 (two
  months of warmup before 2023), and write freqtrade-format feathers into
  user_data/data/binance-hist/futures/.  Resumable: symbols with an existing
  klines feather are skipped.

Serial, memory-light (one symbol at a time), retries per request.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_all_binance_usdt_perp_1m import (  # noqa: E402
    CSV_COLUMNS, DOWNLOAD_BASE, crypto_usdt_perpetuals, list_objects, request,
)

ROOT = Path("user_data/data/binance-hist")
FUT = ROOT / "futures"
SCREEN_JSON = ROOT / "screen.json"
SCREEN_QVOL = 25e6
SCREEN_START = pd.Timestamp("2023-01-01", tz="UTC")
SCREEN_END = pd.Timestamp("2025-12-31", tz="UTC")
MONTHS = pd.period_range("2022-11", "2025-12", freq="M")


def api_daily(symbol: str) -> pd.DataFrame:
    rows = []
    start = int(SCREEN_START.timestamp() * 1000)
    end = int(SCREEN_END.timestamp() * 1000)
    while start < end:
        try:
            r = request("https://fapi.binance.com/fapi/v1/klines",
                        params={"symbol": symbol, "interval": "1d",
                                "startTime": str(start), "endTime": str(end), "limit": "1000"})
        except Exception:
            return pd.DataFrame()
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        start = batch[-1][0] + 86_400_000
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).iloc[:, [0, 7]]
    df.columns = ["open_time", "quote_volume"]
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["quote_volume"] = df["quote_volume"].astype(float)
    return df[["date", "quote_volume"]]


def archive_zip(path: str) -> pd.DataFrame | None:
    try:
        r = request(f"{DOWNLOAD_BASE}/{path}")
    except Exception:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if isinstance(df.iloc[0, 0], str) and not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    return df


def archive_months(symbol: str, kind: str, interval: str | None) -> list[str]:
    """Keys actually present in the archive for this symbol/kind (one listing)."""
    if kind == "fundingRate":
        prefix = f"data/futures/um/monthly/fundingRate/{symbol}/"
    else:
        prefix = f"data/futures/um/monthly/{kind}/{symbol}/{interval}/"
    return [o["key"] for o in list_objects(prefix) if o["key"].endswith(".zip")]


def archive_daily(symbol: str) -> pd.DataFrame:
    frames = []
    keys = archive_months(symbol, "klines", "1d")
    wanted = {f"{m.year}-{m.month:02d}" for m in pd.period_range("2023-01", "2025-12", freq="M")}
    for key in keys:
        if key[-11:-4] not in wanted:
            continue
        df = archive_zip(key)
        if df is None or df.empty:
            continue
        df = df.iloc[:, [0, 7]]
        df.columns = ["open_time", "quote_volume"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
    df["quote_volume"] = df["quote_volume"].astype(float)
    return df[["date", "quote_volume"]]


def screen() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    symbols = crypto_usdt_perpetuals()
    print(f"universe: {len(symbols)} symbols")
    done: dict[str, float] = {}
    if SCREEN_JSON.exists():
        done = json.loads(SCREEN_JSON.read_text()).get("peaks", {})
    for i, sym in enumerate(symbols):
        if sym in done:
            continue
        df = api_daily(sym)
        if df.empty:
            df = archive_daily(sym)
        if df.empty:
            done[sym] = -1.0
        else:
            df = df[(df["date"] >= SCREEN_START) & (df["date"] <= SCREEN_END)]
            peak = df["quote_volume"].rolling(7, min_periods=4).mean().max()
            done[sym] = float(peak) if pd.notna(peak) else -1.0
        if i % 25 == 0 or i == len(symbols) - 1:
            SCREEN_JSON.write_text(json.dumps(
                {"peaks": done,
                 "shortlist": sorted(s for s, v in done.items() if v > SCREEN_QVOL)}))
            print(f"  {i + 1}/{len(symbols)} screened, "
                  f"shortlist {sum(1 for v in done.values() if v > SCREEN_QVOL)}")
    shortlist = sorted(s for s, v in done.items() if v > SCREEN_QVOL)
    SCREEN_JSON.write_text(json.dumps({"peaks": done, "shortlist": shortlist}))
    print(f"screen done: {len(shortlist)} symbols above {SCREEN_QVOL/1e6:.0f}M")


def month_frames(symbol: str, kind: str) -> pd.DataFrame:
    frames = []
    wanted = {f"{m.year}-{m.month:02d}" for m in MONTHS}
    for key in archive_months(symbol, kind, "1h"):
        if key[-11:-4] not in wanted:
            continue
        df = archive_zip(key)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.iloc[:, :len(CSV_COLUMNS)]
    df.columns = CSV_COLUMNS[: df.shape[1]]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)})
    # 2025+ archives stamp microseconds; normalise
    bad = out["date"] > pd.Timestamp("2100-01-01", tz="UTC")
    if bad.any():
        out.loc[bad, "date"] = pd.to_datetime(
            df.loc[bad, "open_time"].astype(np.int64), unit="us", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = df[c].astype(float)
    out = out.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return out


def fetch_symbol(sym: str) -> str:
    base = sym[: -len("USDT")]
    stem = f"{base}_USDT_USDT"
    kl_path = FUT / f"{stem}-1h-futures.feather"
    if kl_path.exists():
        return f"{sym}: exists"
    kl = month_frames(sym, "klines")
    if kl.empty:
        return f"{sym}: no archive klines"
    mk = month_frames(sym, "markPriceKlines")
    fr = month_frames(sym, "fundingRate")
    if not mk.empty:
        to_ohlcv(mk).to_feather(FUT / f"{stem}-1h-mark.feather")
    if not fr.empty:
        fr = fr.iloc[:, :3]
        fr.columns = ["calc_time", "funding_interval_hours", "rate"]
        out = pd.DataFrame({
            "date": pd.to_datetime(fr["calc_time"].astype(np.int64), unit="ms", utc=True),
            "open": fr["rate"].astype(float),
            "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0})
        bad = out["date"] > pd.Timestamp("2100-01-01", tz="UTC")
        if bad.any():
            out.loc[bad, "date"] = pd.to_datetime(
                fr.loc[bad, "calc_time"].astype(np.int64), unit="us", utc=True)
        out = out.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        out.to_feather(FUT / f"{stem}-1h-funding_rate.feather")
    # klines last: their existence marks the symbol complete for resume
    to_ohlcv(kl).to_feather(kl_path)
    return f"{sym}: rows={len(kl)}"


def download() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    FUT.mkdir(parents=True, exist_ok=True)
    shortlist = json.loads(SCREEN_JSON.read_text())["shortlist"]
    print(f"downloading 1h+mark+funding for {len(shortlist)} symbols")
    done = 0
    # network-bound, tiny frames per symbol; 6 workers stays far below the
    # host memory budget in AGENTS.md
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(fetch_symbol, s): s for s in shortlist}
        for fut in as_completed(futs):
            done += 1
            try:
                msg = fut.result()
            except Exception as exc:
                msg = f"{futs[fut]}: ERROR {exc}"
            print(f"  [{done}/{len(shortlist)}] {msg}", flush=True)


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if phase == "screen":
        screen()
    elif phase == "download":
        download()
    else:
        raise SystemExit("phase must be screen or download")
