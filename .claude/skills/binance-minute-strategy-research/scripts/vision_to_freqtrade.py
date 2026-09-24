#!/usr/bin/env python3
"""Convert a data.binance.vision mirror (download_vision.py) into a freqtrade futures datadir.

Output, per symbol (freqtrade 2026.x feather layout, candle types futures / funding_rate / mark):
  {datadir}/futures/{BASE}_USDT_USDT-{interval}-futures.feather      OHLCV from klines
  {datadir}/futures/{BASE}_USDT_USDT-{tf}-futures.feather            optional resamples (--resample)
  {datadir}/futures/{BASE}_USDT_USDT-1h-funding_rate.feather         settled funding, rate in `open`
  {datadir}/futures/{BASE}_USDT_USDT-1h-mark.feather                 1h mark-price OHLC
Schema: date = timestamp[ms, UTC]; open/high/low/close/volume = float64. Sorted, de-duplicated.

Why each rule exists (all were real failures):
  * Newer Vision CSVs have a header row, older ones don't -> sniff the first line.
  * Some archives stamp open_time in microseconds -> anything > 1e14 is divided by 1000.
  * Daily and monthly archives overlap -> de-duplicate on date, keep the last.
  * Funding calc_time carries ms jitter (e.g. 1759280400015) -> round to the hour, or freqtrade's
    inner join of funding with mark on `date` silently drops the row (funding = 0).
  * freqtrade 2026.x reads funding and mark at 1h (exchange.py funding_fee_timeframe /
    mark_ohlcv_timeframe). Files at any other timeframe are never read, silently.
  * Monthly mark archives sometimes miss a day -> reported; --mark-fallback fills the missing
    hours from the futures klines (last-price proxy) and counts how many it filled.

One symbol at a time, written to a temp file and renamed only after validation.
"""
from __future__ import annotations

import argparse
import gc
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
         "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
OHLCV = ["date", "open", "high", "low", "close", "volume"]
TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}


def read_zip(path: Path, names: list[str]) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(path) as zf:
            raw = zf.read(zf.namelist()[0])
    except zipfile.BadZipFile:
        print(f"  bad zip skipped: {path}", file=sys.stderr)
        return None
    has_header = not raw[:1].isdigit()
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, names=names,
                       usecols=range(len(names)))


def to_ms(values: pd.Series) -> pd.Series:
    v = values.astype("int64").to_numpy().copy()
    v[v > 10**14] //= 1000                       # microsecond archives
    return pd.Series(pd.to_datetime(v, unit="ms", utc=True).astype("datetime64[ms, UTC]"))


def klines_frame(files: list[Path]) -> pd.DataFrame:
    parts = []
    for f in files:
        df = read_zip(f, KCOLS)
        if df is None or df.empty:
            continue
        out = pd.DataFrame({"date": to_ms(df.open_time)})
        for c in ("open", "high", "low", "close", "volume"):
            out[c] = df[c].astype("float64").to_numpy()
        parts.append(out)
    if not parts:
        return pd.DataFrame(columns=OHLCV)
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    r = df.set_index("date").resample(f"{TF_MIN[tf]}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
    r = r.reset_index()
    r["date"] = r["date"].astype("datetime64[ms, UTC]")
    return r[OHLCV]


def write(df: pd.DataFrame, path: Path) -> None:
    df = df[OHLCV].copy()
    df["date"] = df["date"].astype("datetime64[ms, UTC]")
    for c in OHLCV[1:]:
        df[c] = df[c].astype("float64")
    assert df["date"].is_monotonic_increasing and not df["date"].duplicated().any(), path
    tmp = path.with_name(path.name + ".tmp")
    df.reset_index(drop=True).to_feather(tmp)
    tmp.replace(path)


def convert(sym: str, raw: Path, out: Path, interval: str, resamples: list[str], mark_fallback: bool) -> dict:
    um = raw / "data/futures/um"
    base = sym[:-4]
    pair = f"{base}_USDT_USDT"
    rep = {"symbol": sym}
    kfiles = sorted((um / "monthly/klines" / sym / interval).glob("*.zip")) + \
        sorted((um / "daily/klines" / sym / interval).glob("*.zip"))
    k = klines_frame(kfiles)
    if k.empty:
        rep["klines"] = 0
        return rep
    step = pd.Timedelta(minutes=TF_MIN[interval])
    gaps = int((k["date"].diff() > step).sum())
    write(k, out / f"{pair}-{interval}-futures.feather")
    rep.update(klines=len(k), first=str(k.date.iloc[0]), last=str(k.date.iloc[-1]), gaps=gaps)
    for tf in resamples:
        write(resample(k, tf), out / f"{pair}-{tf}-futures.feather")

    fcols = ["calc_time", "funding_interval_hours", "last_funding_rate"]
    ffiles = sorted((um / "monthly/fundingRate" / sym).glob("*.zip"))
    fparts = [f for f in (read_zip(x, fcols) for x in ffiles) if f is not None]
    # current / previous month from the REST fallback (download_vision.py), same columns
    fparts += [pd.read_csv(x) for x in sorted((um / "rest/fundingRate" / sym).glob("*.csv"))]
    if fparts:
        f = pd.concat(fparts, ignore_index=True)
        fr = pd.DataFrame({"date": to_ms(f.calc_time).dt.round("1h"), "open": f.last_funding_rate.astype("float64")})
        for c in ("high", "low", "close", "volume"):
            fr[c] = 0.0
        fr = fr.drop_duplicates("date", keep="last").sort_values("date")
        write(fr, out / f"{pair}-1h-funding_rate.feather")
        rep.update(funding=len(fr), funding_first=str(fr.date.iloc[0]))
    else:
        rep["funding"] = 0

    mfiles = sorted((um / "monthly/markPriceKlines" / sym / "1h").glob("*.zip")) + \
        sorted((um / "daily/markPriceKlines" / sym / "1h").glob("*.zip"))
    m = klines_frame(mfiles)
    m["volume"] = 0.0
    k1h = resample(k, "1h")
    missing = k1h.loc[~k1h.date.isin(m.date), "date"] if not m.empty else k1h.date
    rep["mark"] = len(m)
    rep["mark_missing_hours"] = int(len(missing))
    if mark_fallback and len(missing):
        fill = k1h[k1h.date.isin(missing)].assign(volume=0.0)
        m = pd.concat([m, fill], ignore_index=True).sort_values("date")
        rep["mark_filled_from_last_price"] = int(len(fill))
    if not m.empty:
        write(m, out / f"{pair}-1h-mark.feather")
    del k, m, k1h
    gc.collect()
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="user_data/data/binance-vision", help="download_vision.py mirror root")
    ap.add_argument("--datadir", default="user_data/data/binance", help="freqtrade datadir (futures/ is appended)")
    ap.add_argument("--symbols", default="", help="comma separated; default = every symbol found in the mirror")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--resample", default="", help="extra timeframes built from the klines, e.g. 5m,15m,1h")
    ap.add_argument("--mark-fallback", action="store_true", help="fill missing mark hours from last price")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.datadir) / "futures"
    out.mkdir(parents=True, exist_ok=True)
    syms = [s for s in a.symbols.split(",") if s] or sorted({
        p.name for freq in ("monthly", "daily")
        for p in (raw / f"data/futures/um/{freq}/klines").glob("*USDT") if p.is_dir()})
    res = [t for t in a.resample.split(",") if t]
    for i, s in enumerate(syms, 1):
        print(f"[{i}/{len(syms)}]", convert(s, raw, out, a.interval, res, a.mark_fallback), flush=True)


if __name__ == "__main__":
    main()
