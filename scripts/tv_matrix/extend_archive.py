"""Extend the local Binance USD-M archive with new daily data.

Appends klines, mark-price klines and funding rates for a date range to the
existing `user_data/data/binance_public` archive, then rebuilds the freqtrade
feather files (1m plus every aggregated timeframe) from the updated 1m data.

Sources, all from the official public archive, each ZIP verified against its
published SHA-256 CHECKSUM:

  klines      data/futures/um/daily/klines/<SYM>/1m/<SYM>-1m-<date>.zip
  mark price  data/futures/um/daily/markPriceKlines/<SYM>/1m/<SYM>-1m-<date>.zip
  funding     data/futures/um/monthly/fundingRate/<SYM>/<SYM>-fundingRate-<YYYY-MM>.zip

Funding rates for a month whose monthly archive is not published yet are
fetched from the REST endpoint instead; the settlement interval is inferred
from the spacing of the stamps, since REST does not return it.

A symbol that 404s for a date is treated as not listed on that date (delisted
or listed later) and is skipped, not failed. Symbols that appeared after the
archive was last built are reported but not backfilled - they cannot satisfy
the matrix warmup requirement anyway.

Resumable: valid ZIPs already on disk are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("user_data/data/binance_public")
RAW = ROOT / "raw"
RAW_MARK = ROOT / "raw_aux" / "markPrice"
RAW_FUND = ROOT / "raw_aux" / "fundingRate"
KLINES = ROOT / "klines_1m"
MARKP = ROOT / "markprice_1m"
FUNDING = ROOT / "funding"
FEATHER = ROOT / "freqtrade" / "futures"
BASE = "https://data.binance.vision/data/futures/um"
REST = "https://fapi.binance.com/fapi/v1"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
TIMEFRAMES = {
    "5m": ("5min", 5),
    "15m": ("15min", 15),
    "30m": ("30min", 30),
    "1h": ("1h", 60),
    "4h": ("4h", 240),
    "1d": ("1D", 1440),
}


_local = threading.local()


def session() -> requests.Session:
    """One pooled session per thread: the proxy hop is expensive to re-open."""
    if not hasattr(_local, "s"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
        s.mount("https://", adapter)
        _local.s = s
    return _local.s


def fetch_zip(url: str, dest: Path) -> str:
    """Download and checksum-verify one archive. Returns ok / skip / missing."""
    if dest.exists() and dest.stat().st_size > 0:
        try:
            with zipfile.ZipFile(dest) as z:
                if z.testzip() is None:
                    return "skip"
        except zipfile.BadZipFile:
            dest.unlink()

    r = session().get(url, timeout=60)
    if r.status_code == 404:
        return "missing"
    r.raise_for_status()
    c = session().get(url + ".CHECKSUM", timeout=60)
    if c.status_code == 200:
        expected = c.text.split()[0]
        if hashlib.sha256(r.content).hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.rename(dest)
    return "ok"


def read_kline_zip(path: Path, cols: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0]).decode()
    first = raw.split("\n", 1)[0].split(",")[0]
    header = 0 if not first.replace(".", "").isdigit() else None
    df = pd.read_csv(io.StringIO(raw), header=header, names=None if header == 0 else KLINE_COLS)
    df = df.rename(columns={"open_time": "open_time"})
    out = pd.DataFrame({"date": pd.to_datetime(df["open_time"], unit="ms", utc=True)})
    for c in cols:
        out[c] = df[c].astype("float64")
    return out


def build_download_jobs(symbols: list[str], dates: list[str], months: list[str]) -> list[tuple]:
    """One job per file so the pool is not serialised behind a slow symbol."""
    jobs = []
    for s in symbols:
        for d in dates:
            jobs.append(("klines", f"{BASE}/daily/klines/{s}/1m/{s}-1m-{d}.zip",
                         RAW / s / f"{s}-1m-{d}.zip"))
            jobs.append(("mark", f"{BASE}/daily/markPriceKlines/{s}/1m/{s}-1m-{d}.zip",
                         RAW_MARK / s / f"{s}-1m-{d}.zip"))
        for m in months:
            jobs.append(("funding", f"{BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip",
                         RAW_FUND / s / f"{s}-fundingRate-{m}.zip"))
    return jobs


def download_one(job) -> tuple[str, str]:
    kind, url, dest = job
    try:
        return kind, fetch_zip(url, dest)
    except Exception:
        # One transient failure must not abort the run; retry once, then record.
        try:
            return kind, fetch_zip(url, dest)
        except Exception:
            return kind, "error"


def rest_funding(symbol: str, start_ms: int) -> pd.DataFrame:
    rows, cursor = [], start_ms
    while True:
        r = requests.get(
            f"{REST}/fundingRate",
            params={"symbol": symbol, "startTime": cursor, "limit": 1000},
            timeout=60,
        )
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        cursor = int(data[-1]["fundingTime"]) + 1
    if not rows:
        return pd.DataFrame(columns=["date", "funding_rate", "funding_interval_hours"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.round("min")
    df["funding_rate"] = df["fundingRate"].astype("float64")
    gaps = df["date"].diff().dt.total_seconds() / 3600
    # REST does not return the interval; infer it from the settlement spacing.
    df["funding_interval_hours"] = gaps.bfill().ffill().round().astype("float64")
    return df[["date", "funding_rate", "funding_interval_hours"]]


def convert_symbol(job) -> dict:
    symbol, dates, months, rest_start_ms = job
    base = symbol.removesuffix("USDT")
    res = {"symbol": symbol, "klines_added": 0, "mark_added": 0, "funding_added": 0}

    # --- 1m klines ---
    kpaths = [RAW / symbol / f"{symbol}-1m-{d}.zip" for d in dates]
    kpaths = [p for p in kpaths if p.exists()]
    kl_path = KLINES / f"{symbol}.parquet"
    if kpaths and kl_path.exists():
        old = pd.read_parquet(kl_path)
        cols = ["open", "high", "low", "close", "volume", "quote_volume", "count",
                "taker_buy_volume", "taker_buy_quote_volume"]
        new = pd.concat([read_kline_zip(p, cols) for p in kpaths], ignore_index=True)
        new = new[new["date"] > old["date"].max()]
        if len(new):
            merged = pd.concat([old, new[old.columns]], ignore_index=True)
            merged = merged.drop_duplicates("date").sort_values("date").reset_index(drop=True)
            merged.to_parquet(kl_path, index=False)
            res["klines_added"] = len(new)

    # --- mark price ---
    mpaths = [RAW_MARK / symbol / f"{symbol}-1m-{d}.zip" for d in dates]
    mpaths = [p for p in mpaths if p.exists()]
    mp_path = MARKP / f"{symbol}.parquet"
    if mpaths and mp_path.exists():
        old = pd.read_parquet(mp_path)
        new = pd.concat(
            [read_kline_zip(p, ["open", "high", "low", "close"]) for p in mpaths],
            ignore_index=True,
        ).rename(columns={"open": "mark_open", "high": "mark_high",
                          "low": "mark_low", "close": "mark_close"})
        new = new[new["date"] > old["date"].max()]
        if len(new):
            merged = pd.concat([old, new[old.columns]], ignore_index=True)
            merged = merged.drop_duplicates("date").sort_values("date").reset_index(drop=True)
            merged.to_parquet(mp_path, index=False)
            res["mark_added"] = len(new)

    # --- funding: monthly archives, then REST for the unpublished tail ---
    f_path = FUNDING / f"{symbol}.parquet"
    if f_path.exists():
        old = pd.read_parquet(f_path)
        frames = []
        for m in months:
            p = RAW_FUND / symbol / f"{symbol}-fundingRate-{m}.zip"
            if not p.exists():
                continue
            with zipfile.ZipFile(p) as z:
                raw = z.read(z.namelist()[0]).decode()
            df = pd.read_csv(io.StringIO(raw))
            frames.append(pd.DataFrame({
                "date": pd.to_datetime(df["calc_time"], unit="ms", utc=True).dt.round("min"),
                "funding_rate": df["last_funding_rate"].astype("float64"),
                "funding_interval_hours": df["funding_interval_hours"].astype("float64"),
            }))
        if rest_start_ms:
            try:
                frames.append(rest_funding(symbol, rest_start_ms))
            except Exception:
                pass
        if frames:
            new = pd.concat(frames, ignore_index=True)
            new = new[new["date"] > old["date"].max()]
            if len(new):
                merged = pd.concat([old, new[old.columns]], ignore_index=True)
                merged = merged.drop_duplicates("date").sort_values("date").reset_index(drop=True)
                merged.to_parquet(f_path, index=False)
                res["funding_added"] = len(new)

    # --- rebuild freqtrade feathers from the full 1m series ---
    if res["klines_added"] and kl_path.exists():
        minute = pd.read_parquet(kl_path)[["date", "open", "high", "low", "close", "volume"]]
        minute["date"] = minute["date"].astype("datetime64[ms, UTC]")
        prefix = FEATHER / f"{base}_USDT_USDT"
        minute.reset_index(drop=True).to_feather(
            prefix.with_name(prefix.name + "-1m-futures.feather"), compression="lz4"
        )
        indexed = minute.set_index("date")
        for tf, (rule, expected) in TIMEFRAMES.items():
            grouped = indexed.resample(rule, label="left", closed="left")
            sizes = grouped.size()
            out = grouped.agg(AGG)
            # Only complete buckets, matching how the archive was originally built.
            out = out.loc[sizes == expected].dropna(subset=["open", "high", "low", "close"])
            out = out.reset_index()[["date", "open", "high", "low", "close", "volume"]]
            out["date"] = out["date"].astype("datetime64[ms, UTC]")
            out.to_feather(
                prefix.with_name(prefix.name + f"-{tf}-futures.feather"), compression="lz4"
            )
        res["last_1m"] = str(minute["date"].iloc[-1])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-08-17")
    ap.add_argument("--end", default="2026-09-04")
    ap.add_argument("--months", default="2026-08", help="monthly fundingRate archives to pull")
    ap.add_argument("--rest-funding-from", default="2026-09-01")
    ap.add_argument("--download-workers", type=int, default=24)
    ap.add_argument("--convert-workers", type=int, default=12)
    ap.add_argument("--phase", choices=["download", "convert", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N symbols")
    args = ap.parse_args()

    dates = [str(d.date()) for d in pd.date_range(args.start, args.end, freq="D")]
    months = [m for m in args.months.split(",") if m]
    symbols = sorted(p.stem for p in KLINES.glob("*.parquet"))
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"{len(symbols)} symbols, {len(dates)} days ({dates[0]} .. {dates[-1]})", flush=True)

    if args.phase in ("download", "all"):
        jobs = build_download_jobs(symbols, dates, months)
        totals = {"ok": 0, "skip": 0, "missing": 0, "error": 0}
        errors = []
        with ThreadPoolExecutor(max_workers=args.download_workers) as ex:
            futs = [ex.submit(download_one, j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                kind, res = f.result()
                totals[res] = totals.get(res, 0) + 1
                if res == "error":
                    errors.append(kind)
                if i % 5000 == 0:
                    print(f"[dl {i}/{len(jobs)}] {totals}", flush=True)
        print(f"download done: {len(jobs)} files {totals}", flush=True)
        if errors:
            print(f"WARNING: {len(errors)} files failed after retry", flush=True)

    if args.phase in ("convert", "all"):
        rest_ms = int(pd.Timestamp(args.rest_funding_from, tz="UTC").timestamp() * 1000)
        jobs = [(s, dates, months, rest_ms) for s in symbols]
        results = []
        with ProcessPoolExecutor(max_workers=args.convert_workers) as ex:
            for i, r in enumerate(ex.map(convert_symbol, jobs, chunksize=2), 1):
                results.append(r)
                if i % 100 == 0:
                    print(f"[cv {i}/{len(symbols)}]", flush=True)
        rep = pd.DataFrame(results)
        rep.to_csv("user_data/research/tradingview/archive_extend_report.csv", index=False)
        print(f"converted symbols with new klines: {int((rep['klines_added'] > 0).sum())}")
        print(f"rows added: klines {int(rep['klines_added'].sum())}, "
              f"mark {int(rep['mark_added'].sum())}, funding {int(rep['funding_added'].sum())}")
        print(json.dumps(rep["last_1m"].dropna().value_counts().head(5).to_dict(), indent=1))


if __name__ == "__main__":
    main()
