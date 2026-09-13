"""Fetch funding rate and mark-price data for the same universe and range.

Funding rate is the crypto-specific factor input with no equivalent in generic
factor libraries (positioning crowding), and mark price gives the basis once
differenced against close. Mark price is taken at 1m for the same reason the
klines are: one base layer, resampled up, so every timeframe shares coverage.

Stages are manifest -> download -> convert, each resumable, run one at a time.
"""
import io
import json
import os
import socket
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from config import DOWNLOAD_THREADS, MAX_WORKERS  # noqa: E402

META = os.path.join(HERE, "..", "meta")
ROOT = "/root/freqtrade/user_data/data/binance_public"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CDN = "https://data.binance.vision/"

MONTHS = {f"{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13)}
MONTHS = {m for m in MONTHS if "2025-01" <= m <= "2026-07"}
DAYS = {f"2026-08-{d:02d}" for d in range(1, 17)}

DATASETS = {
    "fundingRate": {"path": "fundingRate", "interval": None},
    "markPrice": {"path": "markPriceKlines", "interval": "1m"},
}

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]

socket.setdefaulttimeout(60)


def _listing_opener():
    """Listing host is only reachable through the shell proxy; keep it."""
    return urllib.request.build_opener()


def _cdn_opener():
    """The CDN answers directly ~80x faster than through the proxy tunnel."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def list_prefix(prefix, opener):
    out, marker = [], ""
    while True:
        url = f"{BUCKET}?prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if marker:
            url += "&marker=" + urllib.parse.quote(marker)
        for attempt in range(4):
            try:
                x = ET.fromstring(opener.open(url, timeout=30).read())
                break
            except Exception:
                if attempt == 3:
                    raise
        for c in x.findall("s:Contents", NS):
            out.append((c.findtext("s:Key", namespaces=NS),
                        int(c.findtext("s:Size", namespaces=NS))))
        if x.findtext("s:IsTruncated", default="false", namespaces=NS) != "true":
            return out
        marker = out[-1][0]


def scan(args):
    sym, ds = args
    spec = DATASETS[ds]
    opener = _listing_opener()
    items = []
    for kind, wanted in (("monthly", MONTHS), ("daily", DAYS)):
        base = f"data/futures/um/{kind}/{spec['path']}/{sym}/"
        if spec["interval"]:
            base += f"{spec['interval']}/"
        try:
            entries = list_prefix(base, opener)
        except Exception:
            continue
        for key, size in entries:
            if not key.endswith(".zip"):
                continue
            # Filenames end in the period: ...-YYYY-MM.zip or ...-YYYY-MM-DD.zip
            parts = os.path.basename(key)[:-4].split("-")
            stamp = "-".join(parts[-2:]) if kind == "monthly" else "-".join(parts[-3:])
            if stamp in wanted:
                items.append({"sym": sym, "ds": ds, "key": key, "size": size})
    return items


def build_manifest(symbols):
    jobs = [(s, ds) for ds in DATASETS for s in symbols]
    print(f"scanning {len(jobs)} symbol/dataset pairs ...", flush=True)
    manifest = []
    with ThreadPoolExecutor(max_workers=48) as ex:
        for i, items in enumerate(ex.map(scan, jobs), 1):
            manifest.extend(items)
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}  files: {len(manifest)}", flush=True)
    path = os.path.join(META, "manifest_aux.json")
    with open(path, "w") as f:
        json.dump(manifest, f)
    by_ds = {}
    for m in manifest:
        by_ds.setdefault(m["ds"], [0, 0])
        by_ds[m["ds"]][0] += 1
        by_ds[m["ds"]][1] += m["size"]
    for ds, (n, sz) in by_ds.items():
        print(f"  {ds:14s} {n:6d} files  {sz / 2**30:.2f} GiB")
    return manifest


def intact(path, size):
    if not os.path.exists(path) or os.path.getsize(path) != size:
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False


def fetch(entry):
    dest = os.path.join(ROOT, "raw_aux", entry["ds"], entry["sym"],
                        os.path.basename(entry["key"]))
    if intact(dest, entry["size"]):
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    opener = _cdn_opener()
    for attempt in range(5):
        try:
            with opener.open(CDN + urllib.parse.quote(entry["key"]), timeout=60) as r, \
                    open(tmp, "wb") as f:
                f.write(r.read())
            if not intact(tmp, entry["size"]):
                raise OSError("corrupt archive")
            os.replace(tmp, dest)
            return "ok"
        except Exception as exc:
            if attempt == 4:
                sys.stderr.write(f"FAIL {entry['key']}: {exc}\n")
                return "fail"
    return "fail"


def download(manifest):
    print(f"{len(manifest)} archives, "
          f"{sum(e['size'] for e in manifest) / 2**30:.2f} GiB", flush=True)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as ex:
        futures = [ex.submit(fetch, e) for e in manifest]
        for i, f in enumerate(as_completed(futures), 1):
            counts[f.result()] += 1
            if i % 1000 == 0:
                print(f"  {i}/{len(manifest)}  {counts}", flush=True)
    print(f"download done: {counts}")


def _to_utc(series):
    v = pd.to_numeric(series, errors="coerce").astype("int64")
    return pd.to_datetime(np.where(v > 1e14, v // 1000, v), unit="ms", utc=True)


def convert_symbol(args):
    sym, ds = args
    src = os.path.join(ROOT, "raw_aux", ds, sym)
    if not os.path.isdir(src):
        return sym, ds, 0, "no archives"
    out_dir = os.path.join(ROOT, "funding" if ds == "fundingRate" else "markprice_1m")
    try:
        frames = []
        for fn in sorted(f for f in os.listdir(src) if f.endswith(".zip")):
            with zipfile.ZipFile(os.path.join(src, fn)) as z:
                raw = z.read(z.namelist()[0])
            head = raw[:80].split(b"\n", 1)[0]
            has_header = b"time" in head.lower() or b"open" in head.lower()
            if ds == "fundingRate":
                df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                                 names=None if has_header else
                                 ["calc_time", "funding_interval_hours", "last_funding_rate"])
                df.columns = [c.strip().lower() for c in df.columns]
                df = df.rename(columns={"calc_time": "date",
                                        "last_funding_rate": "funding_rate"})
                # Settlement stamps carry millisecond jitter (00:00:00.015);
                # floor to the minute so they align with bar boundaries. The
                # shift is sub-second and backwards, so it cannot leak a value
                # earlier than the bar that would first see it.
                df["date"] = _to_utc(df["date"]).floor("min")
                df = df[["date", "funding_rate", "funding_interval_hours"]]
            else:
                df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                                 names=None if has_header else KLINE_COLS)
                df.columns = [c.strip().lower() for c in df.columns]
                df["date"] = _to_utc(df["open_time"])
                df = df[["date", "open", "high", "low", "close"]]
                df.columns = ["date", "mark_open", "mark_high", "mark_low", "mark_close"]
            frames.append(df)
        if not frames:
            return sym, ds, 0, "empty"
        df = pd.concat(frames, ignore_index=True)
        df = df[(df["date"] >= pd.Timestamp("2025-01-01", tz="UTC")) &
                (df["date"] < pd.Timestamp("2026-08-17", tz="UTC"))]
        df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(os.path.join(out_dir, f"{sym}.parquet"),
                      compression="zstd", index=False)
        return sym, ds, len(df), None
    except Exception as exc:
        return sym, ds, 0, repr(exc)


def convert(manifest):
    jobs = sorted({(m["sym"], m["ds"]) for m in manifest})
    print(f"converting {len(jobs)} symbol/dataset pairs ...", flush=True)
    ok = bad = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, (sym, ds, n, err) in enumerate(ex.map(convert_symbol, jobs), 1):
            if err:
                bad += 1
                sys.stderr.write(f"FAIL {ds}/{sym}: {err}\n")
            else:
                ok += 1
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}  ok={ok} bad={bad}", flush=True)
    print(f"convert done: ok={ok} bad={bad}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    symbols = open(os.path.join(META, "active_symbols.txt")).read().split()
    path = os.path.join(META, "manifest_aux.json")
    if stage in ("manifest", "all") or not os.path.exists(path):
        manifest = build_manifest(symbols)
    else:
        manifest = json.load(open(path))
    if stage in ("download", "all"):
        download(manifest)
    if stage in ("convert", "all"):
        convert(manifest)
