"""Download Binance's `metrics` dataset: open interest and positioning ratios.

This is the largest gap in the base series. Open interest is not derivable from
OHLCV at all, and it is what separates the four regimes price alone cannot tell
apart -- price up on rising OI is new longs, price up on falling OI is shorts
covering, and the two mean opposite things next. The long/short account and
position ratios give a crowding measure that funding only partly proxies, split
between top traders and everyone else.

Columns per row, at 5-minute granularity:
    sum_open_interest, sum_open_interest_value,
    count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
    count_long_short_ratio, sum_taker_long_short_vol_ratio

Only daily archives exist for this dataset, so it is ~600 files per symbol and
851 of 864 symbols have it. Threads are held well below the usual cap because
this is meant to run alongside a CPU-bound sweep, not instead of it.
"""
import io
import json
import os
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from config import MAX_WORKERS  # noqa: E402

META = os.path.join(HERE, "..", "meta")
ROOT = "/root/freqtrade/user_data/data/binance_public"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CDN = "https://data.binance.vision/"
PREFIX = "data/futures/um/daily/metrics/"

START, END = "2025-01-01", "2026-08-17"
THREADS = int(os.environ.get("METRICS_THREADS", 24))

COLS = ["sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]

socket.setdefaulttimeout(60)


def _listing_opener():
    return urllib.request.build_opener()          # listing host needs the proxy


# One pooled session per worker thread. With 323k archives averaging 11 KB, the
# TLS handshake dominates: measured 2026-08-23 through the proxy, 40 threads
# went from 310 files/min on a fresh connection each time to 660 with a pooled
# session, and 57 threads reach 904. Direct connections are no longer the fast
# path -- the CDN answers at 226 files/min direct against 523 through the proxy
# -- so this deliberately uses the ambient proxy settings rather than bypassing
# them as the kline downloader does.
_LOCAL = threading.local()


def _session():
    s = getattr(_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=0)
        s.mount("https://", adapter)
        _LOCAL.session = s
    return s


def scan(sym):
    opener = _listing_opener()
    out, marker = [], ""
    while True:
        url = f"{BUCKET}?prefix={urllib.parse.quote(PREFIX + sym + '/')}&max-keys=1000"
        if marker:
            url += "&marker=" + urllib.parse.quote(marker)
        for attempt in range(4):
            try:
                x = ET.fromstring(opener.open(url, timeout=30).read())
                break
            except Exception:
                if attempt == 3:
                    return []
        got = [(c.findtext("s:Key", namespaces=NS),
                int(c.findtext("s:Size", namespaces=NS)))
               for c in x.findall("s:Contents", NS)]
        for key, size in got:
            if not key.endswith(".zip"):
                continue
            stamp = "-".join(os.path.basename(key)[:-4].split("-")[-3:])
            if START <= stamp < END:
                out.append({"sym": sym, "key": key, "size": size})
        if x.findtext("s:IsTruncated", default="false", namespaces=NS) != "true":
            return out
        marker = got[-1][0]


def build_manifest(symbols):
    print(f"scanning {len(symbols)} symbols ...", flush=True)
    manifest = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, items in enumerate(ex.map(scan, symbols), 1):
            manifest.extend(items)
            if i % 100 == 0:
                print(f"  {i}/{len(symbols)}  files: {len(manifest)}", flush=True)
    # Temp file plus rename: json.dump streams, so a plain write leaves a
    # valid-looking path on disk for several seconds while it is still being
    # filled. A reader waiting on the file's existence picked up a truncated
    # one and died on a JSON decode error.
    path = os.path.join(META, "manifest_metrics.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    total = sum(m["size"] for m in manifest)
    print(f"{len(manifest)} files, {total / 2**30:.2f} GiB")
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
    dest = os.path.join(ROOT, "raw_metrics", entry["sym"],
                        os.path.basename(entry["key"]))
    if intact(dest, entry["size"]):
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    url = CDN + urllib.parse.quote(entry["key"])
    for attempt in range(5):
        try:
            r = _session().get(url, timeout=60)
            r.raise_for_status()
            with open(tmp, "wb") as f:
                f.write(r.content)
            if not intact(tmp, entry["size"]):
                raise OSError("corrupt archive")
            os.replace(tmp, dest)
            return "ok"
        except Exception as exc:
            if attempt == 4:
                sys.stderr.write(f"FAIL {entry['key']}: {exc}\n")
                return "fail"
            time.sleep(0.5 * (attempt + 1))
    return "fail"


def download(manifest):
    print(f"{len(manifest)} archives, "
          f"{sum(e['size'] for e in manifest) / 2**30:.2f} GiB, "
          f"{THREADS} threads", flush=True)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = [ex.submit(fetch, e) for e in manifest]
        for i, f in enumerate(as_completed(futures), 1):
            counts[f.result()] += 1
            if i % 5000 == 0:
                print(f"  {i}/{len(manifest)}  {counts}", flush=True)
    print(f"download done: {counts}")


def convert_symbol(sym):
    src = os.path.join(ROOT, "raw_metrics", sym)
    if not os.path.isdir(src):
        return sym, 0, "no archives"
    try:
        frames = []
        for fn in sorted(f for f in os.listdir(src) if f.endswith(".zip")):
            with zipfile.ZipFile(os.path.join(src, fn)) as z:
                raw = z.read(z.namelist()[0])
            df = pd.read_csv(io.BytesIO(raw))
            df.columns = [c.strip().lower() for c in df.columns]
            frames.append(df)
        if not frames:
            return sym, 0, "empty"
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["create_time"], utc=True)
        # The archive repeats the first row of each file; drop duplicates on the
        # timestamp rather than trusting row order.
        df = df.drop_duplicates(subset="date").sort_values("date")
        df = df[(df["date"] >= pd.Timestamp(START, tz="UTC")) &
                (df["date"] < pd.Timestamp(END, tz="UTC"))]
        keep = ["date"] + [c for c in COLS if c in df.columns]
        df = df[keep].reset_index(drop=True)
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        out_dir = os.path.join(ROOT, "metrics")
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(os.path.join(out_dir, f"{sym}.parquet"),
                      compression="zstd", index=False)
        return sym, len(df), None
    except Exception as exc:
        return sym, 0, repr(exc)


def convert(symbols):
    print(f"converting {len(symbols)} symbols ...", flush=True)
    ok = bad = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, (sym, n, err) in enumerate(ex.map(convert_symbol, symbols), 1):
            if err and err != "no archives":
                bad += 1
                sys.stderr.write(f"FAIL {sym}: {err}\n")
            elif not err:
                ok += 1
            if i % 100 == 0:
                print(f"  {i}/{len(symbols)}  ok={ok} bad={bad}", flush=True)
    print(f"convert done: ok={ok} bad={bad}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    # Only symbols that ever entered the tradeable universe at some timeframe.
    # The other ~120 never clear the liquidity floor, so their metrics would be
    # downloaded and then masked out of every panel. The saving is smaller than
    # it looks -- 744 of 864 -- because the 15m universe is wider than the 1h
    # one once the volume floor is scaled per bar.
    only = os.path.join(META, "universe_symbols.txt")
    src = only if os.path.exists(only) else os.path.join(META, "active_symbols.txt")
    symbols = open(src).read().split()
    print(f"{len(symbols)} symbols from {os.path.basename(src)}", flush=True)
    path = os.path.join(META, "manifest_metrics.json")
    if stage in ("manifest", "all") or not os.path.exists(path):
        manifest = build_manifest(symbols)
    else:
        manifest = json.load(open(path))
    if stage in ("download", "all"):
        download(manifest)
    if stage in ("convert", "all"):
        convert(sorted({m["sym"] for m in manifest}))
