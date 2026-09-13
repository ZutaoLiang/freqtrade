"""Sample the order-book depth archive to price slippage per symbol.

Every cost number so far assumes a flat 5 bps one-way. That is defensible for
BTCUSDT and indefensible for the micro-caps the carry factor actually selects --
the whitelist builder's long side came back as 4/USDT, GUA/USDT, BTR/USDT. Until
impact is measured per symbol, the break-even figures are comparing a real edge
against an imaginary cost.

The full archive is ~139 GiB for the study window, which buys nothing: book
depth is a slowly-varying property of a market, not a bar-by-bar quantity. Two
days a month per symbol -- the 1st and the 15th -- is 9% of the download and
still resolves how depth changes over twenty months.

Rows are `timestamp, percentage, depth, notional`, one per price band at
+-1..5% from mid, sampled through the day.
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
PREFIX = "data/futures/um/daily/bookDepth/"

# Two days a month across the study window.
SAMPLE_DAYS = sorted(
    f"{y}-{m:02d}-{d:02d}"
    for y in (2025, 2026) for m in range(1, 13) for d in (1, 15)
    if "2025-01-01" <= f"{y}-{m:02d}-{d:02d}" <= "2026-08-16"
)
THREADS = int(os.environ.get("DEPTH_THREADS", 48))

socket.setdefaulttimeout(60)
_LOCAL = threading.local()


def _session():
    s = getattr(_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=0))
        _LOCAL.session = s
    return s


def scan(sym):
    """Only the sampled days, asked for by exact key rather than listed."""
    opener = urllib.request.build_opener()
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
            if stamp in set(SAMPLE_DAYS):
                out.append({"sym": sym, "key": key, "size": size})
        if x.findtext("s:IsTruncated", default="false", namespaces=NS) != "true":
            return out
        marker = got[-1][0]


def build_manifest(symbols):
    print(f"scanning {len(symbols)} symbols for {len(SAMPLE_DAYS)} sampled days ...",
          flush=True)
    manifest = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, items in enumerate(ex.map(scan, symbols), 1):
            manifest.extend(items)
            if i % 100 == 0:
                print(f"  {i}/{len(symbols)}  files: {len(manifest)}", flush=True)
    path = os.path.join(META, "manifest_bookdepth.json")
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
    dest = os.path.join(ROOT, "raw_bookdepth", entry["sym"],
                        os.path.basename(entry["key"]))
    if intact(dest, entry["size"]):
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    url = CDN + urllib.parse.quote(entry["key"])
    for attempt in range(5):
        try:
            r = _session().get(url, timeout=120)
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
            if i % 2000 == 0:
                print(f"  {i}/{len(manifest)}  {counts}", flush=True)
    print(f"download done: {counts}")


def summarise_symbol(sym):
    """Median notional resting inside each price band, per sampled day.

    Collapsing each day to a median across its intraday snapshots keeps the
    output small enough to hold every symbol in memory while still showing how
    depth drifts over the window.
    """
    src = os.path.join(ROOT, "raw_bookdepth", sym)
    if not os.path.isdir(src):
        return sym, None, "no archives"
    try:
        rows = []
        for fn in sorted(f for f in os.listdir(src) if f.endswith(".zip")):
            with zipfile.ZipFile(os.path.join(src, fn)) as z:
                raw = z.read(z.namelist()[0])
            df = pd.read_csv(io.BytesIO(raw))
            df.columns = [c.strip().lower() for c in df.columns]
            if "notional" not in df.columns or "percentage" not in df.columns:
                continue
            day = "-".join(fn[:-4].split("-")[-3:])
            g = df.groupby("percentage")["notional"].median()
            rows.append({"sym": sym, "day": day,
                         **{f"pct{int(k):+d}": float(v) for k, v in g.items()}})
        if not rows:
            return sym, None, "empty"
        return sym, pd.DataFrame(rows), None
    except Exception as exc:
        return sym, None, repr(exc)


def summarise(symbols):
    print(f"summarising {len(symbols)} symbols ...", flush=True)
    frames, bad = [], 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, (sym, df, err) in enumerate(ex.map(summarise_symbol, symbols), 1):
            if err and err != "no archives":
                bad += 1
                sys.stderr.write(f"FAIL {sym}: {err}\n")
            elif df is not None:
                frames.append(df)
            if i % 100 == 0:
                print(f"  {i}/{len(symbols)}  ok={len(frames)} bad={bad}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    dest = os.path.join(ROOT, "bookdepth_summary.parquet")
    out.to_parquet(dest, compression="zstd", index=False)
    print(f"wrote {dest}: {len(out)} symbol-days, {out['sym'].nunique()} symbols")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    only = os.path.join(META, "universe_symbols.txt")
    symbols = open(only).read().split()
    path = os.path.join(META, "manifest_bookdepth.json")
    if stage in ("manifest", "all") or not os.path.exists(path):
        manifest = build_manifest(symbols)
    else:
        manifest = json.load(open(path))
    if stage in ("download", "all"):
        download(manifest)
    if stage in ("summarise", "all"):
        summarise(sorted({m["sym"] for m in manifest}))
