#!/usr/bin/env python3
"""Download Binance USDT-M futures archives from data.binance.vision (serial, resumable, verified).

Mirrors the archive layout under --raw so reruns skip files that are already present and
verified. Every zip is checked against its published .CHECKSUM (SHA-256). The current and
previous month fall back to daily archives when their monthly archive is not packed yet.
404 means "no data for that period" (before listing / after delisting) and is not an error.
Monthly mark-price archives occasionally miss a day; vision_to_freqtrade.py reports that.

Kinds:
  klines          data/futures/um/{monthly|daily}/klines/{SYM}/{interval}/...
  fundingRate     data/futures/um/monthly/fundingRate/{SYM}/...          (monthly only; the
                  current/previous month comes from REST /fapi/v1/fundingRate -> rest/ subdir)
  markPriceKlines data/futures/um/{monthly|daily}/markPriceKlines/{SYM}/1h/...
  metrics         data/futures/um/daily/metrics/{SYM}/...  (daily only, 5m: OI, top-trader and account
                  long/short ratios, taker buy/sell volume ratio; opt-in via --kinds)
  premiumIndexKlines data/futures/um/{monthly|daily}/premiumIndexKlines/{SYM}/1h/...  (opt-in via --kinds;
                  the per-minute input of the funding rate: close = perp/index premium)

Examples:
  # list every USDT-M symbol that has monthly 1m klines
  python download_vision.py --list-symbols > symbols.txt
  # two symbols, 1m klines + funding + 1h mark, Jan 2025 .. Aug 2026
  python download_vision.py --symbols BTCUSDT,ETHUSDT --start 2025-01 --end 2026-08
  # everything in symbols.txt, klines only
  python download_vision.py --symbols-file symbols.txt --kinds klines --start 2025-01 --end 2026-08

Stdlib + requests only. One request at a time by design (low-memory, low-IO hosts; also
polite to the bucket). Nothing is loaded into memory beyond one zip.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path

import requests

BASE = "https://data.binance.vision/"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "freqtrade-minute-research/1.0"


def get(url: str, stream: bool = False) -> requests.Response | None:
    """GET with retries. Returns None on 404."""
    for attempt in range(5):
        try:
            r = SESSION.get(url, timeout=120, stream=stream)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        if r.status_code >= 500 or r.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"giving up on {url}")


def list_prefixes(prefix: str) -> list[str]:
    """S3 ListObjects (v1) with delimiter: returns child 'directories' under prefix."""
    out, marker = [], ""
    while True:
        r = get(f"{S3}?delimiter=/&prefix={prefix}&marker={marker}")
        tree = ET.fromstring(r.content)
        out += [p.findtext("s3:Prefix", namespaces=NS) for p in tree.findall("s3:CommonPrefixes", NS)]
        if tree.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return out
        marker = tree.findtext("s3:NextMarker", namespaces=NS) or out[-1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(key: str, raw: Path, stats: dict) -> bool:
    """Download key (+ .CHECKSUM) into raw/key. True if a verified file is present afterwards."""
    dst = raw / key
    ck = dst.with_name(dst.name + ".CHECKSUM")
    if dst.exists() and ck.exists():
        want = ck.read_text().split()[0]
        if sha256(dst) == want:
            stats["skipped"] += 1
            return True
    r = get(BASE + key + ".CHECKSUM")
    if r is None:
        stats["missing"] += 1
        return False
    want = r.text.split()[0]
    r = get(BASE + key, stream=True)
    if r is None:
        stats["missing"] += 1
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    if sha256(tmp) != want:
        tmp.unlink()
        stats["bad_checksum"] += 1
        print(f"  checksum mismatch, discarded: {key}", file=sys.stderr)
        return False
    tmp.replace(dst)
    ck.write_text(f"{want}  {dst.name}\n")
    stats["downloaded"] += 1
    return True


def fetch_funding_rest(sym: str, y: int, m: int, raw: Path, stats: dict) -> None:
    """Vision has no daily fundingRate and packs the monthly file after month end, so the
    current/previous month comes from the public REST endpoint (no key). limit=200: larger
    limits have been answered with 403 on some hosts. Saved as CSV in the Vision column
    layout under data/futures/um/rest/fundingRate/; vision_to_freqtrade.py reads it.
    Skipped silently if the REST API is geo-blocked on this host."""
    t0 = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    t1 = int(datetime(ny, nm, 1, tzinfo=timezone.utc).timestamp() * 1000) - 1
    rows, start = [], t0
    while start <= t1:
        try:
            r = SESSION.get("https://fapi.binance.com/fapi/v1/fundingRate", timeout=30,
                            params={"symbol": sym, "startTime": start, "endTime": t1, "limit": 200})
        except requests.RequestException:
            stats["funding_rest_failed"] = stats.get("funding_rest_failed", 0) + 1
            return
        if r.status_code != 200:
            stats["funding_rest_failed"] = stats.get("funding_rest_failed", 0) + 1
            return
        batch = r.json()
        if not batch:
            break
        rows += batch
        start = batch[-1]["fundingTime"] + 1
        if len(batch) < 200:
            break
        time.sleep(0.3)
    if not rows:
        return
    dst = raw / f"data/futures/um/rest/fundingRate/{sym}/{sym}-fundingRate-{y:04d}-{m:02d}.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = ["calc_time,funding_interval_hours,last_funding_rate"]
    lines += [f"{x['fundingTime']},,{x['fundingRate']}" for x in rows]
    dst.write_text("\n".join(lines) + "\n")
    stats["funding_rest_rows"] = stats.get("funding_rest_rows", 0) + len(rows)


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def download_symbol(sym: str, kinds: list[str], start: str, end: str, interval: str, raw: Path) -> dict:
    stats = {"downloaded": 0, "skipped": 0, "missing": 0, "bad_checksum": 0, "daily_fallback_months": 0}
    today = datetime.now(timezone.utc).date()
    for y, m in months(start, end):
        tag = f"{y:04d}-{m:02d}"
        if "fundingRate" in kinds:
            ok = fetch(f"data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{tag}.zip", raw, stats)
            if not ok and (today.year * 12 + today.month) - (y * 12 + m) <= 1:
                fetch_funding_rest(sym, y, m, raw, stats)
        if "metrics" in kinds:                      # 5m OI / long-short ratios: daily archives only
            last = calendar.monthrange(y, m)[1]
            for d in range(1, last + 1):
                day = date(y, m, d)
                if day >= today:
                    break
                fetch(f"data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day.isoformat()}.zip", raw, stats)
        for kind, iv in (("klines", interval), ("markPriceKlines", "1h"), ("premiumIndexKlines", "1h")):
            if kind not in kinds:
                continue
            key = f"data/futures/um/monthly/{kind}/{sym}/{iv}/{sym}-{iv}-{tag}.zip"
            if fetch(key, raw, stats):
                continue
            # Monthly archives (partial listing/delisting months included) are packed a few days
            # after month end, so a missing monthly file only means "not packed yet" for the
            # current or previous month. Older gaps mean no data; don't probe 30 daily 404s.
            recent = (today.year * 12 + today.month) - (y * 12 + m) <= 1
            if not recent:
                continue
            stats["daily_fallback_months"] += 1
            last = min(calendar.monthrange(y, m)[1], (today.day - 1) if (y, m) == (today.year, today.month) else 31)
            for d in range(1, last + 1):
                day = date(y, m, d).isoformat()
                fetch(f"data/futures/um/daily/{kind}/{sym}/{iv}/{sym}-{iv}-{day}.zip", raw, stats)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-symbols", action="store_true", help="print USDT-M symbols with monthly klines and exit")
    ap.add_argument("--symbols", default="", help="comma separated, e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--symbols-file", default="", help="one symbol per line")
    ap.add_argument("--kinds", default="klines,fundingRate,markPriceKlines")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start", default="2025-01", help="YYYY-MM")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m"), help="YYYY-MM (inclusive)")
    ap.add_argument("--raw", default="user_data/data/binance-vision", help="local mirror root")
    a = ap.parse_args()

    if a.list_symbols:
        for p in list_prefixes("data/futures/um/monthly/klines/"):
            sym = p.rstrip("/").split("/")[-1]
            if sym.endswith("USDT"):
                print(sym)
        return
    syms = [s.strip() for s in a.symbols.split(",") if s.strip()]
    if a.symbols_file:
        syms += [s.strip() for s in Path(a.symbols_file).read_text().split() if s.strip()]
    if not syms:
        ap.error("give --symbols or --symbols-file (or --list-symbols)")
    kinds = [k.strip() for k in a.kinds.split(",")]
    raw = Path(a.raw)
    for i, sym in enumerate(syms, 1):
        st = download_symbol(sym, kinds, a.start, a.end, a.interval, raw)
        print(f"[{i}/{len(syms)}] {sym}: {st}", flush=True)


if __name__ == "__main__":
    main()
