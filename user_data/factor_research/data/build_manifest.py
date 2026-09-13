"""Build a download manifest for Binance UM perpetual 1m klines.

Enumerates the public data bucket and records which monthly and daily archives
fall inside the requested date range. Delisted symbols (the *SETTLED suffix)
are kept deliberately: dropping them would bake survivorship bias into every
factor computed downstream.
"""
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
META = os.path.join(os.path.dirname(__file__), "..", "meta")

MONTHS = {f"{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13)}
MONTHS = {m for m in MONTHS if "2025-01" <= m <= "2026-07"}
DAYS = {f"2026-08-{d:02d}" for d in range(1, 17)}

# Listing lives on s3-ap-northeast-1.amazonaws.com, which is only reachable
# through the shell's proxy, so this module deliberately keeps the environment
# proxy. download.py bypasses it instead -- see the note there.


def list_prefix(prefix):
    """Return (key, size) for every object under prefix, following pagination."""
    out, marker = [], ""
    while True:
        url = f"{BUCKET}?prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if marker:
            url += "&marker=" + urllib.parse.quote(marker)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    x = ET.fromstring(r.read())
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


def all_symbols():
    syms, marker = [], ""
    base = f"{BUCKET}?delimiter=/&prefix=data/futures/um/monthly/klines/"
    while True:
        url = base + ("&marker=" + urllib.parse.quote(marker) if marker else "")
        with urllib.request.urlopen(url, timeout=60) as r:
            x = ET.fromstring(r.read())
        for p in x.findall("s:CommonPrefixes/s:Prefix", NS):
            syms.append(p.text.rstrip("/").split("/")[-1])
        if x.findtext("s:IsTruncated", default="false", namespaces=NS) != "true":
            return syms
        marker = x.findtext("s:NextMarker", namespaces=NS)


def scan(sym):
    """Monthly needs a full listing; daily is narrowed to the 2026-08 prefix."""
    items = []
    targets = (
        ("monthly", f"data/futures/um/monthly/klines/{sym}/1m/", MONTHS),
        ("daily", f"data/futures/um/daily/klines/{sym}/1m/{sym}-1m-2026-08-", DAYS),
    )
    for kind, prefix, wanted in targets:
        try:
            entries = list_prefix(prefix)
        except Exception:
            continue
        for key, size in entries:
            if not key.endswith(".zip"):
                continue
            stamp = key.rsplit("-1m-", 1)[-1][:-4]
            if stamp in wanted:
                items.append({"sym": sym, "kind": kind, "key": key,
                              "size": size, "stamp": stamp})
    return items


if __name__ == "__main__":
    os.makedirs(META, exist_ok=True)
    raw = all_symbols()
    with open(os.path.join(META, "um_symbols_all.txt"), "w") as f:
        f.write("\n".join(raw))
    # Dated delivery contracts are not perpetuals; BUSD pairs died before 2025.
    syms = [s for s in raw if "_" not in s and not s.endswith("BUSD")]
    print(f"scanning {len(syms)} of {len(raw)} symbol dirs ...", flush=True)

    manifest = []
    with ThreadPoolExecutor(max_workers=64) as ex:
        for i, items in enumerate(ex.map(scan, syms), 1):
            manifest.extend(items)
            if i % 100 == 0:
                print(f"  {i}/{len(syms)}  files so far: {len(manifest)}", flush=True)

    with open(os.path.join(META, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    total = sum(m["size"] for m in manifest)
    active = sorted({m["sym"] for m in manifest})
    with open(os.path.join(META, "active_symbols.txt"), "w") as f:
        f.write("\n".join(active))
    print(f"\nsymbols with data in range: {len(active)}")
    print(f"files: {len(manifest)}  compressed total: {total / 2**30:.2f} GiB")
