"""Download the Binance UM perpetual 1m kline archives listed in the manifest.

Resumable: any file already on disk that passes its internal CRC check is
skipped, so an interrupted run costs nothing to restart.
"""
import json
import os
import socket
import sys
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DOWNLOAD_THREADS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "..", "meta")
RAW = "/root/freqtrade/user_data/data/binance_public/raw"
BASE = "https://data.binance.vision/"
WORKERS = DOWNLOAD_THREADS

# urlretrieve has no timeout of its own, so a stalled connection pins its
# worker forever and the retry loop below never gets a chance to fire. One run
# hung for an hour on a single 8 KB .part before this was added.
socket.setdefaulttimeout(60)

# data.binance.vision answers directly at ~1.9 MB/s, while the shell's proxy
# tunnel caps it near 23 KB/s per connection, so downloads bypass the proxy.
# The manifest stage cannot: its listing host is only reachable through it.
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.ProxyHandler({}))
)


def intact(path, size):
    if not os.path.exists(path) or os.path.getsize(path) != size:
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False


def fetch(entry):
    dest = os.path.join(RAW, entry["sym"], os.path.basename(entry["key"]))
    if intact(dest, entry["size"]):
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(5):
        try:
            # A handful of listed symbols carry non-ASCII names, so the key
            # has to be percent-encoded rather than concatenated raw.
            url = BASE + urllib.parse.quote(entry["key"])
            urllib.request.urlretrieve(url, tmp)
            if not intact(tmp, entry["size"]):
                raise OSError("corrupt archive")
            os.replace(tmp, dest)
            return "ok"
        except Exception as exc:
            if attempt == 4:
                sys.stderr.write(f"FAIL {entry['key']}: {exc}\n")
                return "fail"
    return "fail"


if __name__ == "__main__":
    manifest = json.load(open(os.path.join(META, "manifest.json")))
    print(f"{len(manifest)} archives, "
          f"{sum(e['size'] for e in manifest) / 2**30:.2f} GiB", flush=True)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(fetch, e) for e in manifest]
        for i, f in enumerate(as_completed(futures), 1):
            counts[f.result()] += 1
            if i % 500 == 0:
                print(f"  {i}/{len(manifest)}  {counts}", flush=True)
    print(f"done: {counts}")
