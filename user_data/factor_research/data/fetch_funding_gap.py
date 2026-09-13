"""Backfill the funding-rate tail the public archive does not cover.

Binance publishes fundingRate as monthly archives only, so the current partial
month -- 2026-08-01 onwards -- has no zip to download and every funding factor
would go blank over the last sixteen days of the holdout. The REST endpoint
serves that stretch directly; it is reachable only through the shell proxy.

Only the gap is fetched, so the archive stays the source of truth for
everything before it and re-running this is idempotent.
"""
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/root/freqtrade/user_data/data/binance_public"
FUNDING = os.path.join(ROOT, "funding")
API = "https://fapi.binance.com/fapi/v1/fundingRate"
END = pd.Timestamp("2026-08-17", tz="UTC")
# Eight threads earned a WAF 403 after roughly 200 symbols on 2026-08-21, so
# this is deliberately slow: the whole job is ~850 tiny requests and finishing
# in ten minutes without a ban beats finishing in one and getting one.
THREADS = int(os.environ.get("FUNDING_THREADS", 4))
LIMIT = 200               # above 200 the proxy's WAF answers 403
# Seconds between requests inside a worker. Four threads at 0.35s clears the
# bulk of the symbols but still draws a handful of 403s; a retry pass at
# FUNDING_THREADS=1 FUNDING_PAUSE=1.5 picks up the stragglers.
PAUSE = float(os.environ.get("FUNDING_PAUSE", 0.35))

socket.setdefaulttimeout(30)
_OPENER = urllib.request.build_opener()   # keep the proxy: fapi is not direct


def _get(symbol, start_ms):
    """Most recent settlements, filtered client-side.

    Two WAF quirks through this proxy, both measured rather than guessed:
    sending startTime returns 403 while the identical request without it
    succeeds, and any limit above 200 returns 403 as well -- 2, 100 and 200 all
    pass, 500 and 1000 do not. So the window is not requested and the page is
    capped at 200 records, which is 66 days at an 8h interval and four times
    the partial month this has to cover; the caller drops what is already on
    disk.
    """
    time.sleep(PAUSE)
    url = f"{API}?{urllib.parse.urlencode({'symbol': symbol, 'limit': LIMIT})}"
    for attempt in range(4):
        try:
            return json.load(_OPENER.open(url, timeout=25))
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404):        # delisted or never listed
                return None
            if attempt == 3:
                raise
            # 403/418/429 all mean "slow down"; back off hard rather than
            # burning the remaining attempts at the same rate.
            time.sleep((15 if exc.code in (403, 418, 429) else 2) * (attempt + 1))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def backfill(sym):
    path = os.path.join(FUNDING, f"{sym}.parquet")
    if not os.path.exists(path):
        return sym, 0, "no archive file"
    df = pd.read_parquet(path)
    if df.empty:
        return sym, 0, "empty"
    last = df["date"].max()
    if last >= END - pd.Timedelta(hours=1):
        return sym, 0, None
    try:
        rows = _get(sym, int(last.value // 10**6) + 1)
    except Exception as exc:
        return sym, 0, repr(exc)
    if not rows:
        return sym, 0, None
    add = pd.DataFrame({
        "date": pd.to_datetime([r["fundingTime"] for r in rows], unit="ms", utc=True).floor("min"),
        "funding_rate": [float(r["fundingRate"]) for r in rows],
    })
    add = add[(add["date"] > last) & (add["date"] < END)]
    if add.empty:
        return sym, 0, None
    # The REST payload carries no interval, so inherit the last archived one;
    # a symbol that changed interval mid-gap would be mislabelled, which is why
    # this only ever runs over a partial month.
    add["funding_interval_hours"] = df["funding_interval_hours"].iloc[-1]
    out = pd.concat([df, add[df.columns]], ignore_index=True)
    out = out.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    tmp = path + ".tmp"
    out.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path)
    return sym, len(add), None


if __name__ == "__main__":
    only = os.environ.get("FUNDING_ONLY", "").strip()
    if only:
        # A retry pass wants the handful that 403'd, not another walk over 851
        # files of which most are delisted and will never gain a row.
        syms = sorted(s for s in only.replace(",", " ").split())
    else:
        syms = sorted(f[:-8] for f in os.listdir(FUNDING) if f.endswith(".parquet"))
    print(f"backfilling funding tail for {len(syms)} symbols ...", flush=True)
    added = filled = bad = 0
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        for i, (sym, n, err) in enumerate(ex.map(backfill, syms), 1):
            if err:
                bad += 1
                sys.stderr.write(f"FAIL {sym}: {err}\n")
            elif n:
                filled += 1
                added += n
            if i % 100 == 0:
                print(f"  {i}/{len(syms)}  symbols filled {filled}, rows {added}, "
                      f"errors {bad}", flush=True)
    print(f"funding backfill done: {filled} symbols, {added} rows, {bad} errors")
