#!/usr/bin/env python3
"""Build a freqtrade pair whitelist from a datadir, without look-ahead and without silent drops.

* Liquidity rank = median daily quote volume (close*volume) inside --rank-start..--rank-end.
  Use the in-sample window only: ranking on the whole history leaks the future into the
  universe (coins that later became big are, by construction, the winners).
* --min-days: pair must have at least this many days of data inside the ranking window.
* Pairs missing from freqtrade's bundled binance_leverage_tiers.json cannot be backtested in
  futures mode (freqtrade aborts the WHOLE run). They are removed and listed, because the
  newest listings are exactly the ones missing - a survivorship bias you must report.
* --exclude: regex on the base asset for non-crypto perps (stocks, commodities, gold tokens),
  e.g. '^(PAXG|XAUT|NVDA|TSLA|AAPL|AMZN|GOOGL|META|MSTR|COIN|HOOD|CRCL|PLTR|INTC|EWY|CL|BZ)$'.

Output: JSON list of 'BASE/USDT:USDT' on stdout (paste into exchange.pair_whitelist), report on stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


def tiers_file(explicit: str = "") -> set[str]:
    if explicit:
        return set(json.load(open(explicit)))
    sys.path.insert(0, os.getcwd())           # freqtrade source checkout run from the repo root
    try:
        import freqtrade
        p = Path(os.path.dirname(freqtrade.__file__)) / "exchange" / "binance_leverage_tiers.json"
        return set(json.load(open(p)))
    except Exception as exc:  # freqtrade not importable from here
        print(f"warning: cannot read leverage tiers ({exc}); not filtering on them", file=sys.stderr)
        return set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="user_data/data/binance")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--rank-start", required=True, help="YYYY-MM-DD, start of the in-sample window")
    ap.add_argument("--rank-end", required=True, help="YYYY-MM-DD, end of the in-sample window")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-days", type=int, default=300)
    ap.add_argument("--exclude", default=r"^(PAXG|XAUT)$")
    ap.add_argument("--tiers-file", default="", help="binance_leverage_tiers.json path (default: from freqtrade)")
    a = ap.parse_args()
    t0, t1 = pd.Timestamp(a.rank_start, tz="UTC"), pd.Timestamp(a.rank_end, tz="UTC")
    tiers, rx = tiers_file(a.tiers_file), re.compile(a.exclude) if a.exclude else None
    rows = []
    for f in sorted((Path(a.datadir) / "futures").glob(f"*-{a.interval}-futures.feather")):
        base = f.name.split("_USDT_USDT")[0]
        if rx and rx.search(base):
            continue
        with pa.memory_map(str(f)) as src:
            t = ipc.open_file(src).read_all().select(["date", "close", "volume"]).to_pandas()
        t = t[(t.date >= t0) & (t.date < t1)]
        if t.empty:
            continue
        qv = (t.close * t.volume).groupby(t.date.dt.floor("D")).sum()
        rows.append((f"{base}/USDT:USDT", float(qv.median()), len(qv)))
        del t
    df = pd.DataFrame(rows, columns=["pair", "median_daily_qv", "days"])
    df = df[df.days >= a.min_days].sort_values("median_daily_qv", ascending=False)
    no_tier = df[~df.pair.isin(tiers)] if tiers else df.iloc[:0]
    ok = df[df.pair.isin(tiers)] if tiers else df
    pick = ok.head(a.top)
    print(f"eligible {len(df)} | removed for missing leverage tiers {len(no_tier)}: {no_tier.pair.head(30).tolist()}", file=sys.stderr)
    print(pick.assign(median_daily_qv=(pick.median_daily_qv / 1e6).round(1)).rename(columns={"median_daily_qv": "qv_musd"}).to_string(index=False), file=sys.stderr)
    print(json.dumps(pick.pair.tolist()))


if __name__ == "__main__":
    main()
