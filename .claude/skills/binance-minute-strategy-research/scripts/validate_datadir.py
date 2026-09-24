#!/usr/bin/env python3
"""Audit a freqtrade futures datadir before trusting any backtest built on it.

Per pair it checks the things that break results WITHOUT raising errors in freqtrade:
  * klines: date dtype is timestamp[ms, UTC], sorted, no duplicates; gap count / largest gap.
  * funding starts later than the klines  -> every trade before that date gets funding = 0.
    (Seen in practice: 1h funding files that began 2026-01-01 under 2025 klines.)
  * funding rows that do not find a mark row on the same `date` -> freqtrade inner-joins funding
    with mark, so those settlements are dropped silently.
  * funding stamps not on the hour (calc_time jitter not rounded) -> same silent drop.
  * mark hours missing relative to the klines.

Usage:
  python validate_datadir.py --datadir user_data/data/binance [--interval 1m] [--csv report.csv]
Exit code 1 if any pair has an ERROR-level finding.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


def read_date(path: Path) -> pa.ChunkedArray:
    with pa.memory_map(str(path)) as src:
        return ipc.open_file(src).read_all().column("date")


def dates(path: Path) -> pd.Series:
    return read_date(path).to_pandas()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="user_data/data/binance")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    fut = Path(a.datadir) / "futures"
    rows, errors = [], 0
    for kpath in sorted(fut.glob(f"*-{a.interval}-futures.feather")):
        pair = kpath.name.split(f"-{a.interval}-")[0]
        col = read_date(kpath)
        r = {"pair": pair, "issues": []}
        if str(col.type) != "timestamp[ms, tz=UTC]":
            r["issues"].append(f"ERROR date dtype {col.type}")
        d = col.to_pandas()
        del col
        r.update(first=d.iloc[0], last=d.iloc[-1], rows=len(d))
        if not d.is_monotonic_increasing or d.duplicated().any():
            r["issues"].append("ERROR unsorted or duplicate dates")
        step = pd.Timedelta(a.interval.replace("m", "min").replace("h", "h"))
        diff = d.diff()
        r["gaps"] = int((diff > step).sum())
        r["max_gap"] = str(diff.max())
        fpath, mpath = fut / f"{pair}-1h-funding_rate.feather", fut / f"{pair}-1h-mark.feather"
        if not fpath.exists():
            r["issues"].append("ERROR no 1h funding_rate file (funding will be 0)")
        else:
            fd = dates(fpath)
            r["funding_first"] = fd.iloc[0] if len(fd) else None
            if len(fd) == 0:
                r["issues"].append("ERROR empty funding file")
            else:
                if fd.iloc[0] > d.iloc[0] + pd.Timedelta(days=1):
                    r["issues"].append(f"ERROR funding starts {fd.iloc[0].date()} but klines start {d.iloc[0].date()}")
                off = int((fd != fd.dt.floor("h")).sum())
                if off:
                    r["issues"].append(f"ERROR {off} funding stamps not on the hour")
                if mpath.exists():
                    md = dates(mpath)
                    inside = fd[fd <= d.iloc[-1]]          # funding newer than the klines is harmless
                    unmatched = int((~inside.isin(set(md))).sum())
                    r["funding_unmatched_by_mark"] = unmatched
                    if unmatched:
                        r["issues"].append(f"WARN {unmatched}/{len(fd)} funding rows have no mark row (dropped by freqtrade)")
        if not mpath.exists():
            r["issues"].append("ERROR no 1h mark file (funding cannot be computed)")
        else:
            md = dates(mpath)
            want = pd.date_range(d.iloc[0].floor("h"), d.iloc[-1].floor("h"), freq="h")
            r["mark_missing_hours"] = int((~want.isin(md)).sum())
            if r["mark_missing_hours"]:
                r["issues"].append(f"WARN {r['mark_missing_hours']} mark hours missing")
        errors += any(i.startswith("ERROR") for i in r["issues"])
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no pairs found"); sys.exit(1)
    df["issues"] = df["issues"].apply("; ".join)
    if a.csv:
        df.to_csv(a.csv, index=False)
    bad = df[df.issues != ""]
    print(f"pairs: {len(df)} | with findings: {len(bad)} | with ERROR: {errors}")
    if len(bad):
        print(bad[["pair", "first", "issues"]].head(50).to_string(index=False))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
