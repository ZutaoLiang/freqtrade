"""Run the 24-unit matrix for every month of a year, one frozen pool per month.

SKILL.md section 3 requires a month's pool to be selected only from data before
that month, so each month gets its own universe file and its own ranking CSV.
All units across all months share one worker pool so the long 1m runs overlap
with the short ones.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_matrix import TIMEFRAMES, run_unit  # noqa: E402

RESEARCH = Path("user_data/research/tradingview")


def month_bounds(year: int, month: int, data_end: str) -> tuple[str, str, bool]:
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    hard_end = pd.Timestamp(data_end, tz="UTC")
    partial = end > hard_end
    if partial:
        end = hard_end
    return str(start.date()), str(end.date()), partial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="TrendShiftSignal")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--months", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--data-end", default="2026-08-17")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--config", default=str(RESEARCH / "config_base.json"))
    ap.add_argument("--datadir", default="user_data/data/tv_matrix")
    ap.add_argument("--tag", default="trendshift")
    ap.add_argument("--skip-select", action="store_true")
    ap.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    ap.add_argument("--detail", default="")
    args = ap.parse_args()

    months = [int(m) for m in args.months.split(",")]
    plan = []
    for m in months:
        start, end, partial = month_bounds(args.year, m, args.data_end)
        label = f"{args.year}-{m:02d}"
        uni_path = RESEARCH / f"universe_{label}.json"
        if not args.skip_select or not uni_path.exists():
            subprocess.run(
                [sys.executable, "scripts/tv_matrix/select_universe.py",
                 "--bt-start", start, "--bt-end", end, "--out", str(uni_path),
                 "--workers", "16"],
                check=True,
            )
            subprocess.run(
                [sys.executable, "scripts/tv_matrix/build_datadir.py",
                 "--universe", str(uni_path), "--out", f"{args.datadir}/futures"],
                check=True,
            )
        plan.append((label, start, end, partial, uni_path))
        print(f"[plan] {label} {start} -> {end} partial={partial}", flush=True)

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    jobs = []
    for label, start, end, _partial, uni_path in plan:
        uni = json.loads(uni_path.read_text())
        outdir = RESEARCH / f"matrix_{args.tag}_{label}"
        outdir.mkdir(parents=True, exist_ok=True)
        timerange = f"{start.replace('-', '')}-{end.replace('-', '')}"
        for group, pairs in uni["groups"].items():
            for tf in timeframes:
                jobs.append((label, (group, pairs, tf, args.config, outdir,
                                     args.strategy, args.datadir, timerange, args.detail)))

    # Heaviest timeframes first so they are not left stranded at the end.
    weight = {"1m": 0, "5m": 1, "15m": 2, "30m": 3, "1h": 4, "1d": 5}
    jobs.sort(key=lambda j: weight[j[1][2]])

    rows, per_pair = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_unit, job): label for label, job in jobs}
        for fut, label in futures.items():
            res = fut.result()
            r = dict(res["row"])
            r["month"] = label
            rows.append(r)
            if len(res["per_pair"]):
                per_pair.append(
                    res["per_pair"].assign(month=label, group=r["group"], timeframe=r["timeframe"])
                )
            done += 1
            print(f"[{done}/{len(jobs)}] {label} {r['group']:9s} {r['timeframe']:3s} "
                  f"status={r.get('status')} trades={r.get('trades')} "
                  f"net%={r.get('net_profit_pct')}", flush=True)

    summary = pd.DataFrame(rows)
    out = RESEARCH / f"year_{args.tag}_{args.year}"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    if per_pair:
        pd.concat(per_pair, ignore_index=True).to_csv(out / "per_pair.csv", index=False)
    print(f"wrote {out/'summary.csv'}")


if __name__ == "__main__":
    main()
