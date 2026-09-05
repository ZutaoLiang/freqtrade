"""Sweep the explicit time stop on a continuous window (no monthly chopping)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

R = Path("user_data/research/tradingview")


def run(job) -> dict:
    group, pairs, tf, bars, timerange, outdir, startup = job
    base = json.loads((R / "config_base.json").read_text())
    base["timeframe"] = tf
    base["max_open_trades"] = 20
    base["exchange"] = dict(base["exchange"])
    base["exchange"]["pair_whitelist"] = pairs
    cfg = outdir / f"config_{group}_{tf}_{bars}.json"
    cfg.write_text(json.dumps(base, indent=1))

    params = outdir / f"params_{group}_{tf}_{bars}.json"
    params.write_text(json.dumps({
        "strategy_name": "TrendShiftTimeStop",
        "params": {"buy": {}, "sell": {"max_bars_in_trade": bars}},
    }))
    strat_params = Path("user_data/strategies/tradingview/TrendShiftTimeStop.json")
    strat_params.write_text(params.read_text())

    cmd = ["freqtrade", "backtesting", "-c", str(cfg), "-s", "TrendShiftTimeStop",
           "--datadir", "user_data/data/tv_matrix", "--timeframe", tf,
           "--timerange", timerange, "--cache", "none"]
    o = subprocess.run(cmd, capture_output=True, text=True).stdout

    def g(pat, cast=float):
        m = re.search(pat, o)
        return cast(m.group(1)) if m else None

    return {
        "group": group, "timeframe": tf, "max_bars": bars,
        "trades": g(r"Total/Daily Avg Trades.*?(\d+) /", int),
        "net_pct": g(r"Total profit %.*?([-\d.]+)%"),
        "pf": g(r"Profit factor.*?([\d.]+)"),
        "sharpe": g(r"Sharpe.*?([-\d.]+)"),
        "maxdd_pct": g(r"Max % of account underwater.*?([\d.]+)%"),
        "winrate": g(r"TOTAL .*?(\d+\.\d) \|"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=str(R / "universe_2026-01.json"))
    ap.add_argument("--timerange", default="20260101-20260901")
    ap.add_argument("--timeframes", default="1d")
    ap.add_argument("--bars", default="0,1,2,3,5,7,10,14,21,30")
    ap.add_argument("--outdir", default=str(R / "timestop_2026"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    uni = json.loads(Path(args.universe).read_text())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (g, pairs, tf, int(b), args.timerange, outdir, 200)
        for g, pairs in uni["groups"].items()
        for tf in args.timeframes.split(",")
        for b in args.bars.split(",")
    ]
    # The params file is global to the strategy class, so runs cannot overlap.
    rows = [run(j) for j in jobs]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sweep.csv", index=False)
    for tf in df.timeframe.unique():
        sub = df[df.timeframe == tf]
        print(f"\n=== {tf} net % ===")
        print(sub.pivot_table(index="max_bars", columns="group", values="net_pct").round(2).to_string())
        print(f"=== {tf} max drawdown % ===")
        print(sub.pivot_table(index="max_bars", columns="group", values="maxdd_pct").round(2).to_string())


if __name__ == "__main__":
    main()
