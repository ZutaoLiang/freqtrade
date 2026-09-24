#!/usr/bin/env python3
"""Serial freqtrade backtest runner + aggregator for memory-constrained hosts.

Runs one `freqtrade backtesting` per time chunk, strictly one at a time, and merges the
exported trades into <out>/trades.csv with an IS/OOS summary. Use chunks when the full range
does not fit in memory (measure one chunk's peak RSS first; see SKILL.md resource rules).

  python bt_runner.py --config my-config.json --strategy MyStrat --datadir user_data/data/binance \
      --start 2025-01-01 --end 2026-08-14 --chunk quarter --oos-start 2026-01-01 --out user_data/bt/it07

Chunking caveats (printed in the summary, don't ignore them):
  * each chunk starts with a fresh wallet and flat book; trades still open at a chunk's end are
    force-closed ('force_exit'). Chunk only when typical holds are far shorter than a chunk.
  * a chunk also needs `startup_candle_count` candles of history before its start; the runner
    widens --timerange backwards by --warmup-days and freqtrade trims them again.
Also enforced here: --backtest-directory must exist before freqtrade runs, otherwise the export
is silently skipped (observed with freqtrade 2026.1).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd


def chunks(start: pd.Timestamp, end: pd.Timestamp, kind: str):
    if kind == "none":
        yield start, end
        return
    freq = {"month": "MS", "quarter": "QS"}[kind]
    edges = [start] + [t for t in pd.date_range(start, end, freq=freq) if start < t < end] + [end]
    yield from zip(edges[:-1], edges[1:])


def load_trades(d: str) -> list[dict]:
    zs = sorted(glob.glob(f"{d}/*.zip"))
    if not zs:
        return []
    with zipfile.ZipFile(zs[-1]) as zf:
        name = [n for n in zf.namelist() if n.endswith(".json") and "config" not in n and "meta" not in n][0]
        data = json.loads(zf.read(name))
    return [t for s in data["strategy"].values() for t in s["trades"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--strategy-path", default="user_data/strategies")
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--chunk", choices=["none", "month", "quarter"], default="quarter")
    ap.add_argument("--warmup-days", type=int, default=2)
    ap.add_argument("--oos-start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="", help="extra freqtrade args, e.g. '--timeframe-detail 1m'")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    rows = []
    for c0, c1 in chunks(start, end, a.chunk):
        tag = c0.strftime("%Y%m%d")
        d = os.path.join(a.out, tag)
        os.makedirs(d, exist_ok=True)                        # freqtrade will not create it
        tr = f"{(c0 - pd.Timedelta(days=a.warmup_days)).strftime('%Y%m%d')}-{c1.strftime('%Y%m%d')}"
        cmd = [sys.executable, "-m", "freqtrade", "backtesting", "-c", a.config, "--strategy", a.strategy,
               "--strategy-path", a.strategy_path, "--datadir", a.datadir, "--timerange", tr, "--cache", "none",
               "--export", "trades", "--backtest-directory", d] + a.extra.split()
        r = subprocess.run(["/usr/bin/time", "-f", "PEAK_RSS_KB %M"] + cmd, capture_output=True, text=True)
        open(os.path.join(a.out, f"{tag}.log"), "w").write(r.stdout + r.stderr)
        peak = next((l for l in r.stderr.splitlines() if l.startswith("PEAK_RSS_KB")), "")
        err = next((l for l in (r.stdout + r.stderr).splitlines() if " ERROR " in l), "")
        got = [t for t in load_trades(d) if c0 <= pd.Timestamp(t["open_date"]).tz_localize(None) < c1]
        rows += got
        print(f"{tag}: rc={r.returncode} trades={len(got)} {peak} {err[:150]}", flush=True)
        if r.returncode != 0:
            print("stopping: fix the error above before trusting any aggregate", file=sys.stderr)
            break
    if not rows:
        print("no trades"); return
    t = pd.DataFrame(rows)
    t["open_date"] = pd.to_datetime(t.open_date, utc=True)
    t.to_csv(os.path.join(a.out, "trades.csv"), index=False)
    cut = pd.Timestamp(a.oos_start, tz="UTC")
    print(f"\nexit reasons: {t.exit_reason.value_counts().to_dict()}")
    print(f"funding: nonzero on {(t.funding_fees != 0).sum()}/{len(t)} trades, total {t.funding_fees.sum():.2f}"
          "  (all zero on multi-hour holds = funding/mark data not read; run validate_datadir.py)")
    for seg, m in (("IS", t.open_date < cut), ("OOS", t.open_date >= cut)):
        x = t[m]
        if x.empty:
            print(f"{seg}: no trades"); continue
        g, b = x.profit_abs[x.profit_abs > 0].sum(), -x.profit_abs[x.profit_abs < 0].sum()
        days = x.groupby(x.open_date.dt.floor("D")).profit_ratio.mean()
        tstat = days.mean() / days.std() * np.sqrt(len(days)) if len(days) > 2 and days.std() > 0 else float("nan")
        print(f"{seg}: trades={len(x)} active_days={len(days)} profit={x.profit_abs.sum():.2f} "
              f"mean={x.profit_ratio.mean()*1e4:.1f}bp PF={g/b if b else float('inf'):.2f} win={(x.profit_ratio>0).mean():.2f} "
              f"day_t={tstat:.2f} trades/day={len(x)/max(1,(x.open_date.max()-x.open_date.min()).days):.2f}")
    print("monthly profit:", t.groupby(t.open_date.dt.strftime("%Y-%m")).profit_abs.sum().round(2).to_dict())


if __name__ == "__main__":
    main()
