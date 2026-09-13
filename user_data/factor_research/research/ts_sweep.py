"""Sweep (screen x spec x side x hold) with the per-pair absolute-signal simulation.

Usage: python research/ts_sweep.py --tf 1h --split train --label ts1 [--specs round2] [--workers N]
Each worker computes one spec on full history, then loops screens x sides x holds.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import base as B  # noqa: E402
import factors as F  # noqa: E402
import screens as S  # noqa: E402
import ts_book as TB  # noqa: E402
from config import workers_for  # noqa: E402
from panel import CACHE, Panel  # noqa: E402

REPORTS = os.path.join(HERE, "..", "reports")
HOLDS_H = (6, 24, 72)   # default; override with --holds
SIDES = ("long", "short")
_CTX = {}


def _init(tf, split_name, screens_wanted, holds_h=HOLDS_H, rank_win=TB.RANK_WIN):
    p = Panel(tf)
    splits = json.load(open(os.path.join(CACHE, "splits.json")))
    lo, hi = (pd.Timestamp(x) for x in splits[split_name])
    sl = slice(int(np.searchsorted(p.index, lo)), int(np.searchsorted(p.index, hi)))
    bpd = B.BARS_PER_DAY[tf]
    scr = S.load(tf)
    scr = {k: np.asarray(v) for k, v in scr.items() if (screens_wanted is None or k in screens_wanted)}
    fields = set(p.meta.get("fields", []))
    # carried-forward settled rate from the base cache (the hist panel stores it only on settlement bars)
    fr = np.asarray(B.load(tf, "funding_rate"), dtype="float64") if "funding_rate" in B.names(tf) else np.zeros(p.shape)
    fr = np.where(np.isfinite(fr), fr, 0.0)
    iv = np.asarray(p["funding_interval_hours"], dtype="float64") if "funding_interval_hours" in fields else np.full(p.shape, 8.0)
    iv = np.where(np.isfinite(iv) & (iv > 0), iv, 8.0)
    fund_per_bar = np.where(np.isfinite(fr), fr * (24.0 / bpd) / iv, 0.0)   # paid by a long, per bar
    _CTX.update(tf=tf, sl=sl, bpd=bpd, screens=scr, open=np.asarray(p["open"], dtype="float64"),
                fund=fund_per_bar, index=p.index, symbols=p.symbols, holds=tuple(holds_h), rank_win=rank_win)


def _run(spec):
    tf = _CTX["tf"]
    try:
        f = spec.compute(tf, lambda n: np.asarray(B.load(tf, n), dtype="float64"))
    except Exception as exc:  # noqa: BLE001
        return [{"name": spec.name(), "error": repr(exc)}]
    rows = []
    long_sig, short_sig = TB.entries(f, _CTX["rank_win"])
    for sname, scr in _CTX["screens"].items():
        for side in SIDES:
            sig = (long_sig if side == "long" else short_sig) & scr
            if not sig[_CTX["sl"]].any():
                continue
            for hh in _CTX["holds"]:
                h = max(1, int(round(hh * _CTX["bpd"] / 24)))
                df, ret_bar = TB.simulate_from_signal(sig, side, _CTX["open"], _CTX["fund"], h, _CTX["sl"])
                s = TB.summarize(df, ret_bar, _CTX["index"], _CTX["bpd"])
                s.update(name=spec.name(), tag=spec.tag, screen=sname, side=side, hold_h=hh)
                rows.append(s)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--split", default="train")
    ap.add_argument("--label", default="ts1")
    ap.add_argument("--specs", default="round2")
    ap.add_argument("--screens", default=None, help="comma list; default all")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--holds", default=None, help="comma list of hold hours, e.g. 24,72,168,336,672")
    ap.add_argument("--rank-win", type=int, default=TB.RANK_WIN, help="ts_rank window in bars")
    ap.add_argument("--names", default=None, help="semicolon-separated spec names to run (names contain commas)")
    a = ap.parse_args()
    specs = {"round1": F.ROUND1, "round2": F.ROUND2(), "round3": F.ROUND3(), "daily": F.DAILY()}[a.specs]
    keep, dropped = F.for_timeframe(specs, a.tf, B.names(a.tf))
    if a.names:
        want = set(a.names.split(";"))
        keep = [s for s in keep if s.name() in want]
    if a.limit:
        keep = keep[: a.limit]
    wanted = set(a.screens.split(",")) if a.screens else None
    n_workers = a.workers or workers_for(a.tf, "sweep")
    print(f"{a.tf} {a.split}: {len(keep)} specs ({len(dropped)} dropped), {n_workers} workers", flush=True)
    t0 = time.time()
    out = []
    holds = tuple(int(x) for x in a.holds.split(",")) if a.holds else HOLDS_H
    print(f"  holds {holds} h, rank window {a.rank_win} bars", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init, initargs=(a.tf, a.split, wanted, holds, a.rank_win)) as ex:
        for i, rows in enumerate(ex.map(_run, keep), 1):
            out.extend(rows)
            if i % 20 == 0:
                print(f"  {i}/{len(keep)}  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(out)
    path = os.path.join(REPORTS, f"ts_{a.label}_{a.tf}_{a.split}.csv")
    df.to_csv(path, index=False)
    print(f"wrote {path}: {len(df)} rows, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
