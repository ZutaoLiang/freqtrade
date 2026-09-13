"""Fold open interest and positioning ratios into the panel layer.

The metrics archive is a 5-minute *snapshot*: open interest and the long/short
ratios are stock quantities read at an instant, while the kline volume beside
them is a flow accumulated over the interval. Aggregating both the same way
would be wrong, so stocks take the last value in the bar exactly as `close`
does, and the taker volume ratio -- already a ratio of two flows -- is averaged.

Stages: resample -> panel, each resumable, run one at a time.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from config import MAX_WORKERS, atomic_save  # noqa: E402
from research.panel import CACHE, _step  # noqa: E402

ROOT = "/root/freqtrade/user_data/data/binance_public"
SRC = os.path.join(ROOT, "metrics")
OUT = os.path.join(ROOT, "metrics_resampled")
TIMEFRAMES = {"5m": "5min", "15m": "15min", "30m": "30min",
              "1h": "1h", "4h": "4h", "1d": "1D"}

# Stock quantities: the value standing at the end of the bar.
LAST = ["sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio"]
# Already a ratio of flows within each snapshot window, so averaging is the
# aggregation that keeps its meaning across a longer bar.
MEAN = ["sum_taker_long_short_vol_ratio"]
FIELDS = LAST + MEAN


def _resample_one(sym):
    path = os.path.join(SRC, f"{sym}.parquet")
    if not os.path.exists(path):
        return sym, 0, "no metrics"
    try:
        df = pd.read_parquet(path).set_index("date").sort_index()
        have_last = [c for c in LAST if c in df.columns]
        have_mean = [c for c in MEAN if c in df.columns]
        written = 0
        for tf, rule in TIMEFRAMES.items():
            parts = []
            if have_last:
                parts.append(df[have_last].resample(
                    rule, label="left", closed="left", origin="epoch").last())
            if have_mean:
                parts.append(df[have_mean].resample(
                    rule, label="left", closed="left", origin="epoch").mean())
            if not parts:
                continue
            out = pd.concat(parts, axis=1).dropna(how="all")
            if out.empty:
                continue
            os.makedirs(os.path.join(OUT, tf), exist_ok=True)
            out.reset_index().to_parquet(os.path.join(OUT, tf, f"{sym}.parquet"),
                                         compression="zstd", index=False)
            written += len(out)
        return sym, written, None
    except Exception as exc:
        return sym, 0, repr(exc)


def resample_all(symbols):
    print(f"resampling metrics for {len(symbols)} symbols ...", flush=True)
    ok = bad = miss = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, (sym, n, err) in enumerate(ex.map(_resample_one, symbols), 1):
            if err == "no metrics":
                miss += 1
            elif err:
                bad += 1
                sys.stderr.write(f"FAIL {sym}: {err}\n")
            else:
                ok += 1
            if i % 100 == 0:
                print(f"  {i}/{len(symbols)}  ok={ok} bad={bad} missing={miss}",
                      flush=True)
    print(f"metrics resample done: ok={ok} bad={bad} missing={miss}")


def build_panel(tf):
    meta = json.load(open(os.path.join(CACHE, tf, "meta.json")))
    syms = meta["symbols"]
    index = pd.date_range(meta["start"], meta["end"], freq=_step(tf), tz="UTC")
    src = os.path.join(OUT, tf)
    frames = {}
    for s in syms:
        p = os.path.join(src, f"{s}.parquet")
        if os.path.exists(p):
            frames[s] = pd.read_parquet(p).set_index("date")

    # A snapshot series is carried forward across a gap the way a price would
    # be, but only for a day: past that the contract has stopped reporting and
    # the last reading is stale rather than current.
    limit = max(1, int(pd.Timedelta(hours=24) / _step(tf)))
    d = os.path.join(CACHE, tf)
    for field in FIELDS:
        cols = {s: frames[s][field] for s in syms
                if s in frames and field in frames[s].columns}
        if not cols:
            print(f"{tf}: no source for {field}, skipped")
            continue
        wide = pd.concat(cols, axis=1).reindex(index).reindex(columns=syms)
        wide = wide.ffill(limit=limit)
        arr = wide.to_numpy(dtype="float32")
        atomic_save(os.path.join(d, f"{field}.npy"), lambda f: np.save(f, arr))
        print(f"{tf}: {field:34s} {len(cols):4d} symbols  "
              f"finite {float(np.isfinite(arr).mean()):6.2%}", flush=True)
        del wide, arr

    meta["fields"] = list(dict.fromkeys(list(meta["fields"]) + FIELDS))
    atomic_save(os.path.join(d, "meta.json"),
                lambda f: f.write(json.dumps(meta, indent=2).encode()))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    tfs = sys.argv[2:] or list(TIMEFRAMES)
    if stage in ("resample", "all"):
        syms = sorted(f[:-8] for f in os.listdir(SRC) if f.endswith(".parquet"))
        resample_all(syms)
    if stage in ("panel", "all"):
        for tf in tfs:
            build_panel(tf)
