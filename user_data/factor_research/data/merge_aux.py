"""Fold funding rate and mark price into the existing panel layer.

Both series arrive on their own grids -- funding at 4h/8h settlement stamps,
mark price at 1m -- so they are first resampled onto each research timeframe,
then aligned to the panel's frozen index and symbol order and written as extra
fields alongside the kline fields. Panels are appended to rather than rebuilt:
the kline fields are already verified on disk and rebuilding them would risk
the same partial-write failure for no gain.

Stages: resample -> panel, each runnable alone and each resumable.
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
FUNDING = os.path.join(ROOT, "funding")
MARK = os.path.join(ROOT, "markprice_1m")
OUT = os.path.join(ROOT, "aux_resampled")
TIMEFRAMES = {"5m": "5min", "15m": "15min", "30m": "30min",
              "1h": "1h", "4h": "4h", "1d": "1D"}
FIELDS = ["mark_close", "funding_rate", "funding_interval_hours"]

# Funding is a step function between settlements, so it must be carried forward
# to every bar. Capping the carry at 12h stops a delisted pair's last rate from
# propagating to the end of the panel: the longest real interval is 8h, so any
# gap wider than this means the pair stopped settling, not that the rate held.
FFILL_HOURS = 12


def _resample_one(sym):
    """Put both aux series on every timeframe grid, still sparse for funding."""
    fp = os.path.join(FUNDING, f"{sym}.parquet")
    mp = os.path.join(MARK, f"{sym}.parquet")
    if not os.path.exists(fp) and not os.path.exists(mp):
        return sym, 0, "no aux data"
    try:
        fund = mark = None
        if os.path.exists(fp):
            fund = pd.read_parquet(fp).set_index("date").sort_index()
        if os.path.exists(mp):
            mark = pd.read_parquet(mp, columns=["date", "mark_close"])
            mark = mark.set_index("date").sort_index()
        written = 0
        for tf, rule in TIMEFRAMES.items():
            parts = []
            if mark is not None and len(mark):
                # Last 1m mark inside the bar, matching how close is aggregated.
                parts.append(mark["mark_close"].resample(
                    rule, label="left", closed="left", origin="epoch").last())
            if fund is not None and len(fund):
                # Last settlement inside the bar; several only collide at 1d.
                g = fund[["funding_rate", "funding_interval_hours"]].resample(
                    rule, label="left", closed="left", origin="epoch").last()
                parts.append(g)
            if not parts:
                continue
            out = pd.concat(parts, axis=1).dropna(how="all")
            if out.empty:
                continue
            os.makedirs(os.path.join(OUT, tf), exist_ok=True)
            out.reset_index().to_parquet(
                os.path.join(OUT, tf, f"{sym}.parquet"),
                compression="zstd", index=False)
            written += len(out)
        return sym, written, None
    except Exception as exc:
        return sym, 0, repr(exc)


def resample_all(symbols):
    print(f"resampling aux for {len(symbols)} symbols ...", flush=True)
    ok = bad = miss = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, (sym, n, err) in enumerate(ex.map(_resample_one, symbols), 1):
            if err == "no aux data":
                miss += 1
            elif err:
                bad += 1
                sys.stderr.write(f"FAIL {sym}: {err}\n")
            else:
                ok += 1
            if i % 100 == 0:
                print(f"  {i}/{len(symbols)}  ok={ok} bad={bad} missing={miss}",
                      flush=True)
    print(f"aux resample done: ok={ok} bad={bad} missing={miss}")


def _panel_index(tf):
    meta = json.load(open(os.path.join(CACHE, tf, "meta.json")))
    index = pd.date_range(meta["start"], meta["end"], freq=_step(tf), tz="UTC")
    return meta, index


def build_panel(tf):
    """Append the aux fields to an existing panel, one field at a time."""
    meta, index = _panel_index(tf)
    syms = meta["symbols"]
    src = os.path.join(OUT, tf)
    frames = {}
    for s in syms:
        p = os.path.join(src, f"{s}.parquet")
        if os.path.exists(p):
            frames[s] = pd.read_parquet(p).set_index("date")

    limit = max(1, int(pd.Timedelta(hours=FFILL_HOURS) / _step(tf)))
    d = os.path.join(CACHE, tf)
    covered = 0
    for field in FIELDS:
        cols = {s: frames[s][field] for s in syms
                if s in frames and field in frames[s].columns}
        if not cols:
            print(f"{tf}: no source for {field}, skipped")
            continue
        wide = pd.concat(cols, axis=1).reindex(index).reindex(columns=syms)
        if field.startswith("funding"):
            wide = wide.ffill(limit=limit)
        arr = wide.to_numpy(dtype="float32")
        atomic_save(os.path.join(d, f"{field}.npy"), lambda f: np.save(f, arr))
        covered = max(covered, len(cols))
        cover = float(np.isfinite(arr).mean())
        print(f"{tf}: {field:24s} {len(cols):4d} symbols  finite {cover:6.2%}")
        del wide, arr

    meta["fields"] = list(dict.fromkeys(list(meta["fields"]) + FIELDS))
    atomic_save(os.path.join(d, "meta.json"),
                lambda f: f.write(json.dumps(meta, indent=2).encode()))
    return covered


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    tfs = sys.argv[2:] or list(TIMEFRAMES)
    if stage in ("resample", "all"):
        syms = sorted({f[:-8] for d in (FUNDING, MARK) if os.path.isdir(d)
                       for f in os.listdir(d) if f.endswith(".parquet")})
        resample_all(syms)
    if stage in ("panel", "all"):
        for tf in tfs:
            build_panel(tf)
