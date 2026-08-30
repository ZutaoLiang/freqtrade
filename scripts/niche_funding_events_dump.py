"""Dump event-level rows for the follow-the-funding trade so concentration can be checked.

One row per settlement with |funding| >= MIN_ABS_FR bps: entry at T+1m open, forward
returns in the crowd direction at several horizons. Aggregation happens downstream.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

KL = "/root/freqtrade/user_data/data/binance_public/freqtrade/futures"
FUND = "/root/freqtrade/user_data/data/binance_public/funding"
SCRATCH = "/root/freqtrade/user_data/niche_work"
HORIZONS = (30, 60, 120, 240, 480)
MIN_ABS_FR = 30.0


def _run(sym):
    fsym = sym.replace("_USDT_USDT", "USDT")
    fp = f"{FUND}/{fsym}.parquet"
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_feather(f"{KL}/{sym}-1m-futures.feather").set_index("date")
        df = df[~df.index.duplicated()]
        fr = pd.read_parquet(fp).set_index("date")["funding_rate"]
    except Exception:
        return None
    fr = fr[1e4 * fr.abs() >= MIN_ABS_FR]
    if fr.empty:
        return None
    e = df["open"].reindex(fr.index + pd.Timedelta(minutes=1), method="ffill", limit=3)
    d = pd.DataFrame({"sym": sym, "fr_bps": 1e4 * fr.values, "entry": e.values}, index=fr.index)
    side = np.sign(d["fr_bps"])
    for h in HORIZONS:
        x = df["close"].reindex(fr.index + pd.Timedelta(minutes=1 + h), method="ffill", limit=3)
        d[f"bps_{h}"] = 1e4 * side.values * np.log(x.values / d["entry"].values)
        # worst adverse excursion in the trade direction, for stop sizing
        w = df["low"] if True else None
    lo = df["low"].rolling(240).min().shift(-240)
    hi = df["high"].rolling(240).max().shift(-240)
    adverse = np.where(side.values > 0,
                       1e4 * np.log(lo.reindex(fr.index + pd.Timedelta(minutes=1),
                                               method="ffill", limit=3).values / d["entry"].values),
                       -1e4 * np.log(hi.reindex(fr.index + pd.Timedelta(minutes=1),
                                                method="ffill", limit=3).values / d["entry"].values))
    d["mae_240"] = adverse
    return d.dropna(subset=["entry"]).reset_index(names="date")


def main():
    u = pd.read_csv(f"{SCRATCH}/universe.csv")
    syms = u[(u.med_qv > 3e6) & (u.med_qv < 6e7) & (u.days >= 250)].sym.tolist()
    out = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        for d in ex.map(_run, syms, chunksize=2):
            if d is not None:
                out.append(d)
    d = pd.concat(out).sort_values("date")
    d.to_parquet(f"{SCRATCH}/fund_events.parquet")
    print(len(d), "events", d.date.min(), d.date.max())


if __name__ == "__main__":
    main()
