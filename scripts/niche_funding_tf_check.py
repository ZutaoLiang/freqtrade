"""Does this actually need 1m candles?

Entries sit on funding settlement stamps, which land on 5m/15m/1h boundaries anyway, so 1m
can only matter for how quickly the close-based stop reacts. This re-runs the same events
checking the stop on 1m / 5m / 15m / 1h closes, holding to the same 235 minute cap.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

KL = "/root/freqtrade/user_data/data/binance_public/freqtrade/futures"
EVENTS = "/root/freqtrade/user_data/niche_work/universe_events.parquet"
HOLD = 235
STOP = 0.15
COST = 0.0020
LIQ = 0.99
STEPS = (1, 5, 15, 60)


def _run(args):
    sym, ev = args
    try:
        df = pd.read_feather(f"{KL}/{sym}-1m-futures.feather").set_index("date")
        df = df[~df.index.duplicated()]
    except Exception:
        return None
    opens, closes, idx = df["open"].values, df["close"].values, df.index
    span = HOLD + 2
    rows = []
    for e in ev.itertuples():
        p = idx.searchsorted(e.date)
        if p >= len(idx) - span or idx[p] != e.date:
            continue
        entry = opens[p]
        seg = e.side * (closes[p:p + HOLD + 1] / entry - 1.0)
        nxt = e.side * (opens[p + 1:p + HOLD + 2] / entry - 1.0)
        rec = dict(sym=sym, date=e.date)
        for step in STEPS:
            # a `step`-minute candle only closes every `step` bars; the stop cannot see
            # anything in between, and fills on the next bar's open
            checks = np.arange(step - 1, HOLD + 1, step)
            hit = checks[seg[checks] <= -STOP]
            ret = nxt[hit[0]] if len(hit) else seg[-1]
            rec[f"tf{step}"] = max(ret, -LIQ) - COST
        rows.append(rec)
    return pd.DataFrame(rows) if rows else None


def main():
    d = pd.read_parquet(EVENTS)
    d = d[(d.trail_qv >= 1e7) & (d.fr_bps.abs() >= 40) & (d.streak_40 >= 1)]
    groups = [(s, g) for s, g in d.groupby("sym")]
    out = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        for r in ex.map(_run, groups, chunksize=2):
            if r is not None:
                out.append(r)
    r = pd.concat(out, ignore_index=True)
    r["q"] = pd.PeriodIndex(r.date, freq="Q").astype(str)
    print(f"n={len(r)} events\n")
    print(f"{'止损判定粒度':>12} {'单笔均值':>10} {'中位':>9} {'最差单笔':>9} {'为正季度':>9}")
    for step in STEPS:
        c = f"tf{step}"
        byq = r.groupby("q")[c].sum()
        print(f"{step:>10}m {1e4*r[c].mean():>+9.1f}bps {1e4*r[c].median():>+8.1f} "
              f"{100*r[c].min():>+8.1f}% {(byq > 0).sum():>6}/{len(byq)}")


if __name__ == "__main__":
    main()
