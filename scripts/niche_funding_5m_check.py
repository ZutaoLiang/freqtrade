"""Cost of running this on 5m candles instead of 1m, measured on the same events.

A 5m strategy differs from the 1m one in two ways at once, and the earlier timeframe check
only covered the second:
  - entry fills at T+5m instead of T+1m (the signal candle is 5 minutes wide)
  - the close-based stop can only be evaluated every 5 minutes
The hold is also cut from 235 to 225 minutes so the exit still lands before the next 4h
settlement given the later entry.
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

KL = "/root/freqtrade/user_data/data/binance_public/freqtrade/futures"
EVENTS = "/root/freqtrade/user_data/niche_work/universe_events.parquet"
STOP = 0.15
COST = 0.0020
LIQ = 0.99
# (label, entry offset in minutes, hold in minutes, stop check step in minutes)
VARIANTS = [("1m  entry T+1  hold 235", 1, 235, 1),
            ("5m  entry T+5  hold 235", 5, 235, 5),
            ("5m  entry T+5  hold 225", 5, 225, 5),
            ("1m  entry T+5  hold 225", 5, 225, 1)]


def _run(args):
    sym, ev = args
    try:
        df = pd.read_feather(f"{KL}/{sym}-1m-futures.feather").set_index("date")
        df = df[~df.index.duplicated()]
    except Exception:
        return None
    opens, closes, idx = df["open"].values, df["close"].values, df.index
    rows = []
    for e in ev.itertuples():
        stamp = e.date - pd.Timedelta(minutes=1)          # events store the T+1m entry bar
        rec = dict(sym=sym, date=e.date)
        ok = True
        for lab, off, hold, step in VARIANTS:
            p = idx.searchsorted(stamp + pd.Timedelta(minutes=off))
            if p >= len(idx) - hold - 2 or idx[p] != stamp + pd.Timedelta(minutes=off):
                ok = False
                break
            entry = opens[p]
            seg = e.side * (closes[p:p + hold + 1] / entry - 1.0)
            nxt = e.side * (opens[p + 1:p + hold + 2] / entry - 1.0)
            checks = np.arange(step - 1, hold + 1, step)
            hit = checks[seg[checks] <= -STOP]
            ret = nxt[hit[0]] if len(hit) else seg[-1]
            rec[lab] = max(ret, -LIQ) - COST
        if ok:
            rows.append(rec)
    return pd.DataFrame(rows) if rows else None


def main():
    d = pd.read_parquet(EVENTS)
    d = d[(d.trail_qv >= 1e7) & (d.fr_bps.abs() >= 40) & (d.streak_40 >= 1)]
    out = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        for r in ex.map(_run, [(s, g) for s, g in d.groupby("sym")], chunksize=2):
            if r is not None:
                out.append(r)
    r = pd.concat(out, ignore_index=True)
    r["q"] = pd.PeriodIndex(r.date, freq="Q").astype(str)
    print(f"n={len(r)} events\n")
    print(f"{'配置':>24} {'单笔均值':>10} {'中位':>9} {'最差单笔':>9} {'为正季度':>9}")
    for lab, *_ in VARIANTS:
        byq = r.groupby("q")[lab].sum()
        print(f"{lab:>24} {1e4*r[lab].mean():>+9.1f}bps {1e4*r[lab].median():>+8.1f} "
              f"{100*r[lab].min():>+8.1f}% {(byq > 0).sum():>6}/{len(byq)}")
    a, b = VARIANTS[0][0], VARIANTS[2][0]
    print(f"\n5m 相对 1m 的差: 均值 {1e4*(r[b]-r[a]).mean():+.1f}bps  中位 {1e4*(r[b]-r[a]).median():+.1f}bps")


if __name__ == "__main__":
    main()
