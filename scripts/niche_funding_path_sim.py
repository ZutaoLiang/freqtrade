"""Path simulator for the funding-skew trade: pick the stop rule before rerunning Freqtrade.

The Freqtrade run showed the signal works (1522 normal trades at +24bps net) while five
short squeezes wiped the book, and that an intrabar -25% stop is scalped by 1m wicks and
then reverses. So this walks the actual minute closes after each entry and compares
CLOSE-based stops (a stop that a wick cannot touch) across levels and hold lengths, in
simple returns net of round-trip cost.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

KL = "/root/freqtrade/user_data/data/binance_public/freqtrade/futures"
FUND = "/root/freqtrade/user_data/data/binance_public/funding"
SCRATCH = "/root/freqtrade/user_data/niche_work"
MIN_ABS_FR = 30.0
HOLDS = (60, 120, 235)
STOPS = (0.10, 0.15, 0.25, None)
COST = 0.0020            # 10bps per side: taker fee + half-spread injection
LIQ = 0.99               # isolated 1x: the position is gone before this


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

    idx = df.index
    pos = idx.searchsorted(fr.index + pd.Timedelta(minutes=1))
    ok = (pos < len(idx) - max(HOLDS)) & (idx[np.clip(pos, 0, len(idx) - 1)]
                                          == fr.index + pd.Timedelta(minutes=1))
    pos, rates = pos[ok], fr.values[ok]
    if len(pos) == 0:
        return None

    opens = df["open"].values
    closes = df["close"].values
    rows = []
    for p, rate in zip(pos, rates):
        side = 1.0 if rate > 0 else -1.0
        entry = opens[p]
        n = max(HOLDS) + 2
        path = side * (closes[p:p + n] / entry - 1.0)        # signed simple return on closes
        # a close-based stop is acted on at the NEXT bar's open, as custom_exit does
        nxt = side * (opens[p + 1:p + n + 1] / entry - 1.0)
        rec = dict(sym=sym, date=idx[p], fr_bps=1e4 * rate, side=side)
        for hold in HOLDS:
            seg = path[:hold + 1]
            for stop in STOPS:
                if stop is None:
                    ret = seg[-1]
                else:
                    hit = np.flatnonzero(seg <= -stop)
                    ret = nxt[hit[0]] if len(hit) else seg[-1]
                rec[f"h{hold}_s{'none' if stop is None else int(stop*100)}"] = \
                    max(ret, -LIQ) - COST
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    u = pd.read_csv(f"{SCRATCH}/universe.csv")
    syms = u[(u.med_qv > 3e6) & (u.med_qv < 6e7) & (u.days >= 250)].sym.tolist()
    out = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        for d in ex.map(_run, syms, chunksize=2):
            if d is not None:
                out.append(d)
    d = pd.concat(out)
    d = d[d.date >= "2025-11-01"]
    d.to_parquet(f"{SCRATCH}/path_sim.parquet")

    d["win"] = np.where(d.date <= "2026-05-31", "train", "hold")
    print(f"events: {len(d)}  train={int((d.win=='train').sum())} hold={int((d.win=='hold').sum())}")
    print("net simple return per trade, bps (cost 20bps round trip):\n")
    for hold in HOLDS:
        for stop in STOPS:
            col = f"h{hold}_s{'none' if stop is None else int(stop*100)}"
            parts = [f"hold={hold:3d}m stop={'none' if stop is None else f'-{stop:.0%}':>5s}"]
            for w in ("train", "hold"):
                g = d[d.win == w]
                parts.append(f"{w}: mean={1e4*g[col].mean():+7.1f} med={1e4*g[col].median():+6.1f} "
                             f"hit={(g[col] > 0).mean():.3f} worst={100*g[col].min():+6.1f}%")
            print("  ".join(parts))
        print()


if __name__ == "__main__":
    main()
