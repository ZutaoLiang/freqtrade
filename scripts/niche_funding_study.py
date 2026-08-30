"""Funding-settlement event study on the niche 1m perp band.

For every settlement stamp: take the side that COLLECTS funding (short when the rate is
positive), enter `pre` minutes before the stamp, exit `post` minutes after, and decompose
the result into the funding leg and the price leg. Buckets by |funding rate| so the
threshold can be read off the distribution instead of guessed.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

KL = "/root/freqtrade/user_data/data/binance_public/freqtrade/futures"
FUND = "/root/freqtrade/user_data/data/binance_public/funding"
SCRATCH = "/root/freqtrade/user_data/niche_work"
WINDOWS = {"train": ("2025-11-01", "2026-05-31"), "hold": ("2026-06-01", "2026-08-16")}
LEGS = [(30, 30), (10, 10), (5, 5), (60, 60)]
BUCKETS = [10, 20, 40, 80, 1e9]  # |funding| in bps


def _run(sym):
    fsym = sym.replace("_USDT_USDT", "USDT")
    fp = f"{FUND}/{fsym}.parquet"
    if not os.path.exists(fp):
        return []
    try:
        px = pd.read_feather(f"{KL}/{sym}-1m-futures.feather").set_index("date")["close"]
        px = px[~px.index.duplicated()]
        fr = pd.read_parquet(fp).set_index("date")["funding_rate"]
    except Exception:
        return []

    rows = []
    for pre, post in LEGS:
        e_px = px.reindex(fr.index - pd.Timedelta(minutes=pre), method="ffill", limit=3)
        x_px = px.reindex(fr.index + pd.Timedelta(minutes=post), method="ffill", limit=3)
        d = pd.DataFrame({"fr": fr.values, "e": e_px.values, "x": x_px.values}, index=fr.index).dropna()
        if d.empty:
            continue
        side = -np.sign(d["fr"])                      # collect funding
        d["price_bps"] = 1e4 * side * np.log(d["x"] / d["e"])
        d["fund_bps"] = 1e4 * d["fr"].abs()
        d["tot_bps"] = d["price_bps"] + d["fund_bps"]
        d["absfr"] = 1e4 * d["fr"].abs()
        for label, (a, b) in WINDOWS.items():
            w = d.loc[a:b]
            for i, hi in enumerate(BUCKETS):
                lo = BUCKETS[i - 1] if i else 0.0
                g = w[(w.absfr >= lo) & (w.absfr < hi)]
                if len(g) < 5:
                    continue
                rows.append(dict(sym=sym, window=label, pre=pre, post=post,
                                 bucket=f"{lo:g}-{hi:g}", n=len(g),
                                 fund=g.fund_bps.mean(), price=g.price_bps.mean(),
                                 tot=g.tot_bps.mean(), med=g.tot_bps.median(),
                                 hit=(g.tot_bps > 0).mean()))
    return rows


def main():
    u = pd.read_csv(f"{SCRATCH}/universe.csv")
    syms = u[(u.med_qv > 3e6) & (u.med_qv < 6e7) & (u.days >= 250)].sym.tolist()
    print(f"pairs: {len(syms)}", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        for rows in ex.map(_run, syms, chunksize=2):
            out.extend(rows)
    d = pd.DataFrame(out)
    d.to_csv(f"{SCRATCH}/funding_events.csv", index=False)
    for (w, pre, bucket), g in d.groupby(["window", "pre", "bucket"]):
        n = g.n.sum()
        wm = lambda c: (g[c] * g.n).sum() / n
        print(f"{w:5s} pre/post={pre:2d}m |fr|{bucket:>8s}bps n={n:6d} "
              f"fund={wm('fund'):+6.1f} price={wm('price'):+6.1f} "
              f"tot={wm('tot'):+6.1f}bps hit={wm('hit'):.3f}")


if __name__ == "__main__":
    main()
