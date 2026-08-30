"""Follow-the-funding momentum: trade WITH the crowd after a funding settlement.

The carry study showed the funding-collecting side loses more on price than it collects,
so this measures the mirror trade: at settlement T take the side that PAYS (long when the
rate is positive), enter at T+1m open so no funding is paid, and exit after `h` minutes.
Bucketed by |funding rate| in bps, train/holdout reported separately.
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
HORIZONS = (5, 15, 30, 60, 120, 240)
BUCKETS = [20, 40, 80, 160, 1e9]


def _run(sym):
    fsym = sym.replace("_USDT_USDT", "USDT")
    fp = f"{FUND}/{fsym}.parquet"
    if not os.path.exists(fp):
        return []
    try:
        df = pd.read_feather(f"{KL}/{sym}-1m-futures.feather").set_index("date")
        df = df[~df.index.duplicated()]
        fr = pd.read_parquet(fp).set_index("date")["funding_rate"]
    except Exception:
        return []

    entry = df["open"].reindex(fr.index + pd.Timedelta(minutes=1), method="ffill", limit=3)
    d = pd.DataFrame({"fr": fr.values, "e": entry.values}, index=fr.index)
    side = np.sign(d["fr"])                    # pay funding = go with the crowd
    d["absfr"] = 1e4 * d["fr"].abs()
    for h in HORIZONS:
        x = df["close"].reindex(fr.index + pd.Timedelta(minutes=1 + h), method="ffill", limit=3)
        d[f"bps_{h}"] = 1e4 * side.values * np.log(x.values / d["e"].values)
    d = d.dropna(subset=["e"])

    rows = []
    for label, (a, b) in WINDOWS.items():
        w = d.loc[a:b]
        for i, hi in enumerate(BUCKETS):
            lo = BUCKETS[i - 1] if i else 0.0
            g = w[(w.absfr >= lo) & (w.absfr < hi)]
            if len(g) < 5:
                continue
            row = dict(sym=sym, window=label, bucket=f"{lo:g}-{hi:g}", n=len(g))
            for h in HORIZONS:
                col = f"bps_{h}"
                row[col] = g[col].mean()
                row[f"hit_{h}"] = (g[col] > 0).mean()
            rows.append(row)
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
    d.to_csv(f"{SCRATCH}/funding_momentum.csv", index=False)
    order = {f"{(BUCKETS[i-1] if i else 0):g}-{hi:g}": i for i, hi in enumerate(BUCKETS)}
    for w in ("train", "hold"):
        for bucket in sorted(order, key=order.get):
            g = d[(d.window == w) & (d.bucket == bucket)]
            if g.empty:
                continue
            n = g.n.sum()
            parts = [f"{w:5s} |fr|{bucket:>9s}bps n={n:6d}"]
            for h in HORIZONS:
                parts.append(f"{h:3d}m:{(g[f'bps_{h}']*g.n).sum()/n:+7.1f}"
                             f"/{(g[f'hit_{h}']*g.n).sum()/n:.2f}")
            print("  ".join(parts))


if __name__ == "__main__":
    main()
