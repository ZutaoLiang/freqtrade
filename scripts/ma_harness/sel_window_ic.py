"""How fast should the mover screen look?

A seven-day volatility window may already be stale when a move lasts two or
three days. This sweeps the lookback used to rank movers against several
forward horizons, and reports both the rank IC and what the selected basket
actually got.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/root/freqtrade/user_data/factor_research")
from research.panel import Panel  # noqa: E402

BPD = {"15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--step-days", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=150)
    ap.add_argument("--top-vol", type=int, default=30)
    ap.add_argument("--windows", default="1,2,3,5,7,14")
    ap.add_argument("--horizons", default="2,3,5,7,14,30")
    a = ap.parse_args()
    wins = [int(x) for x in a.windows.split(",")]
    hors = [int(x) for x in a.horizons.split(",")]

    p = Panel(a.tf, mmap=True)
    bpd = BPD[a.tf]
    close, qv, mask = p["close"], p["quote_volume"], p["universe_mask"]
    maxh = max(hors) * bpd

    rows = []
    for t in range(90 * bpd, p.shape[0] - maxh, a.step_days * bpd):
        dollar = np.nansum(qv[t - 30 * bpd:t], axis=0)
        ok = np.isfinite(close[t - 90 * bpd:t]).all(axis=0) & mask[t - 1]
        dollar = np.where(ok, dollar, -1.0)
        liq = np.argsort(-dollar)[:a.top_n]
        liq = liq[dollar[liq] > 0]
        past = np.asarray(close[t - 14 * bpd:t, liq], dtype=np.float64)
        fwd = np.asarray(close[t:t + maxh, liq], dtype=np.float64)
        good = np.isfinite(past).all(axis=0) & np.isfinite(fwd).all(axis=0)
        liq, past, fwd = liq[good], past[:, good], fwd[:, good]
        if liq.size < 40:
            continue
        vols = {w: np.std(np.diff(np.log(past[-w * bpd:]), axis=0), axis=0)
                for w in wins}
        for h in hors:
            f = fwd[:h * bpd]
            up = (f / np.minimum.accumulate(f, axis=0)).max(axis=0) - 1.0
            dn = 1.0 - (f / np.maximum.accumulate(f, axis=0)).min(axis=0)
            swing = np.maximum(up, dn)
            daily = f[::bpd]
            r = np.diff(np.log(daily), axis=0)
            denom = np.abs(r).sum(axis=0)
            eff = np.where(denom > 0, np.abs(r.sum(axis=0)) / denom, np.nan)
            for w in wins:
                v = vols[w]
                sel = np.argsort(-v)[:a.top_vol]
                rows.append(dict(
                    t=t, win=w, hor=h,
                    ic_swing=stats.spearmanr(v, swing).statistic,
                    ic_eff=stats.spearmanr(v, eff, nan_policy="omit").statistic,
                    sel_swing=float(np.median(swing[sel])),
                    sel_eff=float(np.nanmedian(eff[sel])),
                    all_swing=float(np.median(swing)),
                ))
    df = pd.DataFrame(rows)
    print(f"{df.t.nunique()} selection dates, step {a.step_days}d, "
          f"top {a.top_vol} of {a.top_n}\n")
    for metric, label in (("ic_swing", "rank IC vs largest swing in the window"),
                          ("ic_eff", "rank IC vs trend cleanliness"),
                          ("sel_swing", "median swing the selected basket got"),
                          ("sel_eff", "median trend cleanliness of the basket")):
        piv = df.pivot_table(index="win", columns="hor", values=metric)
        print(label)
        print((piv.round(3) if metric.startswith("ic") else
               (100 * piv).round(1)).to_string())
        print()
    piv = df.pivot_table(index="win", columns="hor", values="all_swing")
    print("median swing of the whole 150 for reference (%)")
    print((100 * piv).round(1).iloc[0].to_string())


if __name__ == "__main__":
    main()
