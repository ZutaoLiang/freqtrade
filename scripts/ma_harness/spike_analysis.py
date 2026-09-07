"""Split the harness failure into its two possible causes.

Two different things can be wrong with "screen for the coins that are about to
run, then ride them with a crossover":

  selection   the screen never picks the coins that actually move
  capture     the screen picks them, but the crossover gives the move back

This measures both on the same rebalance dates the walk-forward used. For every
candidate coin it computes a set of cross-sectional predictors known at the
decision bar, and a set of outcomes over the untouched window that follows --
how far the coin travelled, how cleanly it trended, and how much of its best
swing the coin's own selected crossover actually kept.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/root/freqtrade/user_data/factor_research")
from research.panel import Panel  # noqa: E402

BPD = {"5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}


def best_swings(close):
    """Largest up swing and largest down swing available inside a window."""
    run_min = np.minimum.accumulate(close)
    run_max = np.maximum.accumulate(close)
    up = float((close / run_min).max() - 1.0)      # best long, entry to exit
    dn = float(1.0 - (close / run_max).min())      # best short
    return up, dn


def outcomes(close):
    r = np.diff(np.log(close))
    total = float(close[-1] / close[0] - 1.0)
    up, dn = best_swings(close)
    denom = np.abs(r).sum()
    return dict(
        ret=total,
        absret=abs(total),
        rng=float(close.max() / close.min() - 1.0),
        best_swing=max(up, dn),
        efficiency=float(abs(r.sum()) / denom) if denom > 0 else 0.0,
        realised_vol=float(r.std() * np.sqrt(len(r))),
    )


def predictors(p, t, cols, bpd, cfg):
    """Everything a screen could know at the decision bar."""
    close = p["close"]
    qv = p["quote_volume"]
    oi = p["sum_open_interest_value"]
    fr = p["funding_rate"]
    d30, d7, d90 = 30 * bpd, 7 * bpd, 90 * bpd
    c = np.asarray(close[t - d90:t, cols], dtype=np.float64)
    v = np.asarray(qv[t - d90:t, cols], dtype=np.float64)
    o = np.asarray(oi[t - d90:t, cols], dtype=np.float64)
    f = np.asarray(fr[t - d30:t, cols], dtype=np.float64)
    r = np.diff(np.log(c), axis=0)
    out = {}
    out["dollar_vol_30d"] = np.nansum(v[-d30:], axis=0)
    out["vol_surge"] = (np.nansum(v[-d7:], axis=0) / 7.0) / \
                       (np.nansum(v[-d30:], axis=0) / 30.0)
    out["realised_vol_7d"] = np.nanstd(r[-d7:], axis=0) * np.sqrt(d7)
    out["realised_vol_30d"] = np.nanstd(r[-d30:], axis=0) * np.sqrt(d30)
    out["vol_of_vol"] = (out["realised_vol_7d"] / out["realised_vol_30d"])
    out["mom_7d"] = c[-1] / c[-d7] - 1.0
    out["mom_30d"] = c[-1] / c[-d30] - 1.0
    out["mom_90d"] = c[-1] / c[0] - 1.0
    out["efficiency_30d"] = np.abs(r[-d30:].sum(axis=0)) / \
        np.abs(r[-d30:]).sum(axis=0)
    out["oi_change_7d"] = o[-1] / np.where(o[-d7] > 0, o[-d7], np.nan) - 1.0
    out["oi_over_vol"] = o[-1] / (np.nansum(v[-d30:], axis=0) / 30.0)
    out["abs_funding"] = np.nanmean(np.abs(f), axis=0)
    out["funding"] = np.nanmean(f, axis=0)
    out["price_pctile_90d"] = (c[-1] - np.nanmin(c, axis=0)) / \
        (np.nanmax(c, axis=0) - np.nanmin(c, axis=0) + 1e-12)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", required=True)
    ap.add_argument("--horizon-days", type=int, default=7)
    a = ap.parse_args()
    d = json.load(open(a.wf))
    cfg, steps = d["config"], d["steps"]
    tf, k = cfg["tf"], cfg["top_k"]
    bpd = BPD[tf]
    p = Panel(tf, mmap=True)
    close = p["close"]
    h = a.horizon_days * bpd

    recs = []
    for st in steps:
        t = st["t"]
        if t + h >= p.shape[0]:
            continue
        rows = sorted(st["rows"], key=lambda r: r["rank"])
        cols = [r["col"] for r in rows]
        pred = predictors(p, t, cols, bpd, cfg)
        fwd = np.asarray(close[t:t + h, cols], dtype=np.float64)
        for j, r in enumerate(rows):
            c = fwd[:, j]
            if not np.isfinite(c).all():
                continue
            rec = dict(t=t, symbol=r["symbol"], rank=r["rank"], score=r["score"],
                       slow=r["slow"], strat=r["oos_total"],
                       strat_fixed=r.get("oos_total_fixed"))
            rec.update({key: float(val[j]) for key, val in pred.items()})
            rec.update(outcomes(c))
            recs.append(rec)
    df = pd.DataFrame(recs)
    print(f"{len(df)} coin-windows over {df.t.nunique()} rebalances, "
          f"horizon {a.horizon_days}d\n")

    targets = ["best_swing", "rng", "absret", "efficiency", "strat"]
    preds = ["score"] + [c for c in df.columns if c in (
        "dollar_vol_30d", "vol_surge", "realised_vol_7d", "realised_vol_30d",
        "vol_of_vol", "mom_7d", "mom_30d", "mom_90d", "efficiency_30d",
        "oi_change_7d", "oi_over_vol", "abs_funding", "funding",
        "price_pctile_90d")]

    print("mean rank IC by rebalance (t stat in brackets)")
    print(f"{'predictor':<20}" + "".join(f"{x:>22}" for x in targets))
    for pr in preds:
        line = f"{pr:<20}"
        for tg in targets:
            ics = []
            for _, g in df.groupby("t"):
                x, y = g[pr].to_numpy(), g[tg].to_numpy()
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() > 20:
                    ics.append(stats.spearmanr(x[m], y[m]).statistic)
            ics = np.array(ics)
            tt = ics.mean() / (ics.std(ddof=1) / np.sqrt(ics.size))
            line += f"{ics.mean():>+11.3f} [{tt:>+5.1f}]  "
        print(line)

    print("\ncapture: what the crossover keeps of the swing that was there")
    for label, sel in (("whitelist top-%d" % k, df["rank"] < k),
                       ("rank 10-49", (df["rank"] >= 10) & (df["rank"] < 50)),
                       ("whole universe", df["rank"] >= 0)):
        g = df[sel]
        cap = g["strat"] / g["best_swing"].replace(0, np.nan)
        print(f"  {label:<18} best swing available {100*g.best_swing.mean():6.1f}%   "
              f"strategy {100*g.strat.mean():+6.2f}%   "
              f"capture ratio mean {cap.mean():+.3f} median {cap.median():+.3f}   "
              f"efficiency {g.efficiency.mean():.3f}")

    print("\nis the whitelist even picking the movers?")
    for col in ("best_swing", "rng", "realised_vol_7d", "efficiency"):
        top = df[df["rank"] < k][col].mean()
        rest = df[df["rank"] >= k][col].mean()
        _, pv = stats.mannwhitneyu(df[df["rank"] < k][col],
                                   df[df["rank"] >= k][col])
        print(f"  {col:<18} whitelist {top:8.4f}   rest {rest:8.4f}   "
              f"ratio {top/rest:5.2f}   Mann-Whitney p {pv:.3f}")

    out = a.wf.replace(".json", "_spike.parquet")
    df.to_parquet(out)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
