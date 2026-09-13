"""The single holdout evaluation. Refuses to run twice.

Everything that constitutes a choice -- which factors, which direction, which
holding period -- is read from files written before this ran. The holdout
contributes nothing to selection; it only reports what the already-made
decision earned. Re-running against an existing result file is refused rather
than overwritten, so a second look leaves a trace instead of quietly replacing
the first.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import base as B  # noqa: E402
import cost as C  # noqa: E402
import falsify as FA  # noqa: E402

REPORTS = os.path.join(HERE, "..", "reports")


def evaluate(tf, names, signs, horizon_hours=24.0, split_name="holdout"):
    sl, close, mask, span = FA._slice(tf, split_name)
    specs = FA.spec_index()
    h = max(1, int(round(horizon_hours * B.BARS_PER_DAY[tf] / 24.0)))
    rows = []
    for name in names:
        s = specs.get(name)
        if s is None or not s.valid(tf):
            print(f"  skip {name}: unavailable at {tf}", flush=True)
            continue
        f = s.compute(tf, lambda n: np.asarray(B.load(tf, n), dtype="float64"))[sl]
        sign = signs[name]
        c = C.evaluate(f, close, mask, lag_bars=1, hold=h, sign=sign,
                       bars_per_year=C.BARS_PER_YEAR[tf])
        qp = C.quantile_profile(f, close, mask, horizon=h, lag_bars=1)
        rows.append({
            "name": name, "tf": tf, "sign": sign, "hold_bars": h,
            "turnover": c.get("turnover_per_bar"),
            "gross_bps": c.get("gross_bps_per_bar"),
            "gross_sharpe": c.get("gross_sharpe"),
            "pnl_t_nw": c.get("pnl_t_nw"),
            "breakeven_bps": c.get("breakeven_cost_bps"),
            "net_sharpe_2bps": c.get("net_sharpe_2bps"),
            "net_sharpe_5bps": c.get("net_sharpe_5bps"),
            "mean_spread_bps": qp["mean_spread_bps"],
            "median_spread_bps": qp["median_spread_bps"],
            "tail_consistent": qp["tail_consistent"],
        })
        r = rows[-1]
        print(f"  {name:38s} dir {int(sign):+d} turn {r['turnover']:.4f} "
              f"breakeven {r['breakeven_bps']:8.2f}bps t {r['pnl_t_nw']:5.2f} "
              f"SR@5bps {r['net_sharpe_5bps']:6.2f}"
              f"{'' if r['tail_consistent'] else '  TAIL-INCONSISTENT'}", flush=True)
    return pd.DataFrame(rows), span


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=os.path.join(REPORTS, "shortlist_holdout.txt"))
    ap.add_argument("--signs-from", default=os.path.join(REPORTS, "cost_round3_1h.csv"),
                    help="train rows of a cost sweep; the direction comes from there")
    ap.add_argument("--tfs", default="1h,4h")
    ap.add_argument("--horizon-hours", type=float, default=24.0)
    ap.add_argument("--out", default=os.path.join(REPORTS, "HOLDOUT_RESULT.csv"))
    a = ap.parse_args()

    if os.path.exists(a.out):
        sys.exit(f"refusing: {a.out} exists -- the holdout has already been used")

    names = [l.strip() for l in open(a.names) if l.strip()]
    src = pd.read_csv(a.signs_from)
    src = src[src["split"] == "train"].set_index("name")
    signs = {n: float(src.loc[n, "sign"]) for n in names if n in src.index}
    missing = [n for n in names if n not in signs]
    if missing:
        sys.exit(f"refusing: no train-fixed direction for {missing}")

    print(f"{len(names)} factors, directions fixed on train, "
          f"holding {a.horizon_hours}h\n")
    frames = []
    for tf in a.tfs.split(","):
        df, span = evaluate(tf, names, signs, a.horizon_hours)
        print(f"--- {tf} holdout {span[0].date()}..{span[1].date()} ---\n")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
