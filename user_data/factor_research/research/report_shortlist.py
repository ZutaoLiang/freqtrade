"""Score a fixed shortlist on one split: IC, its shift-null z, and cost.

This is the apples-to-apples view the per-timeframe reports cannot give. Every
timeframe and every split is asked about the *same* candidates, so a blank cell
means the factor failed, never that it was crowded out of somebody's top-N.
"""
import argparse
import json
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
import ic as IC  # noqa: E402

REPORTS = os.path.join(HERE, "..", "reports")


def run(tf, names, split_name, horizon_hours=24, directions=None):
    sl, close, mask, span = FA._slice(tf, split_name)
    specs = FA.spec_index()
    h = max(1, int(round(horizon_hours * B.BARS_PER_DAY[tf] / 24.0)))
    rho, _ = IC.average_pair_correlation(
        IC.apply_mask(IC.forward_return(close, 1), mask))
    rows = []
    for name in names:
        s = specs.get(name)
        if s is None or not s.valid(tf):
            rows.append({"name": name, "tf": tf, "split": split_name,
                         "note": "spec unavailable at this timeframe"})
            continue
        f = s.compute(tf, lambda n: np.asarray(B.load(tf, n), dtype="float64"))[sl]
        r = IC.evaluate(f, close, mask, horizons=(h,), lag_bars=1, rho=rho,
                        name=name, n_draws=1)[0]
        # Orientation comes from the decile *mean* spread, not from the rank
        # IC. Those two disagree for a factor whose ranking is right about the
        # typical symbol and wrong about the tail, and it is the tail that
        # decides the P&L -- orienting by rank IC pointed several books the
        # wrong way. The direction is fixed on train and carried into valid, so
        # valid stays a genuine out-of-sample test of a decision already made.
        probe = C.quantile_profile(f, close, mask, horizon=h, lag_bars=1)
        if directions is not None and name in directions:
            direction = directions[name]
        else:
            direction = 1.0 if probe["mean_spread_bps"] >= 0 else -1.0
        c = C.evaluate(f, close, mask, lag_bars=1, hold=h, sign=direction,
                       bars_per_year=C.BARS_PER_YEAR[tf])
        qp = probe
        rows.append({
            "name": name, "tf": tf, "split": split_name, "horizon": h,
            "ic_ts": r["ic_ts"], "ic_ts_z": r["ic_ts_z"], "ic_cs": r["ic_cs"],
            "ic_cs_t": r["ic_cs_t"], "autocorr1": IC.turnover_proxy(
                IC.apply_mask(f, mask), 1),
            "hold_bars": h, "sign": c.get("sign"),
            "mean_spread_bps": qp["mean_spread_bps"],
            "median_spread_bps": qp["median_spread_bps"],
            "tail_consistent": qp["tail_consistent"],
            "turnover": c.get("turnover_per_bar"),
            "gross_bps": c.get("gross_bps_per_bar"),
            "gross_sharpe": c.get("gross_sharpe"),
            "breakeven_bps": c.get("breakeven_cost_bps"),
            "net_sharpe_2bps": c.get("net_sharpe_2bps"),
            "net_sharpe_5bps": c.get("net_sharpe_5bps"),
        })
        last = rows[-1]
        flag = "" if last["tail_consistent"] else "  TAIL-INCONSISTENT"
        print(f"  {name:40s} z {last['ic_ts_z']:6.2f} dir {int(last['sign']):+d} "
              f"turn {last['turnover']:.4f} spread mean {last['mean_spread_bps']:+7.1f} "
              f"med {last['median_spread_bps']:+7.1f} "
              f"breakeven {last['breakeven_bps']:8.2f}bps "
              f"netSR@2bps {last['net_sharpe_2bps']:6.2f}{flag}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="file with one factor name per line")
    ap.add_argument("--tfs", default="1h,4h,15m")
    ap.add_argument("--splits", default="train,valid")
    ap.add_argument("--horizon-hours", type=float, default=24.0)
    ap.add_argument("--out", default=os.path.join(REPORTS, "shortlist_scored.csv"))
    a = ap.parse_args()
    names = [ln.strip() for ln in open(a.names) if ln.strip()]
    splits = a.splits.split(",")
    frames, directions = [], {}
    for split in splits:
        for tf in a.tfs.split(","):
            print(f"=== {tf} {split} ({len(names)} candidates) ===", flush=True)
            fixed = directions.get(tf) if split != splits[0] else None
            d = run(tf, names, split, a.horizon_hours, directions=fixed)
            if split == splits[0]:
                directions[tf] = dict(zip(d["name"], d["sign"]))
            frames.append(d)
            pd.concat(frames, ignore_index=True).to_csv(a.out, index=False)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}\n")

    # Ranked by break-even cost, which is the number a decision is made on.
    # Tail-inconsistent factors are listed but never at the top: a positive
    # break-even built on a mean spread that contradicts the median spread is
    # a bet on a handful of observations.
    for split in splits:
        for tf in a.tfs.split(","):
            g = df[(df["tf"] == tf) & (df["split"] == split)].copy()
            if g.empty or "breakeven_bps" not in g:
                continue
            g = g.sort_values(["tail_consistent", "breakeven_bps"],
                              ascending=[False, False])
            cols = ["name", "sign", "turnover", "mean_spread_bps",
                    "median_spread_bps", "tail_consistent", "breakeven_bps",
                    "net_sharpe_2bps", "net_sharpe_5bps"]
            print(f"--- {tf} {split}, ranked by break-even ---")
            print(g[cols].to_string(index=False,
                                    float_format=lambda v: f"{v:.3f}"))
            print()
