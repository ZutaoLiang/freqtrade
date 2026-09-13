"""Apply the pre-registered gates (reports/PREREGISTRATION_ts_screen.md) to the ts sweeps.

Stage 1 (train): n_trades>=300, mean_net5>=30bps, mean_net10>=0, NW t>=3, >=3/4 quarters positive.
Stage 2 (valid): same sign, mean_net5>0, NW t>=1.5, n_trades>=60.
Stage 3 (hist 2022-11..2024-12, when available): reported, pass = mean_net5>0 and NW t>=1.5.
Also reports the neighbourhood (other holds / windows of the same family in the same screen)
and counts survivors per stage so the base rate is visible.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
KEY = ["name", "screen", "side", "hold_h"]


def load(label, tf, split):
    p = os.path.join(REPORTS, f"ts_{label}_{tf}_{split}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    return d[d.n_trades.notna() & (d.n_trades > 0)].copy() if "n_trades" in d else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="r2")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--hist-tf", default="1h_hist")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--splits", default="train,valid,hist", help="split names for stage 1,2,3")
    ap.add_argument("--n1", type=int, default=300); ap.add_argument("--q1", type=int, default=3); ap.add_argument("--n2", type=int, default=60); ap.add_argument("--t3", type=float, default=1.5)
    a = ap.parse_args()
    pd.set_option("display.width", 260)
    s1, s2, s3 = a.splits.split(",")
    tr = load(a.label, a.tf, s1)
    print(f"train cells: {len(tr)}; specs {tr.name.nunique()}, screens {tr.screen.nunique()}")
    g1 = tr[(tr.n_trades >= a.n1) & (tr.mean_net5_bps >= 30) & (tr.mean_net10_bps >= 0) & (tr.nw_t_daily >= 3) & (tr.quarters_pos >= a.q1)].copy()
    print(f"\nSTAGE 1 survivors: {len(g1)} cells, {g1.name.nunique()} specs; by screen x side:")
    print(g1.groupby(["screen", "side"]).size().unstack(fill_value=0).to_string())
    print("by tag:", g1.tag.value_counts().to_dict())
    cols = ["name", "screen", "side", "hold_h", "n_trades", "n_symbols", "mean_net5_bps", "median_net5_bps", "win", "nw_t_daily", "quarters_pos", "max_concurrent", "q_str"]
    print(g1.sort_values("nw_t_daily", ascending=False)[cols].head(a.top).round(1).to_string(index=False))
    va = load(a.label, a.tf, s2)
    if va is None:
        print("\n(valid sweep not available yet)"); return
    m = g1.merge(va[KEY + ["n_trades", "mean_net5_bps", "median_net5_bps", "win", "nw_t_daily", "quarters_pos", "q_str"]], on=KEY, how="left", suffixes=("", "_v"))
    g2 = m[(m.n_trades_v >= a.n2) & (m.mean_net5_bps_v > 0) & (m.nw_t_daily_v >= 1.5)].copy()
    print(f"\nSTAGE 2 survivors (valid): {len(g2)} of {len(g1)} cells, {g2.name.nunique()} specs; by screen x side:")
    if len(g2):
        print(g2.groupby(["screen", "side"]).size().unstack(fill_value=0).to_string())
        c2 = ["name", "screen", "side", "hold_h", "n_trades", "mean_net5_bps", "median_net5_bps", "nw_t_daily", "n_trades_v", "mean_net5_bps_v", "median_net5_bps_v", "win_v", "nw_t_daily_v", "q_str_v"]
        print(g2.sort_values("nw_t_daily_v", ascending=False)[c2].head(a.top).round(1).to_string(index=False))
    # base rate: how many random cells would pass stage 2 given stage 1? compare against cells that FAILED stage 1 narrowly
    hi = load(a.label, a.hist_tf, s3)
    if hi is None or not len(g2):
        print("\n(hist sweep not available yet)"); g2.to_csv(os.path.join(REPORTS, f"ts_{a.label}_stage2.csv"), index=False); return
    m3 = g2.merge(hi[KEY + ["n_trades", "mean_net5_bps", "median_net5_bps", "win", "nw_t_daily", "quarters_pos", "y_str"]], on=KEY, how="left", suffixes=("", "_h"))
    m3["hist_pass"] = (m3.mean_net5_bps_h > 0) & (m3.nw_t_daily_h >= a.t3)
    print(f"\nSTAGE 3 (2022-11..2024-12): {int(m3.hist_pass.sum())} of {len(m3)} stage-2 cells pass; {int(m3.mean_net5_bps_h.isna().sum())} not computable on the hist panel")
    c3 = ["name", "screen", "side", "hold_h", "mean_net5_bps", "nw_t_daily", "mean_net5_bps_v", "nw_t_daily_v", "n_trades_h", "mean_net5_bps_h", "median_net5_bps_h", "win_h", "nw_t_daily_h", "y_str"]
    print(m3.sort_values(["hist_pass", "nw_t_daily_h"], ascending=False)[c3].head(a.top).round(1).to_string(index=False))
    m3.to_csv(os.path.join(REPORTS, f"ts_{a.label}_stage3.csv"), index=False)


if __name__ == "__main__":
    main()
