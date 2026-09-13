"""H1 and H2 from reports/PREREGISTRATION_vol_regime.md.

H1: across the whole grid, is mean net return in the HIGH bucket above the LOW bucket, after removing
what the market itself did in those buckets, and does the sign hold on valid and holdout?

H2: restricted to HIGH entries, do any cells pass the three-stage gates while their LOW bucket does not?
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import screens as S  # noqa: E402
import vol_regime as VR  # noqa: E402
from panel import Panel  # noqa: E402

REPORTS = os.path.join(HERE, "..", "reports")
SPLITS = ["train", "valid", "holdout"]
KEY = ["name", "screen", "side", "hold_h"]
MIN_N = 50


def market_by_regime(tf, holds_h, bpd, regime_kw):
    """Equal-weight buy-and-hold return of each screen over the same holds, by regime and split.
    This is what a random entry inside the screen would have earned; the factor must beat it."""
    p = Panel(tf)
    open_ = np.asarray(p["open"], dtype="float64")
    state = VR.states_for(tf, **regime_kw)
    idx = p.index
    split_id = np.full(len(idx), "none", dtype=object)
    for name, (lo, hi) in {"train": ("2022-11-01", "2025-01-01"), "valid": ("2025-01-01", "2026-01-01"),
                           "holdout": ("2026-01-01", "2026-08-17")}.items():
        split_id[(idx >= pd.Timestamp(lo, tz="UTC")) & (idx < pd.Timestamp(hi, tz="UTC"))] = name
    scr = S.load(tf)
    rows = []
    for sname, m in scr.items():
        m = np.asarray(m)
        for hh in holds_h:
            h = max(1, int(round(hh * bpd / 24)))
            fwd = np.full_like(open_, np.nan)
            fwd[: -h - 1] = open_[h + 1:] / open_[1:-h] - 1.0      # entry next open, exit h bars later
            v = np.where(m, fwd, np.nan)
            mean_by_bar = np.nanmean(v, axis=1)
            for reg in ("HIGH", "LOW", "MID"):
                for sp in SPLITS:
                    sel = (state == reg) & (split_id == sp) & np.isfinite(mean_by_bar)
                    if sel.sum() > 10:
                        rows.append({"screen": sname, "hold_h": hh, "regime": reg, "split": sp,
                                     "mkt_bps": float(np.nanmean(mean_by_bar[sel]) * 1e4)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="main")
    ap.add_argument("--tf", default="1d_long")
    ap.add_argument("--holds", default="24,72,168,336,672")
    ap.add_argument("--bpd", type=int, default=1)
    ap.add_argument("--rv-days", type=int, default=VR.RV_DAYS)
    ap.add_argument("--rank-days", type=int, default=VR.RANK_DAYS)
    ap.add_argument("--lo-q", type=float, default=VR.LO_Q)
    ap.add_argument("--hi-q", type=float, default=VR.HI_Q)
    ap.add_argument("--kind", default="btc")
    ap.add_argument("--h1-only", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    d = pd.read_parquet(os.path.join(REPORTS, f"vr_{a.label}_{a.tf}.parquet"))
    d = d[d.n_trades >= MIN_N]
    holds = [int(x) for x in a.holds.split(",")]
    mkt = market_by_regime(a.tf, holds, a.bpd, dict(rv_days=a.rv_days, rank_days=a.rank_days,
                                                    lo_q=a.lo_q, hi_q=a.hi_q, kind=a.kind))

    print(f"=== market control: equal-weight screen return per trade (bps), by regime")
    piv = mkt[mkt.screen.isin(["all", "top100"])].pivot_table(index=["screen", "hold_h"], columns=["split", "regime"], values="mkt_bps")
    print(piv.reindex(columns=pd.MultiIndex.from_product([SPLITS, ["LOW", "MID", "HIGH"]]), fill_value=np.nan).round(0).to_string())

    # ---- H1
    hi = d[d.regime == "HIGH"].set_index(KEY + ["split"])
    lo = d[d.regime == "LOW"].set_index(KEY + ["split"])
    j = hi[["mean_net5_bps", "sd_net5_bps", "n_trades"]].join(
        lo[["mean_net5_bps", "sd_net5_bps", "n_trades"]], lsuffix="_hi", rsuffix="_lo", how="inner").reset_index()
    mh = mkt[mkt.regime == "HIGH"].rename(columns={"mkt_bps": "mkt_hi"}).drop(columns="regime")
    ml = mkt[mkt.regime == "LOW"].rename(columns={"mkt_bps": "mkt_lo"}).drop(columns="regime")
    j = j.merge(mh, on=["screen", "hold_h", "split"], how="left").merge(ml, on=["screen", "hold_h", "split"], how="left")
    sgn = np.where(j.side == "long", 1.0, -1.0)
    j["gap"] = j.mean_net5_bps_hi - j.mean_net5_bps_lo
    j["gap_adj"] = j.gap - sgn * (j.mkt_hi - j.mkt_lo)
    j["gap_norm"] = (j.mean_net5_bps_hi / j.sd_net5_bps_hi.replace(0, np.nan)
                     - j.mean_net5_bps_lo / j.sd_net5_bps_lo.replace(0, np.nan))
    print(f"\n=== H1: median gap across cells (bps per trade), cells with >={MIN_N} trades in both buckets")
    g = j.groupby("split").agg(cells=("gap", "size"), median_gap=("gap", "median"), pos_frac=("gap", lambda x: (x > 0).mean()),
                               median_gap_adj=("gap_adj", "median"), pos_frac_adj=("gap_adj", lambda x: (x > 0).mean()),
                               median_gap_norm=("gap_norm", "median"), pos_frac_norm=("gap_norm", lambda x: (x > 0).mean()))
    print(g.reindex(SPLITS).round(3).to_string())
    print("\n   by side:")
    print(j.groupby(["split", "side"]).agg(cells=("gap", "size"), median_gap=("gap", "median"),
                                           median_gap_adj=("gap_adj", "median"),
                                           pos_frac_adj=("gap_adj", lambda x: (x > 0).mean())).reindex(
        pd.MultiIndex.from_product([SPLITS, ["long", "short"]], names=["split", "side"])).round(3).to_string())
    print("\n   by screen (gap_adj median):")
    print(j.pivot_table(index="screen", columns="split", values="gap_adj", aggfunc="median").reindex(columns=SPLITS).round(0).to_string())
    print("\n   by hold (gap_adj median):")
    print(j.pivot_table(index="hold_h", columns="split", values="gap_adj", aggfunc="median").reindex(columns=SPLITS).round(0).to_string())
    ok = all(g.loc[s, "median_gap_adj"] > 0 and g.loc[s, "pos_frac_adj"] > 0.55 for s in SPLITS if s in g.index)
    print(f"\n   H1 verdict: {'HOLDS' if ok else 'FAILS'} (needs median gap_adj > 0 and positive fraction > 0.55 on all three splits)")
    j.to_parquet(os.path.join(REPORTS, f"vr_{a.label}_{a.tf}_h1.parquet"), index=False)
    if a.h1_only:
        return

    # ---- H2
    p = d.pivot_table(index=KEY, columns=["split", "regime"],
                      values=["n_trades", "mean_net5_bps", "mean_net10_bps", "median_net5_bps", "t_cluster", "quarters_pos", "win"])
    def col(v, sp, reg):
        return p[(v, sp, reg)] if (v, sp, reg) in p.columns else pd.Series(np.nan, index=p.index)
    s1 = ((col("n_trades", "train", "HIGH") >= 200) & (col("mean_net5_bps", "train", "HIGH") >= 30)
          & (col("mean_net10_bps", "train", "HIGH") >= 0) & (col("t_cluster", "train", "HIGH") >= 3)
          & (col("quarters_pos", "train", "HIGH") >= 6))
    lo_pass = ((col("n_trades", "train", "LOW") >= 200) & (col("mean_net5_bps", "train", "LOW") >= 30)
               & (col("t_cluster", "train", "LOW") >= 3) & (col("quarters_pos", "train", "LOW") >= 6))
    st1 = p[s1 & ~lo_pass.fillna(False)]
    print(f"\n=== H2 stage 1 (HIGH bucket on train, LOW bucket must NOT also pass): {len(st1)} cells "
          f"({int((s1 & lo_pass.fillna(False)).sum())} excluded for passing in LOW too)")
    if not len(st1):
        return
    idx = st1.index
    st2 = st1[(col("n_trades", "valid", "HIGH").loc[idx] >= 40) & (col("mean_net5_bps", "valid", "HIGH").loc[idx] > 0)
              & (col("t_cluster", "valid", "HIGH").loc[idx] >= 1.5)]
    print(f"=== H2 stage 2 (valid, HIGH): {len(st2)} of {len(st1)}")
    if not len(st2):
        show = st1.head(25)
    else:
        i2 = st2.index
        st3 = st2[(col("mean_net5_bps", "holdout", "HIGH").loc[i2] > 0) & (col("t_cluster", "holdout", "HIGH").loc[i2] >= 1.0)]
        print(f"=== H2 stage 3 (holdout, HIGH): {len(st3)} of {len(st2)}")
        show = (st3 if len(st3) else st2).head(25)
    out = pd.DataFrame({
        "n_tr": col("n_trades", "train", "HIGH").loc[show.index], "tr_hi": col("mean_net5_bps", "train", "HIGH").loc[show.index],
        "tr_lo": col("mean_net5_bps", "train", "LOW").loc[show.index], "tr_t": col("t_cluster", "train", "HIGH").loc[show.index],
        "va_hi": col("mean_net5_bps", "valid", "HIGH").loc[show.index], "va_t": col("t_cluster", "valid", "HIGH").loc[show.index],
        "ho_hi": col("mean_net5_bps", "holdout", "HIGH").loc[show.index], "ho_t": col("t_cluster", "holdout", "HIGH").loc[show.index],
        "med_tr": col("median_net5_bps", "train", "HIGH").loc[show.index], "win_tr": col("win", "train", "HIGH").loc[show.index],
    })
    print(out.round(1).to_string())


if __name__ == "__main__":
    main()
