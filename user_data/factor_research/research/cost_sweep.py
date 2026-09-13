"""Rank a whole round by break-even cost instead of by rank IC.

Everything upstream selects on rank IC, and rank IC turned out to be a poor
proxy for money on this data: the strongest factor by z-score,
decay_linear(ret1,336h) at z = -7.2, has a decile mean spread that contradicts
its median spread, and several factors that never passed the IC gate at all --
decay_linear(vwap_dev,336h) at z = -1.1 -- carry the widest cost cushion in the
whole shortlist. Selecting on the IC survivors therefore searches a set that
was filtered by the wrong criterion.

This sweeps every spec in a round directly on what a decision needs: the
one-way cost at which the book stops making money, plus the decile profile that
says whether that number rests on the whole distribution or on a handful of
outliers. Direction is fixed on train and carried into valid unchanged, so
valid tests a decision rather than re-making it.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import base as B  # noqa: E402
import cost as C  # noqa: E402
import factors as F  # noqa: E402
import ic as IC  # noqa: E402
from config import workers_for  # noqa: E402
from panel import CACHE, Panel  # noqa: E402

REPORTS = os.path.join(HERE, "..", "reports")
_CTX = {}


def _init(tf, split_name, horizon_hours):
    splits = json.load(open(os.path.join(CACHE, "splits.json")))
    lo, hi = (pd.Timestamp(x) for x in splits[split_name])
    p = Panel(tf)
    sl = slice(int(np.searchsorted(p.index, lo)), int(np.searchsorted(p.index, hi)))
    mask = np.asarray(np.load(os.path.join(CACHE, tf, "universe_mask.npy"),
                              mmap_mode="r")[sl])
    fields = set(p.meta.get("fields", []))
    _CTX.update(tf=tf, sl=sl, mask=mask,
                close=np.asarray(p["close"][sl], dtype="float64"),
                funding=(np.asarray(p["funding_rate"][sl], dtype="float64")
                         if "funding_rate" in fields else None),
                interval=(np.asarray(p["funding_interval_hours"][sl], dtype="float64")
                          if "funding_interval_hours" in fields else None),
                h=max(1, int(round(horizon_hours * B.BARS_PER_DAY[tf] / 24.0))))


def _run(args):
    spec, fixed_sign = args
    tf, sl, close, mask, h = (_CTX["tf"], _CTX["sl"], _CTX["close"],
                              _CTX["mask"], _CTX["h"])
    name = spec.name()
    try:
        f = spec.compute(tf, lambda n: np.asarray(B.load(tf, n), dtype="float64"))[sl]
        qp = C.quantile_profile(f, close, mask, horizon=h, lag_bars=1)
        sign = (fixed_sign if fixed_sign is not None
                else (1.0 if qp["mean_spread_bps"] >= 0 else -1.0))
        c = C.evaluate(f, close, mask, lag_bars=1, hold=h, sign=sign,
                       bars_per_year=C.BARS_PER_YEAR[tf],
                       funding_rate=_CTX["funding"],
                       interval_hours=_CTX["interval"])
        return {
            "name": name, "tag": spec.tag, "tf": tf, "hold_bars": h,
            "sign": sign,
            "mean_spread_bps": qp["mean_spread_bps"],
            "median_spread_bps": qp["median_spread_bps"],
            "tail_consistent": qp["tail_consistent"],
            "turnover": c.get("turnover_per_bar"),
            "gross_bps": c.get("gross_bps_per_bar"),
            "price_bps": c.get("price_bps_per_bar"),
            "funding_bps": c.get("funding_bps_per_bar"),
            "gross_sharpe": c.get("gross_sharpe"),
            "pnl_t_nw": c.get("pnl_t_nw"),
            "pnl_autocorr1": c.get("pnl_autocorr1"),
            "eff_obs": c.get("eff_obs"),
            "breakeven_bps": c.get("breakeven_cost_bps"),
            "net_sharpe_2bps": c.get("net_sharpe_2bps"),
            "net_sharpe_5bps": c.get("net_sharpe_5bps"),
            "net_sharpe_10bps": c.get("net_sharpe_10bps"),
        }
    except Exception as exc:
        return {"name": name, "tf": tf, "error": repr(exc)}


def sweep(tf, specs, split_name, horizon_hours=24.0, signs=None, workers=None):
    keep, dropped = F.for_timeframe(specs, tf, B.names(tf))
    n = workers or workers_for(tf, "sweep")
    print(f"{tf} {split_name}: {len(keep)} specs, {len(dropped)} dropped, "
          f"{n} workers", flush=True)
    jobs = [(s, (signs or {}).get(s.name())) for s in keep]
    rows = []
    with ProcessPoolExecutor(max_workers=n, initializer=_init,
                             initargs=(tf, split_name, horizon_hours)) as ex:
        for i, r in enumerate(ex.map(_run, jobs), 1):
            rows.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(keep)}", flush=True)
    df = pd.DataFrame(rows)
    df["split"] = split_name
    if "error" in df.columns:
        bad = df[df["error"].notna()]
        for _, r in bad.iterrows():
            sys.stderr.write(f"FAIL {r['name']}: {r['error']}\n")
        df = df[df["error"].isna()].drop(columns=["error"])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tf", nargs="?", default="1h")
    ap.add_argument("--round", default="round2",
                    choices=("round1", "round2", "round3", "round4"))
    ap.add_argument("--horizon-hours", type=float, default=24.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    specs = {"round1": lambda: F.ROUND1, "round2": F.ROUND2,
              "round3": F.ROUND3, "round4": F.ROUND4}[a.round]()

    tr = sweep(a.tf, specs, "train", a.horizon_hours, workers=a.workers)
    signs = dict(zip(tr["name"], tr["sign"]))
    va = sweep(a.tf, specs, "valid", a.horizon_hours, signs=signs, workers=a.workers)

    df = pd.concat([tr, va], ignore_index=True)
    out = os.path.join(REPORTS, f"cost_{a.round}_{a.tf}.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}\n")

    j = (tr.set_index("name")[["tag", "sign", "turnover", "mean_spread_bps",
                               "median_spread_bps", "tail_consistent",
                               "breakeven_bps", "net_sharpe_5bps", "pnl_t_nw"]]
         .join(va.set_index("name")[["breakeven_bps", "net_sharpe_5bps",
                                     "tail_consistent", "pnl_t_nw"]],
               lsuffix="_tr", rsuffix="_va"))
    # A candidate has to clear cost in both splits and rest on the whole
    # distribution in both, not just where it was selected.
    # Break-even says how much cost the edge absorbs; the t-stat says whether
    # there is an edge to absorb it. Both are required -- several factors with
    # the widest cushion in the shortlist carry a P&L t under 1.
    j["holds"] = (j.breakeven_bps_tr > 0) & (j.breakeven_bps_va > 0) & \
                 j.tail_consistent_tr & j.tail_consistent_va & \
                 (j.pnl_t_nw_tr > 2.0) & (j.pnl_t_nw_va > 2.0)
    j = j.sort_values(["holds", "pnl_t_nw_va"], ascending=[False, False])
    print(f"--- {a.tf} {a.round}: break-even > 0, tail-consistent and P&L "
          f"t > 2 in BOTH splits: {int(j.holds.sum())} of {len(j)} ---")
    cols = ["sign", "turnover", "breakeven_bps_tr", "breakeven_bps_va",
            "pnl_t_nw_tr", "pnl_t_nw_va", "net_sharpe_5bps_va", "holds"]
    print(j[cols].head(a.top).to_string(float_format=lambda v: f"{v:.3f}"))
