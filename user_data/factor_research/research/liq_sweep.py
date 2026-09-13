"""Re-run the cost screen tier by liquidity tier, with funding included.

Two corrections at once, because they are entangled.

The first is funding. `cost.py` computed P&L from price returns only, which is
right for a price factor and first-order wrong for the funding factors the
screen kept selecting: their book sits deliberately on the extremes of the
funding distribution. See `cost.funding_carry`.

The second is liquidity. Every survivor so far was ranked across the whole
tradeable universe, roughly 350 names deep, and the top of the resulting book
was micro-caps. Restricting the cross-section to the most liquid N and asking
whether the edge is still there is cheaper than measuring true slippage and
answers the prior question: if the edge only exists where 5 bps is a fantasy,
no impact model rescues it.

Direction is fixed once, on the train split of the full universe, and carried
into every tier and into valid unchanged. Re-fitting the sign per tier would
give each tier a free parameter and manufacture agreement.
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
import tiers as T  # noqa: E402
from config import workers_for  # noqa: E402
from panel import CACHE, Panel  # noqa: E402

REPORTS = os.path.join(HERE, "..", "reports")
_CTX = {}


def _init(tf, split_name, horizon_hours, only_tiers=None):
    splits = json.load(open(os.path.join(CACHE, "splits.json")))
    lo, hi = (pd.Timestamp(x) for x in splits[split_name])
    p = Panel(tf)
    sl = slice(int(np.searchsorted(p.index, lo)), int(np.searchsorted(p.index, hi)))
    mask = np.asarray(np.load(os.path.join(CACHE, tf, "universe_mask.npy"),
                              mmap_mode="r")[sl])
    rank = np.asarray(T.load(tf)[sl])
    fields = set(p.meta.get("fields", []))
    close = np.asarray(p["close"][sl], dtype="float64")
    h = max(1, int(round(horizon_hours * B.BARS_PER_DAY[tf] / 24.0)))
    r1 = IC.forward_return(close, 1)
    tmask = T.tier_masks(tf, mask, rank)
    # The sign mask stays the full universe even when `all` is not reported,
    # so restricting which tiers are printed can never change which direction
    # a factor is traded in.
    sign_mask = tmask["all"]
    if only_tiers:
        tmask = {k: v for k, v in tmask.items() if k in only_tiers}
    _CTX.update(tf=tf, sl=sl, tiers=tmask, sign_mask=sign_mask, close=close,
                ret1=np.where(np.isfinite(r1), r1, 0.0),
                fwd_h=IC.forward_return(close, h),
                funding=(np.asarray(p["funding_rate"][sl], dtype="float64")
                         if "funding_rate" in fields else None),
                interval=(np.asarray(p["funding_interval_hours"][sl], dtype="float64")
                          if "funding_interval_hours" in fields else None),
                h=h)


def _run(args):
    spec, fixed_sign = args
    tf, sl, close, h = _CTX["tf"], _CTX["sl"], _CTX["close"], _CTX["h"]
    name = spec.name()
    try:
        f = spec.compute(tf, lambda n: np.asarray(B.load(tf, n), dtype="float64"))[sl]
    except Exception as exc:
        return [{"name": name, "tf": tf, "tier": "all", "error": repr(exc)}]

    # Lagging the factor is tier-independent; ranking is not, but the rank a
    # tier needs is used twice -- once for the decile profile, once for the
    # book -- and computing it once here is what makes five tiers affordable.
    f_lag = IC.lag(f, 1)
    sign, cached = fixed_sign, {}
    if sign is None:
        sm = _CTX["sign_mask"]
        rk0 = IC._rank_axis1(IC.apply_mask(f_lag, sm))
        if "all" in _CTX["tiers"]:
            cached["all"] = rk0          # same mask; do not rank it twice
        qp0 = C.quantile_profile(f, close, sm, horizon=h, lag_bars=1,
                                 rank=rk0, fwd=_CTX["fwd_h"])
        sign = 1.0 if qp0["mean_spread_bps"] >= 0 else -1.0
    rows = []
    for tier, m in _CTX["tiers"].items():
        try:
            rk = cached.get(tier)
            if rk is None:
                rk = IC._rank_axis1(IC.apply_mask(f_lag, m))
            qp = C.quantile_profile(f, close, m, horizon=h, lag_bars=1,
                                    rank=rk, fwd=_CTX["fwd_h"])
            c = C.evaluate(f, close, m, lag_bars=1, hold=h, sign=sign,
                           bars_per_year=C.BARS_PER_YEAR[tf],
                           funding_rate=_CTX["funding"],
                           interval_hours=_CTX["interval"],
                           rank=rk, ret1=_CTX["ret1"])
            rows.append({
                "name": name, "tag": spec.tag, "tf": tf, "tier": tier,
                "hold_bars": h, "sign": sign,
                "mean_spread_bps": qp["mean_spread_bps"],
                "median_spread_bps": qp["median_spread_bps"],
                "tail_consistent": qp["tail_consistent"],
                "turnover": c.get("turnover_per_bar"),
                "gross_bps": c.get("gross_bps_per_bar"),
                "price_bps": c.get("price_bps_per_bar"),
                "funding_bps": c.get("funding_bps_per_bar"),
                "gross_sharpe": c.get("gross_sharpe"),
                "pnl_t_nw": c.get("pnl_t_nw"),
                "eff_obs": c.get("eff_obs"),
                "breakeven_bps": c.get("breakeven_cost_bps"),
                "net_sharpe_5bps": c.get("net_sharpe_5bps"),
                "net_sharpe_10bps": c.get("net_sharpe_10bps"),
            })
        except Exception as exc:
            rows.append({"name": name, "tf": tf, "tier": tier, "error": repr(exc)})
    return rows


def sweep(tf, specs, split_name, horizon_hours=24.0, signs=None, workers=None,
          only_tiers=None):
    keep, dropped = F.for_timeframe(specs, tf, B.names(tf))
    n = workers or workers_for(tf, "sweep")
    n_tier = len(only_tiers) if only_tiers else len(T.CUMULATIVE) + 2
    print(f"{tf} {split_name}: {len(keep)} specs x {n_tier} tiers, "
          f"{len(dropped)} dropped, {n} workers", flush=True)
    jobs = [(s, (signs or {}).get(s.name())) for s in keep]
    rows = []
    with ProcessPoolExecutor(max_workers=n, initializer=_init,
                             initargs=(tf, split_name, horizon_hours,
                                       only_tiers)) as ex:
        for i, r in enumerate(ex.map(_run, jobs), 1):
            rows.extend(r)
            if i % 25 == 0:
                print(f"  {i}/{len(keep)}", flush=True)
    df = pd.DataFrame(rows)
    df["split"] = split_name
    if "error" in df.columns:
        for _, r in df[df["error"].notna()].iterrows():
            sys.stderr.write(f"FAIL {r['name']} [{r['tier']}]: {r['error']}\n")
        df = df[df["error"].isna()].drop(columns=["error"])
    return df


# Above the 4.5 bps taker fee, not above zero: a factor that breaks even at
# 2 bps is already dead at the venue's own price list, before any slippage.
MIN_BREAKEVEN_BPS = 5.0
MIN_T = 2.0


def verdict(df):
    """Per-tier pass/fail under the same rule the full-universe screen used."""
    tr = df[df.split == "train"].set_index(["name", "tier"])
    va = df[df.split == "valid"].set_index(["name", "tier"])
    cols = ["breakeven_bps", "pnl_t_nw", "tail_consistent", "turnover",
            "gross_bps", "price_bps", "funding_bps", "net_sharpe_5bps"]
    j = tr[["tag", "sign"] + cols].join(va[cols], lsuffix="_tr", rsuffix="_va",
                                        how="inner")
    j["holds"] = ((j.breakeven_bps_tr > MIN_BREAKEVEN_BPS)
                  & (j.breakeven_bps_va > MIN_BREAKEVEN_BPS)
                  & j.tail_consistent_tr.astype(bool)
                  & j.tail_consistent_va.astype(bool)
                  & (j.pnl_t_nw_tr > MIN_T) & (j.pnl_t_nw_va > MIN_T))
    return j


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tf", nargs="?", default="1h")
    ap.add_argument("--round", default="round3",
                    choices=("round1", "round2", "round3", "round4"))
    ap.add_argument("--horizon-hours", type=float, default=24.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--tiers", default=None,
                    help="comma-separated subset, e.g. all,top100")
    ap.add_argument("--only", default=None,
                    help="file of factor names, one per line: stage-2 shortlist")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    specs = {"round1": lambda: F.ROUND1, "round2": F.ROUND2,
             "round3": F.ROUND3, "round4": F.ROUND4}[a.round]()
    if a.only:
        want = {ln.strip() for ln in open(a.only) if ln.strip()}
        specs = [s for s in specs if s.name() in want]
        missing = want - {s.name() for s in specs}
        if missing:
            sys.stderr.write(f"not in {a.round}: {sorted(missing)[:5]} "
                             f"({len(missing)} total)\n")
        print(f"--only: {len(specs)} of {len(want)} names resolved", flush=True)
    only_tiers = a.tiers.split(",") if a.tiers else None
    stem = os.path.join(REPORTS, f"liq_{a.round}_{a.tf}{a.suffix}")

    tr = sweep(a.tf, specs, "train", a.horizon_hours, workers=a.workers,
               only_tiers=only_tiers)
    tr.to_csv(f"{stem}_train.csv", index=False)     # checkpoint: valid is another
    print(f"wrote {stem}_train.csv", flush=True)    # 15 minutes and jobs do die
    # Signs are identical across tiers by construction; take them from
    # whichever tier is present so --tiers cannot leave this empty.
    first = tr[tr.tier == tr.tier.iloc[0]]
    signs = dict(zip(first["name"], first["sign"]))
    va = sweep(a.tf, specs, "valid", a.horizon_hours, signs=signs,
               workers=a.workers, only_tiers=only_tiers)
    df = pd.concat([tr, va], ignore_index=True)
    df.to_csv(f"{stem}.csv", index=False)
    print(f"wrote {stem}.csv\n", flush=True)

    j = verdict(df)
    print(f"=== {a.tf} {a.round}: break-even > {MIN_BREAKEVEN_BPS} bps and "
          f"P&L t > {MIN_T} in BOTH splits, by liquidity tier ===")
    order = [t for t in ["all"] + [f"top{n}" for n in T.CUMULATIVE] + ["tail"]
             if not only_tiers or t in only_tiers]
    counts = j.groupby("tier").holds.agg(["sum", "count"]).reindex(order)
    print(counts.to_string())
    for tier in order:
        sub = j.xs(tier, level="tier")
        sub = sub[sub.holds].sort_values("pnl_t_nw_va", ascending=False)
        print(f"\n--- {tier}: {len(sub)} hold ---")
        if len(sub):
            print(sub[["sign", "turnover_tr", "breakeven_bps_tr",
                       "breakeven_bps_va", "pnl_t_nw_tr", "pnl_t_nw_va",
                       "funding_bps_tr", "net_sharpe_5bps_va"]]
                  .head(a.top).to_string(float_format=lambda v: f"{v:.3f}"))
