"""Run a factor specification list through the IC evaluator in parallel.

Two things here are not negotiable. Every factor is computed on the **full**
history and only then sliced to the split being evaluated, so a rolling window
is never crippled by a truncated warm-up; and the slice ends at the split
boundary, so the forward return of the last bars runs off the end as NaN
instead of reaching into the next split. That is what keeps the holdout
genuinely untouched while still letting the train split use complete windows.

Worker count comes from config.workers_for(tf, "sweep"), a table measured on
this stage rather than inferred from the operator-call one.
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
import ic as IC  # noqa: E402
import factors as F  # noqa: E402
from config import workers_for  # noqa: E402
from panel import CACHE, Panel  # noqa: E402

REPORTS = os.path.join(HERE, "..", "reports")
HORIZON_HOURS = (4, 24, 72)

_CTX = {}


def _init(tf, split, horizons, rho, n_draws):
    p = Panel(tf)
    idx = p.index
    lo, hi = split
    sl = slice(int(np.searchsorted(idx, lo)), int(np.searchsorted(idx, hi)))
    mask = np.load(os.path.join(CACHE, tf, "universe_mask.npy"), mmap_mode="r")
    _CTX.update(tf=tf, sl=sl, rho=rho, horizons=horizons, n_draws=n_draws,
                close=np.asarray(p["close"][sl], dtype="float64"),
                mask=np.asarray(mask[sl]))


def _run(spec):
    tf, sl = _CTX["tf"], _CTX["sl"]
    try:
        f = spec.compute(tf, lambda n: np.asarray(B.load(tf, n), dtype="float64"))
        f = f[sl]
        rows = IC.evaluate(f, _CTX["close"], _CTX["mask"],
                           horizons=_CTX["horizons"], lag_bars=1,
                           rho=_CTX["rho"], name=spec.name(),
                           n_draws=_CTX["n_draws"])
        ac = IC.turnover_proxy(IC.apply_mask(f, _CTX["mask"]), 1)
        for r in rows:
            r["tag"] = spec.tag
            r["window_bars"] = spec.bars(tf)
            r["autocorr1"] = ac
        return rows
    except Exception as exc:
        return [{"name": spec.name(), "tag": spec.tag, "error": repr(exc)}]


def sweep(tf, specs, split_name="train", workers=None,
          horizon_hours=HORIZON_HOURS, label="round1", n_draws=100):
    splits = json.load(open(os.path.join(CACHE, "splits.json")))
    lo, hi = pd.Timestamp(splits[split_name][0]), pd.Timestamp(splits[split_name][1])
    per_hour = B.BARS_PER_DAY[tf] / 24.0
    horizons = tuple(sorted({max(1, int(round(h * per_hour))) for h in horizon_hours}))

    available = B.names(tf)
    keep, dropped = F.for_timeframe(specs, tf, available)
    print(f"{tf} [{split_name} {lo.date()}..{hi.date()}]  "
          f"{len(keep)} specs, {len(dropped)} dropped, horizons {horizons} bars",
          flush=True)
    for n, why in dropped:
        print(f"  drop {n:38s} {why}", flush=True)
    if not keep:
        return pd.DataFrame()

    # One correlation estimate for the whole sweep: it describes the market,
    # not the factor, and recomputing it per task would dominate the runtime.
    p = Panel(tf)
    sl = slice(int(np.searchsorted(p.index, lo)), int(np.searchsorted(p.index, hi)))
    close = np.asarray(p["close"][sl], dtype="float64")
    mask = np.asarray(np.load(os.path.join(CACHE, tf, "universe_mask.npy"),
                              mmap_mode="r")[sl])
    rho, n_used = IC.average_pair_correlation(IC.apply_mask(IC.forward_return(close, 1), mask))
    print(f"  average pairwise return correlation {rho:.3f} over {n_used} symbols "
          f"-> {IC.effective_count(n_used, rho):.1f} effective", flush=True)
    del close, mask, p

    n_workers = workers or workers_for(tf, "sweep")
    print(f"  {n_workers} workers", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init,
                             initargs=(tf, (lo, hi), horizons, rho, n_draws)) as ex:
        for i, rows in enumerate(ex.map(_run, keep), 1):
            out.extend(rows)
            if i % 10 == 0:
                print(f"  {i}/{len(keep)}", flush=True)

    df = pd.DataFrame(out)
    if "error" in df.columns:
        bad = df[df["error"].notna()]
        for _, r in bad.iterrows():
            sys.stderr.write(f"FAIL {r['name']}: {r['error']}\n")
        df = df[df["error"].isna()].drop(columns=["error"])
    if df.empty:
        return df

    # Multiple testing on the time-series side. The empirical shift-null
    # p-value bottoms out at 1/n_draws, which is coarser than a sweep of a few
    # thousand specs needs, so the q-value is built from a normal
    # approximation to the same z and Benjamini-Hochberg across the sweep.
    from scipy.stats import norm
    df["ic_ts_q"] = np.nan
    for h, g in df.groupby("horizon"):
        z = g["ic_ts_z"].to_numpy(dtype="float64")
        ok = np.isfinite(z)
        if ok.sum() < 2:
            continue
        pv = 2.0 * norm.sf(np.abs(z[ok]))
        order = np.argsort(pv)
        m = pv.size
        q = np.empty(m)
        q[order] = np.minimum.accumulate(
            (pv[order] * m / np.arange(1, m + 1))[::-1])[::-1]
        df.loc[g.index[ok], "ic_ts_q"] = np.minimum(q, 1.0)

    # Deflate against the whole sweep, not against each factor alone: the
    # winner's t-stat has to beat the best of N draws, and N is this sweep.
    df["dsr"] = np.nan
    for h, g in df.groupby("horizon"):
        trials = g["icir"].to_numpy()
        for i, row in g.iterrows():
            if np.isfinite(row["icir"]) and np.isfinite(row["n_bars"]):
                d, _ = IC.deflated_sharpe(abs(row["icir"]), row["n_bars"],
                                          sr_trials=np.abs(trials))
                df.loc[i, "dsr"] = d

    df = df.sort_values(["horizon", "ic_cs_t"], key=lambda s: s.abs()
                        if s.name == "ic_cs_t" else s, ascending=[True, False])
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, f"sweep_{label}_{tf}_{split_name}.csv")
    df.to_csv(path, index=False)
    print(f"  wrote {path}", flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tf", nargs="?", default="1h")
    ap.add_argument("--round", default="round1", choices=("round1", "round2", "round3", "round4"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--draws", type=int, default=100,
                    help="circular-shift null draws for the time-series test")
    ap.add_argument("--specs-from", default=None,
                    help="file of factor names, one per line: sweep only these. "
                         "Used to score an existing shortlist on another split "
                         "without re-running the whole round.")
    a = ap.parse_args()
    # The holdout is evaluated exactly once, at the very end of phase 5, and
    # never from here -- reaching it needs falsify.py's explicit flag.
    if a.split == "holdout":
        sys.exit("refusing: run falsify.py --final for the holdout")
    specs = {"round1": lambda: F.ROUND1, "round2": F.ROUND2,
              "round3": F.ROUND3, "round4": F.ROUND4}[a.round]()
    if a.specs_from:
        want = {ln.strip() for ln in open(a.specs_from) if ln.strip()}
        by_name = {sp.name(): sp for sp in specs}
        missing = want - set(by_name)
        specs = [by_name[n] for n in sorted(want & set(by_name))]
        print(f"restricted to {len(specs)} named specs" +
              (f", {len(missing)} not found in {a.round}" if missing else ""), flush=True)
        for n in sorted(missing):
            print(f"  missing {n}", flush=True)
    df = sweep(a.tf, specs, a.split, a.workers, label=a.round, n_draws=a.draws)
    if not df.empty:
        cols = ["name", "tag", "horizon", "ic_cs", "ic_cs_t", "icir",
                "ic_ts", "ic_ts_z", "ic_ts_q", "autocorr1", "dsr",
                "passes_cs", "passes_ts"]
        for h, g in df.groupby("horizon"):
            print(f"\n--- horizon {h} bars, top 12 by |t| ---")
            print(g[cols].head(12).to_string(index=False,
                                             float_format=lambda v: f"{v:.4f}"))
