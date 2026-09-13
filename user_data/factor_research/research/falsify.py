"""Phase 5: try to kill the factors that survived the sweep.

Four independent attacks, in the order that costs least first:

* **Decorrelation.** Two factors correlated above 0.7 are one factor with two
  names; keeping both doubles the apparent evidence for a single effect.
* **Purged K-fold.** Plain contiguous folds leak, because an h-bar forward
  return straddles the boundary. Each fold therefore drops h bars of purge at
  both edges plus an embargo, and a factor has to hold its IC sign in most
  folds, not just on average.
* **Sign stability by quarter.** An effect that was strong for one regime and
  absent afterwards averages out to a good t-stat and is worthless forward.
* **Holdout.** Evaluated exactly once, only via --final, and the result file
  refuses to be overwritten so a second look leaves a trace.

Nothing here selects factors -- selection happened in the sweep. Everything
here only removes them.
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
import factors as F  # noqa: E402
import ic as IC  # noqa: E402
from panel import CACHE, Panel  # noqa: E402
from config import workers_for  # noqa: E402

REPORTS = os.path.join(HERE, "..", "reports")
# 0.7 is the conventional equity-factor threshold, and it is too loose for this
# pool. Measured on the round-2 1h survivors (2026-08-23): 19 factors stayed
# distinct at 0.7, 10 at 0.5, 5 at 0.3. The equity convention assumes a far more
# heterogeneous factor pool than a cartesian product over one operator library,
# where neighbouring windows of the same operator are near-duplicates by
# construction.
CORR_LIMIT = 0.5
# The correlation matrix is subsampled to a fixed number of points, not a fixed
# stride. A stride of 4 costs 1.8M points at 1h but 7.3M at 15m, and the O(n^2)
# comparison then runs an order of magnitude slower for an estimate that is no
# better: deciding whether |r| clears 0.5 does not need 7 million observations.
# Measured 2026-08-23, the fixed stride left a 212-spec decorrelation at 15m
# still running after 45 minutes.
CORR_SAMPLE_POINTS = 2_000_000


def spec_index():
    """Every specification any round can produce, keyed by its printed name."""
    out = {}
    for specs in (F.ROUND1, F.ROUND2(), F.ROUND3(), F.ROUND4()):
        for s in specs:
            out.setdefault(s.name(), s)
    return out


def _slice(tf, split_name):
    splits = json.load(open(os.path.join(CACHE, "splits.json")))
    lo, hi = (pd.Timestamp(x) for x in splits[split_name])
    p = Panel(tf)
    sl = slice(int(np.searchsorted(p.index, lo)), int(np.searchsorted(p.index, hi)))
    mask = np.asarray(np.load(os.path.join(CACHE, tf, "universe_mask.npy"),
                              mmap_mode="r")[sl])
    close = np.asarray(p["close"][sl], dtype="float64")
    return sl, close, mask, (lo, hi)


def _factor(tf, spec, sl, mask):
    f = spec.compute(tf, lambda n: np.asarray(B.load(tf, n), dtype="float64"))
    return IC.apply_mask(IC.lag(f, 1)[sl], mask)


# Both expensive stages here recompute a factor per candidate and are
# embarrassingly parallel across candidates, but ran single-process until
# 2026-08-23: 20 factors at 15m took an hour on one core while 63 sat idle.
# The greedy selection itself stays serial -- it has to, each decision depends
# on everything kept so far -- so only the vector computation is farmed out.
_CTX = {}


def _worker_init(tf, split_name):
    sl, close, mask, _ = _slice(tf, split_name)
    _CTX.update(tf=tf, split_name=split_name, sl=sl, close=close, mask=mask,
                specs=spec_index())


def _worker_vector(args):
    name, stride = args
    s = _CTX["specs"].get(name)
    if s is None:
        return name, None
    v = IC._rank_axis1(_factor(_CTX["tf"], s, _CTX["sl"], _CTX["mask"]))[::stride]
    return name, v.ravel()


def _worker_falsify(args):
    name, horizon, min_fold, min_quarter = args
    s = _CTX["specs"][name]
    tf, split_name = _CTX["tf"], _CTX["split_name"]
    folds, fold_agree = purged_kfold(tf, s, horizon, split_name=split_name)
    quarters, q_agree = sign_stability(tf, s, horizon, split_name=split_name)
    return {
        "name": name,
        "fold_agreement": fold_agree, "n_folds": len(folds),
        "quarter_agreement": q_agree, "n_quarters": len(quarters),
        "worst_fold_ic": float(folds["ic"].min()) if len(folds) else np.nan,
        "best_fold_ic": float(folds["ic"].max()) if len(folds) else np.nan,
        "survives": bool(fold_agree >= min_fold and q_agree >= min_quarter),
    }


def _pool(tf, split_name):
    return ProcessPoolExecutor(max_workers=workers_for(tf, "sweep"),
                               initializer=_worker_init,
                               initargs=(tf, split_name))


def decorrelate(tf, ranked, split_name="train", limit=CORR_LIMIT):
    """Greedy: walk the ranking, drop anything too close to something kept."""
    sl, _, mask, _ = _slice(tf, split_name)
    n_rows, n_sym = mask.shape
    stride = max(1, int(np.ceil(n_rows * n_sym / CORR_SAMPLE_POINTS)))
    print(f"correlation sample: every {stride} bars "
          f"({n_rows // stride * n_sym / 1e6:.1f}M points per factor)")
    # Cross-sectional ranks, so the comparison matches how IC is measured and a
    # monotone rescaling of a factor counts as the same factor.
    order = [n for n, _ in ranked]
    with _pool(tf, split_name) as ex:
        computed = dict(ex.map(_worker_vector, [(n, stride) for n in order]))

    kept, dropped, vectors = [], [], []
    for name, score in ranked:
        v = computed.get(name)
        if v is None:
            dropped.append((name, "spec not found", np.nan))
            continue
        worst_name, worst = None, 0.0
        for kname, kv in zip(kept, vectors):
            ok = np.isfinite(v) & np.isfinite(kv)
            if ok.sum() < 1000:
                continue
            c = abs(float(np.corrcoef(v[ok], kv[ok])[0, 1]))
            if c > worst:
                worst_name, worst = kname, c
        if worst >= limit:
            dropped.append((name, f"|corr| {worst:.2f} with {worst_name}", worst))
        else:
            kept.append(name)
            vectors.append(v)
    return kept, dropped


def purged_kfold(tf, spec, horizon, k=5, embargo=0.01, split_name="train"):
    """IC inside each fold, with purge and embargo around the boundaries."""
    sl, close, mask, _ = _slice(tf, split_name)
    f = _factor(tf, spec, sl, mask)
    r = IC.apply_mask(IC.forward_return(close, horizon), mask)
    t = f.shape[0]
    edge = horizon + int(embargo * t)
    fold = t // k
    rows = []
    for i in range(k):
        a, b = i * fold, (i + 1) * fold if i < k - 1 else t
        a, b = a + (edge if i else 0), b - (edge if i < k - 1 else 0)
        if b - a < max(50, 2 * horizon):
            continue
        ics = IC.rank_ic_cross_section(f[a:b], r[a:b])
        mu, se, tt, n = IC.newey_west(ics, band=max(horizon - 1, 0))
        rows.append({"fold": i, "bars": b - a, "ic": mu, "t": tt, "n": n})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, np.nan
    same = float(np.mean(np.sign(df["ic"]) == np.sign(df["ic"].mean())))
    return df, same


def sign_stability(tf, spec, horizon, split_name="train", freq="QE"):
    """IC recomputed per calendar quarter -- regime dependence shows up here."""
    sl, close, mask, _ = _slice(tf, split_name)
    p = Panel(tf)
    idx = p.index[sl]
    f = _factor(tf, spec, sl, mask)
    r = IC.apply_mask(IC.forward_return(close, horizon), mask)
    ics = pd.Series(IC.rank_ic_cross_section(f, r), index=idx)
    g = ics.groupby(pd.Grouper(freq=freq))
    out = pd.DataFrame({"ic": g.mean(), "bars": g.count()})
    out = out[out["bars"] > max(20, horizon)]
    if out.empty:
        return out, np.nan
    return out, float(np.mean(np.sign(out["ic"]) == np.sign(out["ic"].mean())))


def report(tf, label="round1", split_name="train", horizon=None, top=15,
           min_fold_agreement=0.8, min_quarter_agreement=0.75, gate="ts",
           max_decorrelate=200, corr_limit=CORR_LIMIT, only=None):
    path = os.path.join(REPORTS, f"sweep_{label}_{tf}_{split_name}.csv")
    df = pd.read_csv(path)
    if horizon is None:
        horizon = int(df["horizon"].median())
    # Rank by the gate being applied, so the shortlist that reaches the
    # expensive falsification is the one the gate actually endorses.
    col = {"ts": "passes_ts", "cs": "passes_cs", "any": "passes"}[gate]
    rank_on = "ic_ts_z" if gate == "ts" else "ic_cs_t"
    g = df[(df["horizon"] == horizon) & df[col]].copy()
    g["abs_t"] = g[rank_on].abs()
    g = g.sort_values("abs_t", ascending=False)
    print(f"{tf} {label} horizon {horizon}: {len(g)} factors passed the "
          f"{gate} gate (|{rank_on}| > {IC.T_GATE})")
    if g.empty:
        return pd.DataFrame()

    if only:
        # Confirmation across timeframes has to test the *same* factors at each
        # one. Letting every timeframe pick its own top-N confounds "failed the
        # test" with "never entered the shortlist": at 4h, ts_rank(amihud,336h)
        # shows a blank not because it was rejected but because 4h's own
        # ranking admitted the 168h window instead. With --only the candidate
        # set is fixed by the caller and the gate and decorrelation are skipped.
        want = [n for n in only if n in set(g["name"])]
        missing = [n for n in only if n not in set(g["name"])]
        g = g[g["name"].isin(want)]
        kept = list(g.sort_values("abs_t", ascending=False)["name"])
        print(f"explicit candidate list: {len(kept)} of {len(only)} present "
              f"and past the {gate} gate")
        for n in missing:
            row = df[(df["horizon"] == horizon) & (df["name"] == n)]
            why = (f"below gate ({rank_on}="
                   f"{row.iloc[0][rank_on]:.2f})" if len(row) else "not in this sweep")
            print(f"  absent {n:38s} {why}")
        if not kept:
            return pd.DataFrame()
        dropped = []
        return _falsify_list(tf, kept, g, horizon, split_name, rank_on, gate,
                             min_fold_agreement, min_quarter_agreement,
                             corr_limit, label, explicit=True)

    # Decorrelate the whole passing set BEFORE truncating to `top`. Truncating
    # first hands the shortlist to whichever family happens to score highest:
    # in round 2 at 1h, 133 specs passed and the top 25 were all amihud under
    # different operators and windows, so every other family was cut before
    # anything looked at it. Collapsing duplicates first means `top` counts
    # distinct factors.
    head = g.head(max_decorrelate)
    ranked = list(zip(head["name"], head["abs_t"]))
    kept, dropped = decorrelate(tf, ranked, split_name, corr_limit)
    print(f"\ndecorrelation over {len(ranked)} passing specs at |r| > {corr_limit}: "
          f"{len(kept)} distinct, {len(dropped)} dropped")
    for n, why, _ in dropped:
        print(f"  drop {n:38s} {why}")
    if len(g) > max_decorrelate:
        print(f"  note: {len(g) - max_decorrelate} specs below rank "
              f"{max_decorrelate} were not examined")
    kept = kept[:top]
    print(f"taking the top {len(kept)} distinct factors into falsification")

    return _falsify_list(tf, kept, g, horizon, split_name, rank_on, gate,
                         min_fold_agreement, min_quarter_agreement,
                         corr_limit, label)


def _falsify_list(tf, kept, g, horizon, split_name, rank_on, gate,
                  min_fold_agreement, min_quarter_agreement, corr_limit, label,
                  explicit=False):
    """Purged K-fold and quarterly sign stability for an explicit candidate list."""
    jobs = [(n, horizon, min_fold_agreement, min_quarter_agreement) for n in kept]
    with _pool(tf, split_name) as ex:
        verdicts = {r["name"]: r for r in ex.map(_worker_falsify, jobs)}

    rows = []
    for name in kept:
        base = g[g["name"] == name].iloc[0]
        v = verdicts[name]
        rows.append({
            "name": name, "tag": base["tag"], "ic_cs": base["ic_cs"],
            "ic_cs_t": base["ic_cs_t"], "ic_ts": base["ic_ts"],
            "ic_ts_z": base["ic_ts_z"], "ic_ts_q": base.get("ic_ts_q", np.nan),
            "icir": base["icir"],
            "autocorr1": base["autocorr1"], "dsr": base["dsr"],
            **{k: v[k] for k in ("fold_agreement", "n_folds", "quarter_agreement",
                                 "n_quarters", "worst_fold_ic", "best_fold_ic",
                                 "survives")},
        })
    out = pd.DataFrame(rows).sort_values(rank_on, key=abs, ascending=False)
    tag = "only" if explicit else f"r{corr_limit:g}"
    dest = os.path.join(
        REPORTS, f"falsify_{label}_{gate}_{tf}_{split_name}_h{horizon}_{tag}.csv")
    out.to_csv(dest, index=False)
    print(f"\n{int(out['survives'].sum())}/{len(out)} survive falsification")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {dest}")
    return out


def final(tf, names, horizon, label="round1"):
    """The one and only holdout evaluation. Refuses to run twice."""
    dest = os.path.join(REPORTS, f"HOLDOUT_{label}_{tf}_h{horizon}.csv")
    if os.path.exists(dest):
        sys.exit(f"refusing: {dest} already exists -- the holdout was already used")
    sl, close, mask, span = _slice(tf, "holdout")
    specs = spec_index()
    rows = []
    for name in names:
        s = specs[name]
        f = _factor(tf, s, sl, mask)
        r = IC.apply_mask(IC.forward_return(close, horizon), mask)
        ics = IC.rank_ic_cross_section(f, r)
        mu, se, t, n = IC.newey_west(ics, band=max(horizon - 1, 0))
        sd = float(np.nanstd(ics, ddof=1))
        rows.append({"name": name, "ic": mu, "se": se, "t": t,
                     "icir": mu / sd if sd else np.nan, "n_bars": n})
    out = pd.DataFrame(rows)
    out.attrs["span"] = [str(span[0]), str(span[1])]
    out.to_csv(dest, index=False)
    print(f"holdout {span[0].date()}..{span[1].date()}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {dest}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tf", nargs="?", default="1h")
    ap.add_argument("--round", default="round1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--top", type=int, default=15,
                    help="distinct factors to falsify, counted after decorrelation")
    ap.add_argument("--only", default=None,
                    help="comma-separated factor names, or a path to a file with "
                         "one per line; skips the gate and decorrelation so every "
                         "timeframe is asked about the same candidates")
    ap.add_argument("--corr-limit", type=float, default=CORR_LIMIT,
                    help="drop a factor correlated above this with one already kept")
    ap.add_argument("--max-decorrelate", type=int, default=200,
                    help="how many passing specs to decorrelate before taking --top")
    ap.add_argument("--gate", default="ts", choices=("ts", "cs", "any"),
                    help="ts = per-symbol block test (default, time-series focus)")
    ap.add_argument("--final", action="store_true",
                    help="evaluate the survivors on the holdout, once, forever")
    a = ap.parse_args()
    only = None
    if a.only:
        only = ([ln.strip() for ln in open(a.only) if ln.strip()]
                if os.path.exists(a.only) else
                [n.strip() for n in a.only.split(",") if n.strip()])
    out = report(a.tf, a.round, a.split, a.horizon, a.top, gate=a.gate,
                 max_decorrelate=a.max_decorrelate, corr_limit=a.corr_limit,
                 only=only)
    if a.final:
        if out.empty or not out["survives"].any():
            sys.exit("nothing survived; the holdout stays untouched")
        h = a.horizon if a.horizon is not None else int(
            pd.read_csv(os.path.join(REPORTS, f"sweep_{a.round}_{a.tf}_{a.split}.csv"))
            ["horizon"].median())
        final(a.tf, list(out[out["survives"]]["name"]), h, a.round)
