"""Signal confluence: trade only where several distinct factors agree.

Every evaluation so far has been single-factor, and single factors have now
been swept to exhaustion -- 4900 specs across four rounds, and the only ones
that clear trading cost are the perpetual carry series. That is the case for
testing what a discretionary trader actually does: wait for independent signals
to line up, accept far fewer trades, and hope the ones taken are better.

Confluence here means **direction agreement**, not a weighted sum. Each factor
is reduced to -1 / 0 / +1 by its cross-sectional quantile at each bar, and a
position is taken only where at least `k` of the `K` chosen factors point the
same way. That is a genuinely different object from a linear combination: it
concentrates into fewer names and it can fail in a way a sum cannot, by never
agreeing at all.

The multiple-testing hazard is severe -- C(N,k) grows fast enough that sweeping
combinations of a large pool guarantees a flattering winner. The pool is
therefore deliberately tiny and fixed in advance: one representative per
economic family, so that a combination means "carry plus trend plus flow"
rather than "three windows of the same idea".
"""
import argparse
import itertools
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
import falsify as FA  # noqa: E402
import ic as IC  # noqa: E402
from config import workers_for  # noqa: E402

REPORTS = os.path.join(HERE, "..", "reports")

# One per family, each the strongest survivor of its own kind in the rounds so
# far, so that agreement between two of them is agreement between two different
# statements about the market rather than two spellings of one.
POOL = {
    "carry":       "ts_ema(funding_rate,336h)",
    "basis":       "ts_mean(basis,72h)",
    "positioning": "ts_rank(ls_retail,12h)",
    "trend":       "ts_zscore(close,168h)",
    "oscillator":  "ts_rsi(close,168h)",
    "flow":        "ts_corr(close,log_qv,72h)",
}

# Fast signals only, and deliberately none that cleared cost on its own. The
# mixed pool answered a different question: gating a carry tilt destroys the one
# thing that made it work, its 0.004 turnover, so confluence lost there by
# construction. The claim worth testing is the trader's actual one -- that
# several fast, individually-marginal signals agreeing beats any of them alone.
# Every member here turns over at least an order of magnitude faster than carry.
FAST_POOL = {
    "reversal":    "ts_zscore(close,24h)",
    "oscillator":  "ts_rsi(close,24h)",
    "divergence":  "ts_corr(ret1,vol_ratio,24h)",
    "closeloc":    "ts_delta(close_loc,24h)",
    "vwap":        "ts_delta(vwap_dev,72h)",
    "body":        "ts_rank(body,336h)",
    "orderflow":   "ts_corr(taker_imbalance,vol_ratio,12h)",
    "positioning": "ts_rank(ls_retail,12h)",
}

POOLS = {"mixed": POOL, "fast": FAST_POOL}


def signal(factor, mask, direction, quantile=0.2):
    """-1 / 0 / +1 from the cross-sectional quantile of a factor at each bar.

    `direction` is the sign established on train: +1 means a high factor value
    is the long side. Everything between the quantile bands is 0, which is what
    makes agreement meaningful -- a factor with no opinion cannot vote.
    """
    f = IC.apply_mask(IC.lag(factor, 1), mask)
    r = IC._rank_axis1(f)                       # (0, 1] per bar, NaN outside
    s = np.zeros_like(r)
    s = np.where(r >= 1.0 - quantile, 1.0, s)
    s = np.where(r <= quantile, -1.0, s)
    s = np.where(np.isfinite(r), s, 0.0)
    return direction * s


def confluence_book(signals, k):
    """Dollar-neutral book over names where at least k signals agree."""
    stack = np.stack(signals, axis=0)           # (K, T, N)
    net = stack.sum(axis=0)
    votes_long = (stack > 0).sum(axis=0)
    votes_short = (stack < 0).sum(axis=0)
    w = np.where(votes_long >= k, 1.0, 0.0) - np.where(votes_short >= k, 1.0, 0.0)
    # A name with k longs and k shorts is not agreement; drop it.
    w = np.where((votes_long >= k) & (votes_short >= k), 0.0, w)
    gross = np.nansum(np.abs(w), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(gross > 0, w / gross, 0.0)
    return np.where(np.isfinite(w), w, 0.0), net


_CTX = {}


def _init(payload):
    _CTX.update(payload)


def _one(job):
    """Score one (combination, k) pair. Independent of every other job."""
    combo, k = job
    sig, ret1, h, tf, split_name = (_CTX["sig"], _CTX["ret1"], _CTX["h"],
                                    _CTX["tf"], _CTX["split"])
    w_raw, _ = confluence_book([sig[c] for c in combo], k)
    w = C.held_book(w_raw, h)
    pnl = np.nansum(w * ret1, axis=1)
    turn = np.concatenate([[np.nan], np.nansum(np.abs(np.diff(w, axis=0)), axis=1)])
    ok = np.isfinite(pnl) & np.isfinite(turn)
    pnl_v, turn_v = pnl[ok], turn[ok]
    if pnl_v.size < 100:
        return None
    mt = float(turn_v.mean())
    band = IC.andrews_band(pnl_v)
    _, _, t_nw, _ = IC.newey_west(pnl_v, band=band)
    sd = float(pnl_v.std(ddof=1))
    ann = np.sqrt(C.BARS_PER_YEAR[tf])
    net = pnl_v - turn_v * 5e-4
    return {
        "tf": tf, "split": split_name, "combo": "+".join(combo),
        "size": len(combo), "k": k,
        "names_active": float((np.abs(w_raw) > 0).sum(axis=1).mean()),
        "bars_with_position": float((np.abs(w_raw).sum(axis=1) > 0).mean()),
        "turnover": mt,
        "gross_bps": float(pnl_v.mean()) * 1e4,
        "gross_sharpe": float(pnl_v.mean() / sd * ann) if sd > 0 else np.nan,
        "pnl_t_nw": t_nw,
        "breakeven_bps": float(pnl_v.mean()) / mt * 1e4 if mt > 0 else np.inf,
        "net_sharpe_5bps": float(net.mean() / net.std(ddof=1) * ann),
    }


def evaluate(tf, split_name, pool, directions, ks=(1, 2, 3), quantile=0.2,
             horizon_hours=24.0, max_size=4):
    sl, close, mask, span = FA._slice(tf, split_name)
    specs = FA.spec_index()
    h = max(1, int(round(horizon_hours * B.BARS_PER_DAY[tf] / 24.0)))
    ret1 = IC.forward_return(close, 1)
    ret1 = np.where(np.isfinite(ret1), ret1, 0.0)

    sig = {}
    for family, name in pool.items():
        s = specs.get(name)
        if s is None or not s.valid(tf):
            print(f"  skip {family} ({name}): unavailable at {tf}", flush=True)
            continue
        f = s.compute(tf, lambda n: np.asarray(B.load(tf, n), dtype="float64"))[sl]
        sig[family] = signal(f, mask, directions[name], quantile)
    fams = sorted(sig)
    print(f"{tf} {split_name} {span[0].date()}..{span[1].date()}: "
          f"{len(fams)} families, holding {h} bars", flush=True)

    jobs = [(combo, k)
            for size in range(1, max_size + 1)
            for combo in itertools.combinations(fams, size)
            for k in ks if k <= size]
    payload = {"sig": sig, "ret1": ret1, "h": h, "tf": tf, "split": split_name}
    n = min(workers_for(tf, "sweep"), max(1, len(jobs)))
    print(f"  {len(jobs)} configurations, {n} workers", flush=True)
    with ProcessPoolExecutor(max_workers=n, initializer=_init,
                             initargs=(payload,)) as ex:
        rows = [r for r in ex.map(_one, jobs) if r is not None]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", default="1h,4h")
    ap.add_argument("--splits", default="train,valid")
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--horizon-hours", type=float, default=24.0)
    ap.add_argument("--max-size", type=int, default=4)
    ap.add_argument("--pool", default="mixed", choices=tuple(POOLS))
    ap.add_argument("--out", default=os.path.join(REPORTS, "combo.csv"))
    a = ap.parse_args()

    # Directions come from the train-split cost sweeps, never from this run.
    directions = {}
    for r in ("round2", "round3", "round4"):
        for tf in ("1h", "4h"):
            p = os.path.join(REPORTS, f"cost_{r}_{tf}.csv")
            if os.path.exists(p):
                d = pd.read_csv(p)
                d = d[d["split"] == "train"]
                directions.update(dict(zip(d["name"], d["sign"])))
    # A name the cost sweeps never covered -- ts_rsi postdates round 3 -- still
    # gets its direction the same way every other one did: the sign of the
    # decile mean spread on train. Defaulting it to +1 would smuggle in a free
    # parameter that nothing had fixed.
    pool = POOLS[a.pool]
    missing = [n for n in pool.values() if n not in directions]
    if missing:
        print(f"deriving train direction for {missing}", flush=True)
        specs = FA.spec_index()
        sl, close, mask, _ = FA._slice("1h", "train")
        for n in missing:
            sp = specs[n]
            f = sp.compute("1h", lambda x: np.asarray(B.load("1h", x),
                                                      dtype="float64"))[sl]
            h = max(1, int(round(a.horizon_hours * B.BARS_PER_DAY["1h"] / 24.0)))
            qp = C.quantile_profile(f, close, mask, horizon=h, lag_bars=1)
            directions[n] = 1.0 if qp["mean_spread_bps"] >= 0 else -1.0
            print(f"  {n}: mean spread {qp['mean_spread_bps']:+.1f} bps "
                  f"-> direction {int(directions[n]):+d}", flush=True)

    frames = []
    for split in a.splits.split(","):
        for tf in a.tfs.split(","):
            frames.append(evaluate(tf, split, pool, directions,
                                   quantile=a.quantile,
                                   horizon_hours=a.horizon_hours,
                                   max_size=a.max_size))
            pd.concat(frames, ignore_index=True).to_csv(a.out, index=False)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
