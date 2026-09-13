"""Many out-of-sample confirmations without spending the holdout.

Two complementary designs over the **development set** (train + valid, the
first 475 days). The holdout is never read here.

*Walk-forward* asks the question deployment asks: does a factor selected on the
past predict the future? Every fold trains on everything up to a date and tests
on the month after it, so the arrow of time is never reversed. Eight folds give
eight independent forward confirmations.

*CPCV* -- combinatorial purged cross-validation -- asks a different question:
is the effect real at all, anywhere in the sample? It cuts the development set
into blocks, tests on every pair of them, and reports the resulting spread. It
does evaluate on periods that precede part of the selection window, which is
why it cannot replace walk-forward; it is a stability check, and reporting a
distribution beats reporting one number.

Both purge and embargo around every test window. An h-bar forward return that
straddles a boundary would otherwise carry information across it, which is the
standard way a naive time-series split leaks.
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
import falsify as FA  # noqa: E402
import ic as IC  # noqa: E402
from panel import CACHE, Panel  # noqa: E402
from config import workers_for  # noqa: E402

REPORTS = os.path.join(HERE, "..", "reports")
EMBARGO_FRAC = 0.01
MIN_TEST_BARS = 200


def dev_span():
    """Development set = train + valid. The holdout is deliberately excluded."""
    sp = json.load(open(os.path.join(CACHE, "splits.json")))
    return pd.Timestamp(sp["train"][0]), pd.Timestamp(sp["valid"][1])


def _panel_slice(tf):
    lo, hi = dev_span()
    p = Panel(tf)
    a = int(np.searchsorted(p.index, lo))
    b = int(np.searchsorted(p.index, hi))
    mask = np.asarray(np.load(os.path.join(CACHE, tf, "universe_mask.npy"),
                              mmap_mode="r")[a:b])
    close = np.asarray(p["close"][a:b], dtype="float64")
    return slice(a, b), close, mask, p.index[a:b]


def walk_forward_windows(n_bars, h, n_folds=8, initial_frac=0.45):
    """Anchored expanding windows: train grows, test is the slice after it.

    The fold count is capped by how many test windows of at least
    MIN_TEST_BARS actually fit. Asking for eight folds of a 2850-bar 4h
    development set produced 196-bar windows, every one of which fell below the
    minimum and was dropped, leaving zero folds -- so the cap is computed here
    rather than letting the filter silently empty the list.
    """
    embargo = max(h, int(EMBARGO_FRAC * n_bars))
    start = int(initial_frac * n_bars)
    usable = n_bars - start - embargo
    feasible = max(1, usable // MIN_TEST_BARS)
    if feasible < n_folds:
        print(f"  only {feasible} folds fit ({usable} bars after the initial "
              f"window, {MIN_TEST_BARS} minimum per test); asked for {n_folds}")
        n_folds = feasible
    step = max(MIN_TEST_BARS, usable // n_folds)
    out = []
    for i in range(n_folds):
        a = start + i * step + embargo
        b = min(n_bars, a + step)
        if b - a >= MIN_TEST_BARS:
            out.append((a, b))
    return out


def cpcv_windows(n_bars, h, n_blocks=6, k=2):
    """Every choice of k test blocks out of n, purged at each boundary."""
    edges = np.linspace(0, n_bars, n_blocks + 1).astype(int)
    embargo = max(h, int(EMBARGO_FRAC * n_bars))
    out = []
    for combo in itertools.combinations(range(n_blocks), k):
        spans = []
        for b in combo:
            a = edges[b] + (embargo if b > 0 else 0)
            z = edges[b + 1] - (embargo if b < n_blocks - 1 else 0)
            if z - a >= MIN_TEST_BARS:
                spans.append((a, z))
        if spans:
            out.append((combo, spans))
    return out


def _ic_on(f, close, mask, spans, h):
    """Pooled per-symbol IC over the union of a set of windows."""
    fs = np.concatenate([f[a:b] for a, b in spans], axis=0)
    rs = np.concatenate([IC.apply_mask(IC.forward_return(close[a:b], h),
                                       mask[a:b]) for a, b in spans], axis=0)
    fs = IC.apply_mask(fs, np.concatenate([mask[a:b] for a, b in spans], axis=0))
    ic, _ = IC.rank_ic_per_symbol(fs, rs)
    null = IC.ts_null_test(fs, rs)
    return float(np.nanmean(ic)), null["z"], fs.shape[0]


# One factor per worker. Each evaluates 8 walk-forward folds and 15 CPCV
# combinations, all independent of every other factor, so this was 33 factors
# queued behind one core while 63 idled.
_CTX = {}


def _init(tf, horizon_hours):
    sl, close, mask, index = _panel_slice(tf)
    h = max(1, int(round(horizon_hours * B.BARS_PER_DAY[tf] / 24.0)))
    _CTX.update(tf=tf, sl=sl, close=close, mask=mask, h=h, specs=spec_index_cached())


def spec_index_cached():
    return FA.spec_index()


def _evaluate(args):
    name, wf, cp = args
    tf, sl, close, mask, h = (_CTX["tf"], _CTX["sl"], _CTX["close"],
                              _CTX["mask"], _CTX["h"])
    s = _CTX["specs"].get(name)
    if s is None or not s.valid(tf):
        return {"name": name, "tf": tf, "note": "unavailable at this timeframe"}
    f = s.compute(tf, lambda x: np.asarray(B.load(tf, x), dtype="float64"))[sl]
    f = IC.lag(f, 1)

    wf_ic, wf_z = [], []
    for a, b in wf:
        i, z, _ = _ic_on(f, close, mask, [(a, b)], h)
        wf_ic.append(i)
        wf_z.append(z)
    wf_ic = np.array(wf_ic, dtype="float64")
    wf_z = np.array(wf_z, dtype="float64")
    have_wf = wf_ic.size > 0 and np.isfinite(wf_ic).any()
    ref = np.sign(np.nanmean(wf_ic)) if have_wf else 0.0

    cp_ic = np.array([_ic_on(f, close, mask, spans, h)[0] for _, spans in cp],
                     dtype="float64")
    if not have_wf and np.isfinite(cp_ic).any():
        ref = np.sign(np.nanmean(cp_ic))

    return {
        "name": name, "tf": tf,
        "wf_folds": int(np.isfinite(wf_ic).sum()),
        "wf_ic_mean": float(np.nanmean(wf_ic)) if have_wf else np.nan,
        "wf_ic_worst": (float(np.nanmin(wf_ic * ref) * ref)
                        if have_wf and ref else np.nan),
        "wf_sign_agree": (float(np.nanmean(np.sign(wf_ic) == ref))
                          if have_wf else np.nan),
        "wf_frac_z2": (float(np.nanmean(np.abs(wf_z) > 2.0))
                       if have_wf else np.nan),
        "cpcv_paths": int(np.isfinite(cp_ic).sum()),
        "cpcv_ic_mean": float(np.nanmean(cp_ic)),
        "cpcv_ic_sd": float(np.nanstd(cp_ic, ddof=1)),
        "cpcv_ic_p05": float(np.nanpercentile(cp_ic, 5)),
        "cpcv_sign_agree": float(np.nanmean(np.sign(cp_ic) == ref)),
    }


def run(tf, names, horizon_hours=24.0, n_folds=8, n_blocks=6, k=2):
    sl, close, mask, index = _panel_slice(tf)
    h = max(1, int(round(horizon_hours * B.BARS_PER_DAY[tf] / 24.0)))
    n = close.shape[0]
    wf = walk_forward_windows(n, h, n_folds)
    cp = cpcv_windows(n, h, n_blocks, k)
    print(f"{tf}: dev set {index[0].date()}..{index[-1].date()}, {n} bars, "
          f"horizon {h} bars", flush=True)
    print(f"  walk-forward folds: {len(wf)}  "
          f"({', '.join(str(index[a].date()) + '..' + str(index[b-1].date()) for a, b in wf)})",
          flush=True)
    print(f"  CPCV combinations: {len(cp)} over {n_blocks} blocks, "
          f"{k} tested at a time", flush=True)
    del close, mask

    n_workers = workers_for(tf, "sweep")
    print(f"  {n_workers} workers over {len(names)} factors", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init,
                             initargs=(tf, horizon_hours)) as ex:
        for r in ex.map(_evaluate, [(nm, wf, cp) for nm in names]):
            if r.get("note"):
                print(f"  skip {r['name']}: {r['note']}", flush=True)
                continue
            rows.append(r)
            wf_txt = ("wf    n/a           " if not np.isfinite(r["wf_ic_mean"]) else
                      f"wf {r['wf_ic_mean']:+.4f} sign {r['wf_sign_agree']:.0%} "
                      f"z>2 {r['wf_frac_z2']:.0%}")
            print(f"  {r['name']:40s} {wf_txt} | "
                  f"cpcv {r['cpcv_ic_mean']:+.4f} +-{r['cpcv_ic_sd']:.4f} "
                  f"sign {r['cpcv_sign_agree']:.0%}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--tfs", default="1h,4h")
    ap.add_argument("--horizon-hours", type=float, default=24.0)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(REPORTS, "walkforward.csv"))
    a = ap.parse_args()
    names = [ln.strip() for ln in open(a.names) if ln.strip()]
    frames = []
    for tf in a.tfs.split(","):
        frames.append(run(tf, names, a.horizon_hours, a.folds, a.blocks, a.k))
        # Checkpoint after every timeframe: the first version wrote only at the
        # end, so a crash on the last one threw away everything before it.
        pd.concat(frames, ignore_index=True).to_csv(a.out, index=False)
    df = pd.concat(frames, ignore_index=True)
    print(f"wrote {a.out}")
