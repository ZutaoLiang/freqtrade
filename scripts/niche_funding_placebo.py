"""Control: is the funding signal adding anything over simply being short these coins?

Every qualifying trade is a 235-minute short on an alt perp during 2025-2026. If alts drift
down on their own, part of the measured edge is just that drift. For each event this draws
PLACEBO entries at random minutes in the same pair within a +/-`WINDOW_DAYS` neighbourhood
of the real entry, applies the identical exit rule, and reports the gap.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

KL = "/root/freqtrade/user_data/data/binance_public/freqtrade/futures"
EVENTS = "/root/freqtrade/user_data/niche_work/universe_events.parquet"
OUT = "/root/freqtrade/user_data/niche_work/placebo.parquet"
HOLD = 235
STOP = 0.15
COST = 0.0020
LIQ = 0.99
WINDOW_DAYS = 7
DRAWS = 20


def _outcome(opens, closes, p, side):
    entry = opens[p]
    seg = side * (closes[p:p + HOLD + 1] / entry - 1.0)
    nxt = side * (opens[p + 1:p + HOLD + 2] / entry - 1.0)
    hit = np.flatnonzero(seg <= -STOP)
    ret = nxt[hit[0]] if len(hit) else seg[-1]
    return max(ret, -LIQ) - COST


def _run(args):
    sym, ev = args
    try:
        df = pd.read_feather(f"{KL}/{sym}-1m-futures.feather").set_index("date")
        df = df[~df.index.duplicated()]
    except Exception:
        return None
    opens, closes, idx = df["open"].values, df["close"].values, df.index
    rng = np.random.default_rng(abs(hash(sym)) % 2**32)
    span = HOLD + 2
    rows = []
    for e in ev.itertuples():
        p0 = idx.searchsorted(e.date)
        if p0 >= len(idx) - span or idx[p0] != e.date:
            continue
        lo = max(0, p0 - WINDOW_DAYS * 1440)
        hi = min(len(idx) - span, p0 + WINDOW_DAYS * 1440)
        if hi <= lo:
            continue
        draws = rng.integers(lo, hi, DRAWS)
        rows.append(dict(sym=sym, date=e.date, side=e.side,
                         real=_outcome(opens, closes, p0, e.side),
                         placebo=float(np.mean([_outcome(opens, closes, int(p), e.side)
                                                for p in draws]))))
    return pd.DataFrame(rows) if rows else None


def main():
    d = pd.read_parquet(EVENTS)
    d = d[(d.trail_qv >= 1e7) & (d.fr_bps.abs() >= 40) & (d.streak_40 >= 1)]
    groups = [(s, g) for s, g in d.groupby("sym")]
    print(f"events {len(d)}  pairs {len(groups)}", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        for r in ex.map(_run, groups, chunksize=2):
            if r is not None:
                out.append(r)
    r = pd.concat(out, ignore_index=True)
    r.to_parquet(OUT)
    gap = r.real - r.placebo
    print(f"\nn={len(r)}  (每个事件配 {DRAWS} 个同 pair、±{WINDOW_DAYS} 天内的随机入场)")
    print(f"  真实事件      mean {1e4*r.real.mean():+7.1f}bps   median {1e4*r.real.median():+7.1f}")
    print(f"  安慰剂(同向)  mean {1e4*r.placebo.mean():+7.1f}bps   median {1e4*r.placebo.median():+7.1f}")
    print(f"  差值          mean {1e4*gap.mean():+7.1f}bps   median {1e4*gap.median():+7.1f}   "
          f"P(gap>0)={np.mean(gap > 0):.3f}")
    rng = np.random.default_rng(0)
    g = {k: v.values for k, v in gap.groupby(pd.PeriodIndex(r.date, freq="M").astype(str))}
    keys = list(g)
    bs = np.array([np.concatenate([g[keys[i]] for i in rng.choice(len(keys), len(keys), True)]).mean()
                   for _ in range(20000)])
    print(f"  差值按月分块 bootstrap 95% CI [{1e4*np.percentile(bs,2.5):+7.1f}, "
          f"{1e4*np.percentile(bs,97.5):+7.1f}]bps   P(<=0)={np.mean(bs <= 0):.3f}")


if __name__ == "__main__":
    main()
