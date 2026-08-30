"""Scan every USDT perp for funding-skew events, tagged with liquidity known at the time.

The earlier study fixed a 3M-60M median-volume band chosen from a window that overlapped
the test period. Here nothing is pre-selected: every pair with both klines and funding is
scanned over its whole history, and each event carries the pair's TRAILING 30-day median
daily quote volume computed strictly from days before the event. Bucketing happens
downstream, so liquidity is a result rather than a sampling decision.

Outcomes are simple returns net of a 20bps round trip, with a close-based stop acted on at
the next bar's open, matching scripts/niche_funding_path_sim.py.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

KL = "/root/freqtrade/user_data/data/binance_public/freqtrade/futures"
FUND = "/root/freqtrade/user_data/data/binance_public/funding"
OUT = "/root/freqtrade/user_data/niche_work/universe_events.parquet"
MIN_ABS_FR = 30.0
THRESHOLDS = (30.0, 40.0, 60.0, 80.0)
HOLDS = (120, 235)
STOPS = (10, 15, 25, None)
COST = 0.0020
LIQ = 0.99
VOL_DAYS = 30


def _run(sym):
    fsym = sym.replace("_USDT_USDT", "USDT")
    fp = f"{FUND}/{fsym}.parquet"
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_feather(f"{KL}/{sym}-1m-futures.feather").set_index("date")
        df = df[~df.index.duplicated()]
        fr = pd.read_parquet(fp).set_index("date")["funding_rate"] * 1e4
        fr = fr[~fr.index.duplicated()].sort_index()
    except Exception:
        return None
    if len(df) < 5000 or fr.empty:
        return None

    # liquidity as of the event: median daily quote volume over the prior VOL_DAYS days
    daily = (df["close"] * df["volume"]).resample("1D").sum()
    trail = daily.rolling(VOL_DAYS, min_periods=10).median().shift(1)

    streaks = {}
    for th in THRESHOLDS:
        state = np.sign(fr) * (fr.abs() >= th)
        streaks[th] = state.groupby((state != state.shift()).cumsum()).cumcount()

    hits = fr[fr.abs() >= MIN_ABS_FR]
    if hits.empty:
        return None
    idx = df.index
    entry_ts = hits.index + pd.Timedelta(minutes=1)
    pos = idx.searchsorted(entry_ts)
    span = max(HOLDS) + 2
    ok = (pos < len(idx) - span) & (idx[np.clip(pos, 0, len(idx) - 1)] == entry_ts)
    pos, stamps = pos[ok], hits.index[ok]
    if len(pos) == 0:
        return None

    opens, closes = df["open"].values, df["close"].values
    rows = []
    for p, stamp in zip(pos, stamps):
        rate = fr.loc[stamp]
        side = 1.0 if rate > 0 else -1.0
        entry = opens[p]
        path = side * (closes[p:p + span] / entry - 1.0)
        nxt = side * (opens[p + 1:p + span + 1] / entry - 1.0)
        rec = dict(sym=sym, stamp=stamp, date=idx[p], fr_bps=rate, side=side,
                   trail_qv=trail.get(stamp.normalize(), np.nan))
        for th in THRESHOLDS:
            rec[f"streak_{th:g}"] = streaks[th].loc[stamp]
        for hold in HOLDS:
            seg = path[:hold + 1]
            for stop in STOPS:
                if stop is None:
                    ret = seg[-1]
                else:
                    hit = np.flatnonzero(seg <= -stop / 100)
                    ret = nxt[hit[0]] if len(hit) else seg[-1]
                rec[f"h{hold}_s{'none' if stop is None else stop}"] = max(ret, -LIQ) - COST
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    syms = sorted({os.path.basename(f).replace("-1m-futures.feather", "")
                   for f in os.listdir(KL) if f.endswith("_USDT_USDT-1m-futures.feather")})
    print(f"symbols: {len(syms)}", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        for d in ex.map(_run, syms, chunksize=2):
            if d is not None:
                out.append(d)
    d = pd.concat(out, ignore_index=True)
    d.to_parquet(OUT)
    print(f"events {len(d)}  pairs {d.sym.nunique()}  {d.date.min():%Y-%m-%d} .. {d.date.max():%Y-%m-%d}")
    print(f"trailing volume known for {d.trail_qv.notna().mean():.1%} of events")


if __name__ == "__main__":
    main()
