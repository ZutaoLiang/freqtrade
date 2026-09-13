"""One-pass sweep that buckets every trade by (volatility regime at entry) x (split of the entry date).

Frozen in reports/PREREGISTRATION_vol_regime.md. Runs each spec once over the whole 2022-11..2026-08
history -- a third of the compute of three separate split runs -- and emits one row per
(spec, screen, side, hold, regime, split).

t-stat: trades are grouped by entry date, the daily mean net return is taken, and a Newey-West t with
the Andrews bandwidth is computed on that series. Trades entered the same day share a market shock,
so treating them as independent would overstate significance.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import base as B  # noqa: E402
import factors as F  # noqa: E402
import ic as IC  # noqa: E402
import screens as S  # noqa: E402
import ts_book as TB  # noqa: E402
import vol_regime as VR  # noqa: E402
from config import workers_for  # noqa: E402
from panel import Panel  # noqa: E402

REPORTS = os.path.join(HERE, "..", "reports")
SPLITS = {"train": ("2022-11-01", "2025-01-01"), "valid": ("2025-01-01", "2026-01-01"),
          "holdout": ("2026-01-01", "2026-08-17")}
SIDES = ("long", "short")
_CTX = {}


def _init(tf, holds_h, rank_win, screens_wanted, regime_kw):
    p = Panel(tf)
    bpd = B.BARS_PER_DAY[tf]
    scr = {k: np.asarray(v) for k, v in S.load(tf).items() if (screens_wanted is None or k in screens_wanted)}
    fr = np.asarray(B.load(tf, "funding_rate"), dtype="float64") if "funding_rate" in B.names(tf) else np.zeros(p.shape)
    fr = np.where(np.isfinite(fr), fr, 0.0)
    fund_per_bar = fr * (24.0 / bpd) / 8.0
    state = VR.states_for(tf, **regime_kw)
    split_id = np.full(len(p.index), "none", dtype=object)
    for name, (lo, hi) in SPLITS.items():
        m = (p.index >= pd.Timestamp(lo, tz="UTC")) & (p.index < pd.Timestamp(hi, tz="UTC"))
        split_id[m] = name
    # signals are eligible over the whole history; the bucket comes from the entry bar
    sl = slice(0, len(p.index))
    _CTX.update(tf=tf, sl=sl, bpd=bpd, screens=scr, open=np.asarray(p["open"], dtype="float64"),
                fund=fund_per_bar, index=p.index, holds=tuple(holds_h), rank_win=rank_win,
                state=state, split_id=split_id, day=p.index.floor("1D").to_numpy())


def _summ(df, day, idx):
    """Mean / median / win / cluster-robust t for one bucket of trades."""
    n = len(df)
    if n < 10:
        return {"n_trades": n}
    daily = pd.Series(df.net5.values, index=day[df.t.values]).groupby(level=0).mean()
    if len(daily) > 20:
        band = IC.andrews_band(daily.values)
        _, _, t, _ = IC.newey_west(daily.values, band=band)
    else:
        t = np.nan
    q = pd.Series(df.net5.values, index=idx[df.t.values]).groupby(pd.Grouper(freq="QE")).mean()
    return {"n_trades": n, "n_symbols": int(df.sym.nunique()), "n_days": int(len(daily)),
            "mean_net5_bps": float(df.net5.mean() * 1e4), "median_net5_bps": float(df.net5.median() * 1e4),
            "mean_net10_bps": float(df.net10.mean() * 1e4), "sd_net5_bps": float(df.net5.std() * 1e4),
            "win": float((df.net5 > 0).mean()), "t_cluster": float(t),
            "quarters_pos": int((q > 0).sum()), "quarters": int(q.notna().sum())}


def _run(spec):
    tf = _CTX["tf"]
    try:
        f = spec.compute(tf, lambda n: np.asarray(B.load(tf, n), dtype="float64"))
    except Exception as exc:  # noqa: BLE001
        return [{"name": spec.name(), "error": repr(exc)}]
    long_sig, short_sig = TB.entries(f, _CTX["rank_win"])
    state, split_id, day, idx = _CTX["state"], _CTX["split_id"], _CTX["day"], _CTX["index"]
    rows = []
    for sname, scr in _CTX["screens"].items():
        for side in SIDES:
            sig = (long_sig if side == "long" else short_sig) & scr
            if not sig.any():
                continue
            for hh in _CTX["holds"]:
                h = max(1, int(round(hh * _CTX["bpd"] / 24)))
                df, _ = TB.simulate_from_signal(sig, side, _CTX["open"], _CTX["fund"], h, _CTX["sl"])
                if df.empty:
                    continue
                df = df.assign(regime=state[df.t.values], split=split_id[df.t.values])
                for (reg, sp), g in df.groupby(["regime", "split"], observed=True):
                    if sp == "none" or reg == "NA":
                        continue
                    r = _summ(g, day, idx)
                    r.update(name=spec.name(), tag=spec.tag, screen=sname, side=side,
                             hold_h=hh, regime=reg, split=sp)
                    rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d_long")
    ap.add_argument("--label", default="vr")
    ap.add_argument("--specs", default="daily")
    ap.add_argument("--holds", default="24,72,168,336,672")
    ap.add_argument("--rank-win", type=int, default=90)
    ap.add_argument("--screens", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rv-days", type=int, default=VR.RV_DAYS)
    ap.add_argument("--rank-days", type=int, default=VR.RANK_DAYS)
    ap.add_argument("--lo-q", type=float, default=VR.LO_Q)
    ap.add_argument("--hi-q", type=float, default=VR.HI_Q)
    ap.add_argument("--kind", default="btc")
    a = ap.parse_args()
    specs = {"round1": F.ROUND1, "round2": F.ROUND2(), "round3": F.ROUND3(), "daily": F.DAILY()}[a.specs]
    keep, dropped = F.for_timeframe(specs, a.tf, B.names(a.tf))
    if a.limit:
        keep = keep[: a.limit]
    wanted = set(a.screens.split(",")) if a.screens else None
    holds = tuple(int(x) for x in a.holds.split(","))
    regime_kw = dict(rv_days=a.rv_days, rank_days=a.rank_days, lo_q=a.lo_q, hi_q=a.hi_q, kind=a.kind)
    n_workers = a.workers or workers_for(a.tf, "sweep")
    print(f"{a.tf}: {len(keep)} specs ({len(dropped)} dropped), holds {holds}, rank win {a.rank_win}, "
          f"regime {regime_kw}, {n_workers} workers", flush=True)
    t0, out = time.time(), []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init,
                             initargs=(a.tf, holds, a.rank_win, wanted, regime_kw)) as ex:
        for i, rows in enumerate(ex.map(_run, keep), 1):
            out.extend(rows)
            if i % 20 == 0:
                print(f"  {i}/{len(keep)}  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(out)
    path = os.path.join(REPORTS, f"vr_{a.label}_{a.tf}.parquet")
    df.to_parquet(path, index=False)
    print(f"wrote {path}: {len(df)} rows, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
