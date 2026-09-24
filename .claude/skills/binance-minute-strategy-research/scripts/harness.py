"""Fast pre-screen harness: test a signal on freqtrade feather data before writing a strategy.

Import it from an iteration script:

    import sys; sys.path.insert(0, ".claude/skills/binance-minute-strategy-research/scripts")
    import harness as H
    d = H.load("user_data/data/binance", "BTC", "1m")            # dict of numpy arrays + 'date'
    sig = np.where(my_condition(d))[0]                             # bar indexes, signal on the close
    tr = H.run(d, sig, side=+1, hold=60, sl=0.01, tp=0.02, cost_bps=7.5)
    print(H.report({"BTC": tr}, oos_start="2026-01-01"))

Conventions (identical to what freqtrade does, so the numbers carry over):
  * signal on the close of bar i -> fill at open of bar i+1; one position per pair at a time;
  * stop / take-profit checked on the bar low/high; if both are touched in one bar the stop wins;
    a gap through the stop fills at the open;
  * costs per side in bps on top of the price move; arithmetic returns (never log);
  * funding is NOT charged here. Anything that holds across settlements must be re-checked in
    freqtrade with real funding (validate_datadir.py first).
Memory: float64 arrays, one pair ~ 40 MB for 20 months of 1m. Load pairs one at a time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from numba import njit


def load(datadir: str, base: str, tf: str = "1m") -> dict[str, np.ndarray]:
    path = f"{datadir}/futures/{base}_USDT_USDT-{tf}-futures.feather"
    with pa.memory_map(path) as src:
        t = ipc.open_file(src).read_all()
    d = {k: t.column(k).to_numpy().astype(np.float64) for k in ("open", "high", "low", "close", "volume")}
    d["date"] = t.column("date").to_pandas()
    return d


@njit(cache=True)
def _simulate(o, h, l, c, sig_idx, sig_side, hold, sl, tp, cost_bps, exit_sig):
    n = o.shape[0]
    m = sig_idx.shape[0]
    ent = np.empty(m, np.int64); ext = np.empty(m, np.int64)
    sd = np.empty(m, np.int8); ret = np.empty(m, np.float64)
    k = 0
    busy = -1
    cost = 2.0 * cost_bps / 1e4
    use_exit = exit_sig.shape[0] == n
    for q in range(m):
        i = sig_idx[q]
        e = i + 1
        if e >= n or i < busy:
            continue
        d = sig_side[q]
        px = o[e]
        stop = px * (1.0 - d * sl)
        take = px * (1.0 + d * tp)
        last = min(e + hold - 1, n - 1)
        xp = c[last]; xj = last
        for j in range(e, last + 1):
            if sl > 0.0:
                if d == 1 and l[j] <= stop:
                    xp = min(o[j], stop) if j > e else stop; xj = j; break
                if d == -1 and h[j] >= stop:
                    xp = max(o[j], stop) if j > e else stop; xj = j; break
            if tp > 0.0:
                if d == 1 and h[j] >= take:
                    xp = max(o[j], take) if j > e else take; xj = j; break
                if d == -1 and l[j] <= take:
                    xp = min(o[j], take) if j > e else take; xj = j; break
            if use_exit and exit_sig[j] and j + 1 < n:
                xp = o[j + 1]; xj = j + 1; break
        ent[k] = e; ext[k] = xj; sd[k] = d
        ret[k] = d * (xp / px - 1.0) - cost
        k += 1
        busy = xj
    return ent[:k], ext[:k], sd[:k], ret[:k]


_NO_EXIT = np.zeros(1, dtype=np.bool_)


def run(d, sig_idx, side, hold, sl=0.0, tp=0.0, cost_bps=7.5, exit_sig=None) -> pd.DataFrame:
    """side: +1 / -1 scalar or per-signal array. hold in bars. exit_sig: bool array (exit next open)."""
    sig_idx = np.asarray(sig_idx, np.int64)
    side = np.broadcast_to(np.asarray(side, np.int8), sig_idx.shape)
    order = np.argsort(sig_idx, kind="stable")
    ex = _NO_EXIT if exit_sig is None else np.asarray(exit_sig, np.bool_)
    e, x, s, r = _simulate(d["open"], d["high"], d["low"], d["close"], sig_idx[order], side[order].copy(),
                           int(hold), float(sl), float(tp), float(cost_bps), ex)
    return pd.DataFrame({"entry": e, "exit": x, "side": s, "ret": r,
                         "t": d["date"].iloc[e].to_numpy() if len(e) else pd.Series([], dtype="datetime64[ms, UTC]")})


def stats(r: np.ndarray) -> dict:
    if len(r) == 0:
        return {"n": 0}
    g, b = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": int(len(r)), "mean_bp": round(float(r.mean() * 1e4), 2), "med_bp": round(float(np.median(r) * 1e4), 2),
            "win": round(float((r > 0).mean()), 3), "pf": round(float(g / b), 3) if b > 0 else float("inf")}


def day_clustered(trades: dict[str, pd.DataFrame]) -> dict:
    """Mean of per-day mean returns and its t-stat. Events on alts cluster on crash days;
    trade-level t-stats overstate evidence by an order of magnitude. Always read this one."""
    parts = [pd.Series(t.ret.to_numpy(), index=pd.DatetimeIndex(t.t).floor("D")) for t in trades.values() if len(t)]
    if not parts:
        return {"days": 0}
    dm = pd.concat(parts).groupby(level=0).mean()
    t = dm.mean() / dm.std() * np.sqrt(len(dm)) if len(dm) > 2 and dm.std() > 0 else float("nan")
    return {"days": int(len(dm)), "day_mean_bp": round(float(dm.mean() * 1e4), 2), "day_t": round(float(t), 2),
            "days_pos": round(float((dm > 0).mean()), 3)}


def report(trades: dict[str, pd.DataFrame], oos_start: str) -> str:
    """IS / OOS split by entry time (column t). One line per segment: trade stats + day-clustered."""
    cut = pd.Timestamp(oos_start, tz="UTC")
    lines = []
    for seg in ("IS", "OOS"):
        sub = {p: t[(t.t < cut) if seg == "IS" else (t.t >= cut)] for p, t in trades.items()}
        allr = np.concatenate([t.ret.to_numpy() for t in sub.values()]) if sub else np.array([])
        lines.append(f"{seg:3s} {stats(allr)} {day_clustered(sub)}")
    return "\n".join(lines)
