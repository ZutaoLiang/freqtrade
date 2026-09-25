"""r5 engine: panel-backed pre-screen, TRAIN/VALID discipline per skill §3/§4.

Panels: user_data/data/binance_public/panels/{5m,15m,1h,4h,1d}  2025-01-01..2026-08-16, 860 symbols.
Splits: TRAIN 2025-01-01..2025-10-01 | VALID 2025-10-01..2026-03-01 | HOLDOUT 2026-03-01..unread.
Fresh stage (independent-era gate): 1h_hist panel 2022-11-01..2025-12-31 (598 symbols).
Conventions: signal on bar close -> fill next bar open; SL wins over TP intrabar; one pos per coin;
cost 7.5bp (BTC/ETH) else 10bp per side; arithmetic returns; funding NOT modeled at this stage.
"""
from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd

PANELS = "/root/freqtrade/user_data/data/binance_public/panels"
TRAIN = ("2025-01-01", "2025-10-01")
VALID = ("2025-10-01", "2026-03-01")


class Panel:
    def __init__(self, tf: str):
        self.tf = tf
        self.dir = f"{PANELS}/{tf}"
        meta = json_meta = self._meta()
        self.symbols = meta["symbols"]
        step = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "1h_hist": "1h", "4h": "4h", "1d": "1D"}[tf]
        self.dates = pd.date_range(meta["start"], meta["end"], freq=step)
        self.n, self.m = len(self.dates), len(self.symbols)
        for k in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "count",
                  "funding_rate", "mark_close", "sum_open_interest", "sum_open_interest_value",
                  "sum_toptrader_long_short_ratio", "count_long_short_ratio", "count_toptrader_long_short_ratio",
                  "funding_interval_hours"):
            p = f"{self.dir}/{k}.npy"
            try:
                setattr(self, k, np.load(p, mmap_mode="r"))
            except FileNotFoundError:
                setattr(self, k, None)

    def _meta(self):
        import json
        with open(f"{self.dir}/meta.json") as f:
            return json.load(f)

    def idx(self, seg: tuple[str, str]) -> tuple[int, int]:
        i0 = int(np.searchsorted(self.dates, pd.Timestamp(seg[0], tz="UTC")))
        i1 = int(np.searchsorted(self.dates, pd.Timestamp(seg[1], tz="UTC")))
        return i0, i1

    def cost(self, sym: str) -> float:
        return 7.5 if sym in ("BTCUSDT", "ETHUSDT") else 10.0


@nb.njit(cache=True)
def _sim(o, h, l, c, sig_long, sig_short, hold, start, end, cost, sl, tp):
    k = 0
    busy = -1
    m = end - start
    ent = np.empty(m, np.int64); ext = np.empty(m, np.int64)
    sd = np.empty(m, np.int8); ret = np.empty(m, np.float64)
    fee = 2.0 * cost / 1e4
    for i in range(start, end - 1):
        if i < busy:
            continue
        lo = sig_long[i]; sh = sig_short[i]
        if not (lo or sh):
            continue
        d = np.int8(1) if lo else np.int8(-1)
        e = i + 1
        px = o[e]
        if not (px > 0.0) or np.isnan(px):
            continue
        stop = px * (1.0 - d * sl) if sl > 0 else 0.0
        take = px * (1.0 + d * tp) if tp > 0 else 1e18
        last = min(e + hold - 1, end - 1)
        xp = c[last]; xj = last
        for j in range(e, last + 1):
            if sl > 0:
                if d == 1 and l[j] <= stop:
                    xp = min(o[j], stop) if j > e else stop; xj = j; break
                if d == -1 and h[j] >= stop:
                    xp = max(o[j], stop) if j > e else stop; xj = j; break
            if tp > 0:
                if d == 1 and h[j] >= take:
                    xp = max(o[j], take) if j > e else take; xj = j; break
                if d == -1 and l[j] <= take:
                    xp = min(o[j], take) if j > e else take; xj = j; break
        ent[k] = e; ext[k] = xj; sd[k] = d
        ret[k] = d * (xp / px - 1.0) - fee
        k += 1
        busy = xj
    return ent[:k], ext[:k], sd[:k], ret[:k]


def evaluate(panel: Panel, long_sig: np.ndarray, short_sig: np.ndarray, hold: int, seg: tuple[str, str],
             universe: np.ndarray | None = None, sl: float = 0.0, tp: float = 0.0,
             names: dict[str, str] | None = None) -> dict:
    """long_sig/short_sig: (n,m) bool arrays (already causal). universe: column indices."""
    i0, i1 = panel.idx(seg)
    n_days = (panel.dates[i1 - 1] - panel.dates[i0]).days + 1
    cols = np.arange(panel.m) if universe is None else universe
    rows = []
    for col in cols:
        sym = panel.symbols[col]
        ent, ext, sd, ret = _sim(np.ascontiguousarray(panel.open[i0:i1, col], dtype=np.float64),
                                 np.ascontiguousarray(panel.high[i0:i1, col], dtype=np.float64),
                                 np.ascontiguousarray(panel.low[i0:i1, col], dtype=np.float64),
                                 np.ascontiguousarray(panel.close[i0:i1, col], dtype=np.float64),
                                 np.ascontiguousarray(long_sig[i0:i1, col]),
                                 np.ascontiguousarray(short_sig[i0:i1, col]),
                                 hold, 0, i1 - i0, panel.cost(sym), sl, tp)
        if len(ret):
            rows.append(pd.DataFrame({"ret": ret, "t": panel.dates[i0 + ent],
                                      "sym": sym, "side": sd}))
    if not rows:
        return {"n": 0}
    df = pd.concat(rows, ignore_index=True)
    df["day"] = df.t.dt.floor("D")
    n = len(df)
    r = df.ret
    g, b = r[r > 0].sum(), -r[r < 0].sum()
    dm = df.groupby("day").ret.mean()
    day_t = dm.mean() / dm.std() * np.sqrt(len(dm)) if len(dm) > 2 and dm.std() > 0 else float("nan")
    tot = r.sum()
    return {
        "n": int(n), "per_day": round(n / n_days, 2),
        "mean_bp": round(float(r.mean() * 1e4), 2),
        "med_bp": round(float(r.median() * 1e4), 2),
        "win": round(float((r > 0).mean()), 3),
        "pf": round(float(g / b), 3) if b > 0 else 99.0,
        "days": int(len(dm)), "day_mean_bp": round(float(dm.mean() * 1e4), 2),
        "day_t": round(float(day_t), 2), "days_pos": round(float((dm > 0).mean()), 3),
        "coin_share": round(float(df.groupby("sym").ret.sum().abs().max() / abs(tot)), 2) if tot != 0 else 0,
        "day_share": round(float(df.groupby("day").ret.sum().abs().max() / abs(tot)), 2) if tot != 0 else 0,
    }


def universe_cols(panel: Panel, path: str = "/root/freqtrade/user_data/minute_research/r3/universe_rank.csv",
                  min_rank: int = 1, max_rank: int = 60) -> np.ndarray:
    u = pd.read_csv(path)
    sel = u[(u.med_qv >= 0) & (u["rank"].between(min_rank, max_rank))] if "rank" in u.columns else u.head(max_rank)
    s = set(sel.base + "USDT") if "base" in u.columns else set()
    idx = [i for i, sym in enumerate(panel.symbols) if sym in s]
    return np.array(idx, dtype=np.int64)
