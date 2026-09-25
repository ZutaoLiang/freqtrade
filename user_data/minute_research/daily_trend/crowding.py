"""Daily crowding entries with long-hold exits (pre-registered in this docstring, 2026-09-25, before any run).

Universe U162 (static). Signal on day d's close, entry d+1 open, one position per coin, cost 10 bp/side, real daily funding
(longs pay positive funding). Absolute returns only (no market adjustment).

Family A — funding crowding (panel 1d_long, 2022-11..2026-08-16): fr7 = 7-day mean of daily funding (sum of the day's settlements).
  A_short: fr7 crosses up through +F (longs crowded)   -> short
  A_long : fr7 crosses down through -F (shorts crowded) -> long            F in {0.10%, 0.20%} per day
Family B — retail positioning crowding (panel 1d, 2025-01..2026-08-16): p = per-coin trailing-90-day percentile of the retail
  long/short account ratio (min 30 days); oi7 = 7-day change of open interest value.
  B_short: p crosses up through 0.95 while oi7 > 0 (crowded longs being added) -> short
  B_long : p crosses down through 0.05 while oi7 > 0                            -> long
Confirmation: none | trend agrees on the signal day (EMA10 below EMA30 for shorts, above for longs).
Exits (all with a 20% hard stop on price, intraday, filled at the stop or at the open if gapped; max hold 120 days):
  CH5 / CH7 chandelier: close beyond the most favourable extreme since entry by 5 / 7 x ATR22 -> exit next open
  MAX       opposite EMA10/30 cross -> exit next open
CORRECTION (2026-09-25, after the first run, before any rule change): the panels' funding_rate is NOT the daily sum
(1d = last settlement of the day, 1d_long = mean of the day's settlements). Daily funding is rebuilt as the true sum of
raw settlements (binance_public/funding for 2025+, binance-hist 1h funding feathers before) and used both for fr7 and for
the funding charged while holding. Rules and thresholds unchanged; the first run is discarded.
Grid: A 2 sides x 2 F x 2 conf x 3 exits = 24; B 2 sides x 2 conf x 3 exits = 12.
Segments: DEV (A: 2022-11-01..2025-10-01, B: 2025-01-01..2025-10-01), VALID-C 2025-12-01..2026-03-01,
HOLDOUT 2026-03-01..2026-08-17 read once with --holdout for at most three survivors.
Gates (absolute): DEV mean > 0, PF >= 1.2, week-clustered t >= 2.0, >= 0.2 trades/day across U162 (skill relaxed floor is 0.4,
reported); VALID-C mean > 0 and PF >= 1.2.
"""
import itertools, json, sys
import numba as nb
import numpy as np, pandas as pd

BASE = "/root/freqtrade/user_data/data/binance_public/panels"
U = pd.read_csv("/root/freqtrade/user_data/minute_research/r3/universe_rank.csv")
SEGS = {"VALID-C": ("2025-12-01", "2026-03-01"), "HOLDOUT": ("2026-03-01", "2026-08-17")}
DEV = {"A": ("2022-11-01", "2025-10-01"), "B": ("2025-01-01", "2025-10-01")}


class Panel:
    def __init__(self, tf):
        P = f"{BASE}/{tf}"; m = json.load(open(f"{P}/meta.json"))
        self.syms = m["symbols"]; self.dates = pd.date_range(m["start"], m["end"], freq="D")
        self.cols = [self.syms.index(b + "USDT") for b in U[U.med_qv >= 1e7].base if b + "USDT" in self.syms]
        self.ld = lambda f: np.array(np.load(f"{P}/{f}.npy", mmap_mode="r"), dtype=np.float64)[:, self.cols]
        self.O, self.H, self.L, self.C = (self.ld(f) for f in ("open", "high", "low", "close"))
        self.FR = daily_funding_sum([self.syms[c] for c in self.cols], self.dates)


def daily_funding_sum(syms, dates):
    import os
    out = np.zeros((len(dates), len(syms)))
    for k, s in enumerate(syms):
        parts = []
        h = f"/root/freqtrade/user_data/data/binance-hist/futures/{s[:-4]}_USDT_USDT-1h-funding_rate.feather"
        if os.path.exists(h):
            x = pd.read_feather(h)[["date", "open"]].rename(columns={"open": "fr"}); parts.append(x[x.date < "2025-01-01"])
        pth = f"/root/freqtrade/user_data/data/binance_public/funding/{s}.parquet"
        if os.path.exists(pth):
            y = pd.read_parquet(pth)[["date", "funding_rate"]].rename(columns={"funding_rate": "fr"}); parts.append(y[y.date >= "2025-01-01"])
        if not parts:
            continue
        f = pd.concat(parts); f["date"] = pd.to_datetime(f.date, utc=True).dt.round("h")
        f = f.drop_duplicates("date", keep="last")
        d = f.groupby(f.date.dt.floor("D")).fr.sum()
        out[:, k] = d.reindex(dates).fillna(0.0).to_numpy()
    return out


def ema(x, n): return pd.DataFrame(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def atr22(p):
    pc = np.vstack([np.full((1, p.C.shape[1]), np.nan), p.C[:-1]])
    tr = np.nanmax(np.stack([p.H - p.L, np.abs(p.H - pc), np.abs(p.L - pc)]), axis=0)
    return pd.DataFrame(tr).ewm(alpha=1 / 22, adjust=False, min_periods=22).mean().to_numpy()


@nb.njit(cache=True)
def sim(o, h, l, c, fr, atr, sig, exitx, side, mode, M, stop, cost):
    """mode 0 = chandelier(M), 1 = exit signal array exitx. Returns lists of entry, exit, return, stopped flag."""
    T = o.shape[0]; E = np.empty(T, np.int64); X = np.empty(T, np.int64); R = np.empty(T); S = np.empty(T, np.int64); n = 0
    i = 0
    while i < T - 1:
        if not sig[i] or np.isnan(o[i + 1]):
            i += 1; continue
        e = i + 1; px = o[e]; best = h[e] if side == 1 else l[e]; x = -1; xp = np.nan; stopped = 0
        stop_px = px * (1 - stop) if side == 1 else px * (1 + stop)
        for j in range(e, min(e + 120, T - 1)):
            if np.isnan(h[j]):
                continue
            if side == 1 and l[j] <= stop_px:
                xp = min(o[j], stop_px) if j > e else stop_px; x = j; stopped = 1; break
            if side == -1 and h[j] >= stop_px:
                xp = max(o[j], stop_px) if j > e else stop_px; x = j; stopped = 1; break
            best = max(best, h[j]) if side == 1 else min(best, l[j])
            if mode == 0:
                if not np.isnan(atr[j]) and ((side == 1 and c[j] < best - M * atr[j]) or (side == -1 and c[j] > best + M * atr[j])):
                    x = j + 1; break
            else:
                if exitx[j]:
                    x = j + 1; break
        if x == -1:
            x = min(e + 120, T - 1)
        if np.isnan(xp):
            xp = o[x]
        if np.isnan(xp):
            i = x; continue
        f = 0.0
        for j in range(e, x):
            f += fr[j]
        R[n] = side * (xp / px - 1.0) - side * f - 2 * cost / 1e4; E[n] = e; X[n] = x; S[n] = stopped; n += 1
        i = x
    return E[:n], X[:n], R[:n], S[:n]


def signals(fam, p, side, F=None):
    if fam == "A":
        fr7 = pd.DataFrame(p.FR).rolling(7, min_periods=7).mean().to_numpy()
        prev = np.vstack([np.full((1, fr7.shape[1]), np.nan), fr7[:-1]])
        return (fr7 >= F) & (prev < F) if side == -1 else (fr7 <= -F) & (prev > -F)
    ls = p.ld("count_long_short_ratio"); oi = p.ld("sum_open_interest_value")
    pct = pd.DataFrame(ls).rolling(90, min_periods=30).rank(pct=True).to_numpy()
    prev = np.vstack([np.full((1, pct.shape[1]), np.nan), pct[:-1]])
    oi7 = oi / np.vstack([np.full((7, oi.shape[1]), np.nan), oi[:-7]]) - 1
    return ((pct >= 0.95) & (prev < 0.95) & (oi7 > 0)) if side == -1 else ((pct <= 0.05) & (prev > 0.05) & (oi7 > 0))


def run(fam, p, side, F, conf, exit_name):
    sig = signals(fam, p, side, F)
    e10, e30 = ema(p.C, 10), ema(p.C, 30)
    if conf:
        sig = sig & ((e10 < e30) if side == -1 else (e10 > e30))
    up = (e10 > e30) & np.vstack([np.zeros((1, e10.shape[1]), bool), e10[:-1] <= e30[:-1]])
    dn = (e10 < e30) & np.vstack([np.zeros((1, e10.shape[1]), bool), e10[:-1] >= e30[:-1]])
    exitx = up if side == -1 else dn
    A = atr22(p); mode, M = (1, 0.0) if exit_name == "MAX" else (0, float(exit_name[2:]))
    rows = []
    for k in range(p.C.shape[1]):
        E, X, R, S = sim(p.O[:, k], p.H[:, k], p.L[:, k], p.C[:, k], p.FR[:, k], A[:, k], sig[:, k], exitx[:, k], side, mode, M, 0.20, 10.0)
        for e, x, r, s in zip(E, X, R, S):
            rows.append((p.syms[p.cols[k]], p.dates[e], x - e, r, s))
    return pd.DataFrame(rows, columns=["sym", "t", "hold", "ret", "stopped"])


def st(t, a, b):
    x = t[(t.t >= a) & (t.t < b)]; days = (pd.Timestamp(b) - pd.Timestamp(a)).days
    if len(x) < 5:
        return dict(n=len(x))
    g, l = x.ret[x.ret > 0].sum(), -x.ret[x.ret < 0].sum(); wk = x.groupby(x.t.dt.to_period("W")).ret.mean()
    return dict(n=len(x), per_day=round(len(x) / days, 2), mean_bp=round(x.ret.mean() * 1e4), pf=round(g / l, 2) if l else np.inf,
                t_w=round(wk.mean() / wk.std() * np.sqrt(len(wk)), 2) if len(wk) > 2 else np.nan, sum_pct=round(x.ret.sum() * 100),
                hold_d=round(x.hold.mean(), 1), stop_share=round(x.stopped.mean(), 2), worst_pct=round(x.ret.min() * 100))


if __name__ == "__main__":
    PA, PB = Panel("1d_long"), Panel("1d")
    hold = "--holdout" in sys.argv; only = [a[4:] for a in sys.argv if a.startswith("cfg=")]
    grid = [("A", s, F, c, x) for s, F, c, x in itertools.product((-1, 1), (0.001, 0.002), (0, 1), ("CH5", "CH7", "MAX"))] + \
           [("B", s, None, c, x) for s, c, x in itertools.product((-1, 1), (0, 1), ("CH5", "CH7", "MAX"))]
    out = []
    for fam, side, F, conf, ex in grid:
        name = f"{fam}_{'S' if side == -1 else 'L'}{'' if F is None else f'_F{F*100:.2f}%'}_{'conf' if conf else 'raw'}_{ex}"
        if only and name not in only:
            continue
        t = run(fam, PA if fam == "A" else PB, side, F, conf, ex)
        row = {"cfg": name}
        for seg, (a, b) in [("DEV", DEV[fam]), ("VALID-C", SEGS["VALID-C"])] + ([("HOLDOUT", SEGS["HOLDOUT"])] if hold else []):
            row.update({f"{seg}:{k}": v for k, v in st(t, a, b).items()})
        d, v = row, row
        row["PASS"] = bool(d.get("DEV:mean_bp", -1) > 0 and d.get("DEV:pf", 0) >= 1.2 and (d.get("DEV:t_w") or 0) >= 2.0
                           and d.get("DEV:per_day", 0) >= 0.2 and v.get("VALID-C:mean_bp", -1) > 0 and v.get("VALID-C:pf", 0) >= 1.2)
        out.append(row)
    df = pd.DataFrame(out); pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print(df.to_string(index=False)); df.to_csv("crowding_holdout.csv" if hold else "crowding_results.csv", index=False)
