"""Funding & positioning signals on daily bars, U162 (pre-registered in this docstring, 2026-09-25, before any run).

Panel binance_public/panels/1d (2025-01-01..2026-08-16); funding = true daily sum of raw settlements (crowding.daily_funding_sum);
OI = sum_open_interest_value, retail = count_long_short_ratio (accounts), top = sum_toptrader_long_short_ratio (positions),
last snapshot of each day. Signal on day d close -> entry d+1 open; one position per coin; cost 10 bp/side; funding charged.
Percentiles are per coin, trailing 90 days (min 30). fr7 = 7-day mean of daily funding.
  H1 short: fr7 >= 0.06%/day and OI 7-day change >= +30%
  H2 short: 7-day return <= -10% and fr7 >= +0.03%/day
  H3 long : 7-day return >= +15% and fr7 <= -0.03%/day
  H4 short: top-trader pct <= 0.10 and retail pct >= 0.90 ; H4 long: top pct >= 0.90 and retail pct <= 0.10
  H5 long : OI 3-day change <= -25% and 3-day return <= -15%
  REF B_S : retail pct crosses up through 0.95 while OI 7-day change > 0 -> short (crowding.py, unchanged)
Signals fire on the first day the condition becomes true (fresh). Exits (all with a 20% hard price stop, max 120 days):
  MAX opposite EMA10/30 cross | CH7 chandelier 7 x ATR22 | D10 fixed 10 days.
Segments DEV 2025-01-01..10-01, VALID-C 2025-12-01..2026-03-01, HOLDOUT 2026-03-01..08-17 (--holdout, survivors only).
Gates: DEV and VALID-C mean > 0 and PF >= 1.2; DEV week-clustered t >= 2; >= 0.4 trades/day in DEV across U162.
"""
import itertools, sys
import numba as nb
import numpy as np, pandas as pd
import crowding as Cw

SEGS = {"DEV": ("2025-01-01", "2025-10-01"), "VALID-C": ("2025-12-01", "2026-03-01"), "HOLDOUT": ("2026-03-01", "2026-08-17")}


@nb.njit(cache=True)
def sim(o, h, l, c, fr, atr, sig, exitx, side, mode, M, maxhold, stop, cost):
    T = o.shape[0]; E = np.empty(T, np.int64); X = np.empty(T, np.int64); R = np.empty(T); S = np.empty(T, np.int64); n = 0
    i = 0
    while i < T - 1:
        if not sig[i] or np.isnan(o[i + 1]):
            i += 1; continue
        e = i + 1; px = o[e]; best = h[e] if side == 1 else l[e]; x = -1; xp = np.nan; stopped = 0
        sp = px * (1 - stop) if side == 1 else px * (1 + stop)
        for j in range(e, min(e + maxhold, T - 1)):
            if np.isnan(h[j]):
                continue
            if side == 1 and l[j] <= sp:
                xp = min(o[j], sp) if j > e else sp; x = j; stopped = 1; break
            if side == -1 and h[j] >= sp:
                xp = max(o[j], sp) if j > e else sp; x = j; stopped = 1; break
            best = max(best, h[j]) if side == 1 else min(best, l[j])
            if mode == 0 and not np.isnan(atr[j]) and ((side == 1 and c[j] < best - M * atr[j]) or (side == -1 and c[j] > best + M * atr[j])):
                x = j + 1; break
            if mode == 1 and exitx[j]:
                x = j + 1; break
        if x == -1:
            x = min(e + maxhold, T - 1)
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


def fresh(cond):
    prev = np.vstack([np.zeros((1, cond.shape[1]), bool), cond[:-1]])
    return cond & ~prev


def signals(p):
    C = p.C; FR = p.FR
    oi = p.ld("sum_open_interest_value"); retail = p.ld("count_long_short_ratio"); top = p.ld("sum_toptrader_long_short_ratio")
    fr7 = pd.DataFrame(FR).rolling(7, min_periods=7).mean().to_numpy()
    lag = lambda x, k: np.vstack([np.full((k, x.shape[1]), np.nan), x[:-k]])
    r7, r3 = C / lag(C, 7) - 1, C / lag(C, 3) - 1
    oi7, oi3 = oi / lag(oi, 7) - 1, oi / lag(oi, 3) - 1
    pr = lambda x: pd.DataFrame(x).rolling(90, min_periods=30).rank(pct=True).to_numpy()
    rp, tp = pr(retail), pr(top)
    with np.errstate(invalid="ignore"):
        S = {"H1": (fresh((fr7 >= 0.0006) & (oi7 >= 0.30)), -1),
             "H2": (fresh((r7 <= -0.10) & (fr7 >= 0.0003)), -1),
             "H3": (fresh((r7 >= 0.15) & (fr7 <= -0.0003)), 1),
             "H4S": (fresh((tp <= 0.10) & (rp >= 0.90)), -1),
             "H4L": (fresh((tp >= 0.90) & (rp <= 0.10)), 1),
             "H5": (fresh((oi3 <= -0.25) & (r3 <= -0.15)), 1),
             "REF_B_S": (Cw.signals("B", p, -1), -1)}
    return S


def run(p, sig, side, ex, A, up, dn):
    mode, M, mh = {"MAX": (1, 0.0, 120), "CH7": (0, 7.0, 120), "D10": (2, 0.0, 10)}[ex]
    exitx = up if side == -1 else dn
    rows = []
    for k in range(p.C.shape[1]):
        E, X, R, S = sim(p.O[:, k], p.H[:, k], p.L[:, k], p.C[:, k], p.FR[:, k], A[:, k], sig[:, k], exitx[:, k], side, mode, M, mh, 0.20, 10.0)
        rows += [(p.syms[p.cols[k]], p.dates[e], x - e, r, s) for e, x, r, s in zip(E, X, R, S)]
    return pd.DataFrame(rows, columns=["sym", "t", "hold", "ret", "stopped"])


if __name__ == "__main__":
    segs = ["DEV", "VALID-C"] + (["HOLDOUT"] if "--holdout" in sys.argv else [])
    only = [a[4:] for a in sys.argv if a.startswith("cfg=")]
    p = Cw.Panel("1d"); A = Cw.atr22(p); e10, e30 = Cw.ema(p.C, 10), Cw.ema(p.C, 30)
    up = (e10 > e30) & np.vstack([np.zeros((1, e10.shape[1]), bool), e10[:-1] <= e30[:-1]])
    dn = (e10 < e30) & np.vstack([np.zeros((1, e10.shape[1]), bool), e10[:-1] >= e30[:-1]])
    S = signals(p); out = []
    for (name, (sig, side)), ex in itertools.product(S.items(), ("MAX", "CH7", "D10")):
        cfg = f"{name}_{ex}"
        if only and cfg not in only:
            continue
        t = run(p, sig, side, ex, A, up, dn); row = {"cfg": cfg, "side": "L" if side == 1 else "S"}
        for s in segs:
            row.update({f"{s}:{k}": v for k, v in Cw.st(t, *SEGS[s]).items() if k in ("n", "per_day", "mean_bp", "pf", "t_w", "sum_pct", "hold_d", "stop_share")})
        row["PASS"] = bool(row.get("DEV:mean_bp", -1) > 0 and row.get("DEV:pf", 0) >= 1.2 and (row.get("DEV:t_w") or 0) >= 2.0
                           and row.get("DEV:per_day", 0) >= 0.4 and row.get("VALID-C:mean_bp", -1) > 0 and row.get("VALID-C:pf", 0) >= 1.2)
        out.append(row)
    df = pd.DataFrame(out); pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False)); df.to_csv("posfund_holdout.csv" if "--holdout" in sys.argv else "posfund_results.csv", index=False)
