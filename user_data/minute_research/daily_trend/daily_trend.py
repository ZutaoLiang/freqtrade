"""Daily volume-breakout + MA-cross entry, chandelier exit (pre-registered in this docstring, 2026-09-25).

Universe U162 (static; ZEC reported separately). Panel binance_public/panels/1d_long (2022-11-01..2026-08-16).
Long: volume_t > V x SMA20(volume, previous 20 days) and EMA(fast) crossed above EMA(slow) within the last 3 days;
entry next open. Exit: close < highest high since entry - M x ATR22 (Wilder), exit next open; max hold 60 days.
Short: mirror (death cross, lowest low + M x ATR22). Grid (fast,slow) {(10,30),(20,50)} x V {1.5,2.0} x M {3,4} per side.
Cost 10 bp/side; real daily funding (panel funding_rate = sum of the day's settlements; longs pay positive funding).
Segments: DEV 2022-11-01..2025-10-01, VALID-C 2025-12-01..2026-03-01, HOLDOUT 2026-03-01..2026-08-16 (read once, only for
survivors, with --holdout). Metrics per segment: trades, trades/day, mean, PF, t clustered by entry week, market-adjusted mean
(trade minus the equal-weight U162 buy-and-hold return over the same holding days, signed by side), share of weeks positive.
Gates: DEV and VALID-C mean > 0, PF >= 1.2, adjusted mean > 0; DEV week-clustered t >= 2; >= 0.4 trades/day portfolio-wide.
"""
import itertools, json, sys
import numba as nb
import numpy as np, pandas as pd

P = "/root/freqtrade/user_data/data/binance_public/panels/1d_long"
m = json.load(open(f"{P}/meta.json"))
SYMS = m["symbols"]; DATES = pd.date_range(m["start"], m["end"], freq="D")
U = pd.read_csv("/root/freqtrade/user_data/minute_research/r3/universe_rank.csv")
U162 = [SYMS.index(b + "USDT") for b in U[U.med_qv >= 1e7].base if b + "USDT" in SYMS]
ld = lambda f: np.array(np.load(f"{P}/{f}.npy", mmap_mode="r"), dtype=np.float64)
O, H, L, C, V, FR = (ld(f) for f in ("open", "high", "low", "close", "volume", "funding_rate"))
FR = np.nan_to_num(FR)
SEGS = {"DEV": ("2022-11-01", "2025-10-01"), "VALID-C": ("2025-12-01", "2026-03-01"), "HOLDOUT": ("2026-03-01", "2026-08-17")}


def ema(x, n):
    return pd.DataFrame(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def atr22(h, l, c):
    pc = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    tr = np.nanmax(np.stack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return pd.DataFrame(tr).ewm(alpha=1 / 22, adjust=False, min_periods=22).mean().to_numpy()


@nb.njit(cache=True)
def sim(o, h, l, c, fr, atr, sig, side, M, cost):
    T = o.shape[0]
    ent, ext, rets = [], [], []
    i = 0
    while i < T - 1:
        if not sig[i]:
            i += 1; continue
        e = i + 1
        if np.isnan(o[e]):
            i += 1; continue
        ext_i = -1; best = h[e] if side == 1 else l[e]
        for j in range(e, min(e + 60, T - 1)):
            best = max(best, h[j]) if side == 1 else min(best, l[j])
            if np.isnan(atr[j]) or np.isnan(c[j]):
                continue
            if (side == 1 and c[j] < best - M * atr[j]) or (side == -1 and c[j] > best + M * atr[j]):
                ext_i = j + 1; break
        if ext_i == -1:
            ext_i = min(e + 60, T - 1)
        if np.isnan(o[ext_i]):
            i = ext_i; continue
        f = 0.0
        for j in range(e, ext_i):
            f += fr[j]
        rets.append(side * (o[ext_i] / o[e] - 1.0) - side * f - 2 * cost / 1e4)
        ent.append(e); ext.append(ext_i)
        i = ext_i
    return ent, ext, rets


def market_path():
    """equal-weight U162 buy-and-hold cumulative index (open to open)."""
    r = O[1:, U162] / O[:-1, U162] - 1
    daily = np.nanmean(np.where(np.isfinite(r), r, np.nan), axis=1)
    return np.concatenate([[1.0], np.cumprod(1 + np.nan_to_num(daily))])


def run(fast, slow, Vm, M, side, cols):
    ef, es = ema(C, fast), ema(C, slow)
    cross = (ef > es) & (np.vstack([np.zeros((1, C.shape[1]), bool), ef[:-1] <= es[:-1]])) if side == 1 else \
            (ef < es) & (np.vstack([np.zeros((1, C.shape[1]), bool), ef[:-1] >= es[:-1]]))
    recent = pd.DataFrame(cross).rolling(3, min_periods=1).max().to_numpy() > 0
    vma = pd.DataFrame(V).rolling(20, min_periods=20).mean().shift(1).to_numpy()
    sig = recent & (V > Vm * vma)
    A = atr22(H, L, C)
    mk = market_path()
    rows = []
    for j in cols:
        ent, ext, r = sim(O[:, j], H[:, j], L[:, j], C[:, j], FR[:, j], A[:, j], sig[:, j], side, M, 10.0)
        for e, x, rr in zip(ent, ext, r):
            rows.append((SYMS[j], DATES[e], e, x, rr, rr - side * (mk[x] / mk[e] - 1)))
    return pd.DataFrame(rows, columns=["sym", "t", "e", "x", "ret", "adj"])


def seg_stats(t, a, b):
    x = t[(t.t >= a) & (t.t < b)]
    days = (pd.Timestamp(b) - pd.Timestamp(a)).days
    if len(x) < 5:
        return dict(n=len(x))
    g, l = x.ret[x.ret > 0].sum(), -x.ret[x.ret < 0].sum()
    wk = x.groupby(x.t.dt.to_period("W")).ret.mean()
    return dict(n=len(x), per_day=round(len(x) / days, 2), mean_bp=round(x.ret.mean() * 1e4, 1), pf=round(g / l, 2) if l else np.inf,
                t_week=round(wk.mean() / wk.std() * np.sqrt(len(wk)), 2), adj_bp=round(x.adj.mean() * 1e4, 1),
                weeks_pos=round((wk > 0).mean(), 2), hold_d=round((x.x - x.e).mean(), 1))


if __name__ == "__main__":
    segs = ["DEV", "VALID-C"] + (["HOLDOUT"] if "--holdout" in sys.argv else [])
    only = [a for a in sys.argv[1:] if a.startswith("cfg=")]
    out = []
    for (fast, slow), Vm, M, side in itertools.product([(10, 30), (20, 50)], [1.5, 2.0], [3, 4], [1, -1]):
        name = f"{'L' if side == 1 else 'S'}_{fast}/{slow}_V{Vm}_M{M}"
        if only and f"cfg={name}" not in only:
            continue
        for label, cols in (("U162", U162), ("ZEC", [SYMS.index("ZECUSDT")])):
            t = run(fast, slow, Vm, M, side, cols)
            row = {"cfg": name, "univ": label}
            for s in segs:
                for k, v in seg_stats(t, *SEGS[s]).items():
                    row[f"{s}:{k}"] = v
            out.append(row)
    df = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    df.to_csv("daily_trend_results.csv" if "--holdout" not in sys.argv else "daily_trend_holdout.csv", index=False)
