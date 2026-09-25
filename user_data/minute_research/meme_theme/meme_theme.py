"""MEME-theme copy of privacy_theme.py (same rules; theme list and leader replaced before any run). Original:
Privacy-theme study (pre-registered in this docstring, 2026-09-25, before any run).

Theme P = ZEC, XMR, DASH, ZEN, SCRT, ROSE (Binance USDT perps, 2025-01..2026-08). Cost 10 bp/side, real funding
(8h settlements summed while held), absolute returns. Scheme C: DEV 2025-01-01..10-01, VALID-C 2025-12-01..2026-03-01,
HOLDOUT 2026-03-01..09-01 read once (--holdout) for at most two survivors per direction. Oct-Nov 2025 excluded.

Direction 1 — leader follow (1h bars): ZEC return over the last W hours >= X -> long each follower (XMR, DASH, ZEN, SCRT,
ROSE) at the next open; mirror: ZEC <= -X -> short followers. Optional lag filter: only followers whose own W-hour return is
below half of ZEC's (above half for the short mirror). Hold H hours, one position per follower.
Grid W {1,4} x X {4%,8%} x H {4,24} x lag {no,yes} x side {long,short} = 32.
Direction 2 — theme rotation (daily bars): every 7 days rank P by trailing L-day return; long the top N (form "long") or long
top N and short bottom N (form "ls"), equal weight, hold 7 days; all 7 weekday offsets evaluated and averaged.
Grid L {7,14,28} x N {1,2} x form {long, ls} = 12.
Gates: DEV and VALID-C net mean > 0 and PF >= 1.2; DEV t >= 2 (dir 1: day-clustered over trades; dir 2: over weekly book
returns, median across offsets).
"""
import itertools, sys
import numpy as np, pandas as pd

R = "/root/freqtrade/user_data/data/binance_public"
P = ["DOGE", "1000PEPE", "FARTCOIN", "TRUMP", "WIF", "PENGU", "PNUT", "1000SHIB", "1000BONK", "NEIRO", "POPCAT", "MOODENG", "SPX", "1000FLOKI", "VINE", "BOME", "PEOPLE", "TURBO", "CHILLGUY", "DOGS", "MELANIA", "NEIROETH", "BRETT", "1000000MOG", "1MBABYDOGE", "MEW", "1000CAT"]  # MEME theme, pre-registered; leader = DOGE (highest TRAIN liquidity)
SEG = {"DEV": ("2025-01-01", "2025-10-01"), "VALID-C": ("2025-12-01", "2026-03-01"), "HOLDOUT": ("2026-03-01", "2026-09-01")}
COST = 10e-4


def bars(b, rule):
    k = pd.read_parquet(f"{R}/klines_1m/{b}USDT.parquet", columns=["date", "open", "close"]).set_index("date")
    k = k[k.index < "2026-09-01"]
    return k.resample(rule).agg({"open": "first", "close": "last"})


def funding(b, index):
    f = pd.read_parquet(f"{R}/funding/{b}USDT.parquet"); f["date"] = f.date.dt.round("h")
    f = f.drop_duplicates("date", keep="last").set_index("date").funding_rate
    return f.groupby(f.index.floor(index.freq)).sum().reindex(index).fillna(0.0)


def st(r, t, a, b):
    m = (t >= a) & (t < b); r = r[m]; t = t[m]
    if len(r) < 5:
        return dict(n=int(len(r)))
    g, l = r[r > 0].sum(), -r[r < 0].sum(); dm = pd.Series(r).groupby(pd.DatetimeIndex(t).floor("D")).sum()
    return dict(n=int(len(r)), mean_bp=round(r.mean() * 1e4, 1), pf=round(g / l, 2) if l else np.inf,
                t=round(dm.mean() / dm.std() * np.sqrt(len(dm)), 2) if dm.std() > 0 else np.nan, sum_pct=round(r.sum() * 100, 1))


def direction1(segs):
    H = {b: bars(b, "1h") for b in P}; idx = H["DOGE"].index
    O = pd.DataFrame({b: H[b].open for b in P}).reindex(idx); C = pd.DataFrame({b: H[b].close for b in P}).reindex(idx)
    FR = pd.DataFrame({b: funding(b, idx) for b in P})
    rows = []
    for W, X, Hd, lag, side in itertools.product((1, 4), (0.04, 0.08), (4, 24), (0, 1), (1, -1)):
        zr = C.DOGE / C.DOGE.shift(W) - 1
        trig = (zr >= X) if side == 1 else (zr <= -X)
        R_, T_ = [], []
        for b in P[1:]:
            fr = C[b] / C[b].shift(W) - 1
            sig = trig & ((fr < zr / 2) if (lag and side == 1) else (fr > zr / 2) if lag else True)
            o, f = O[b].to_numpy(), FR[b].to_numpy(); s = sig.to_numpy(); busy = -1
            for i in np.where(s)[0]:
                e, x = i + 1, i + 1 + Hd
                if i < busy or x >= len(o) or np.isnan(o[e]) or np.isnan(o[x]):
                    continue
                R_.append(side * (o[x] / o[e] - 1) - side * f[e + 1:x + 1].sum() - 2 * COST); T_.append(idx[e]); busy = x
        r, t = np.array(R_), pd.DatetimeIndex(T_)
        row = dict(W=W, X=X, H=Hd, lag=lag, side="L" if side == 1 else "S")
        for sname in segs:
            row.update({f"{sname}:{k}": v for k, v in st(r, t, *SEG[sname]).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def direction2(segs):
    D = {b: bars(b, "1D") for b in P}; idx = D["DOGE"].index
    O = pd.DataFrame({b: D[b].open for b in P}).reindex(idx); C = pd.DataFrame({b: D[b].close for b in P}).reindex(idx)
    FR = pd.DataFrame({b: funding(b, idx) for b in P})
    rows = []
    for L, N, form in itertools.product((7, 14, 28), (1, 2), ("long", "ls")):
        mom = C / C.shift(L) - 1
        res = {s: [] for s in segs}
        for off in range(7):
            book = []
            for d in range(L + off, len(idx) - 8, 7):
                m = mom.iloc[d].dropna()
                if len(m) < 8:
                    continue
                e, x = d + 1, d + 8
                ret = lambda b, sd: sd * (O[b].iloc[x] / O[b].iloc[e] - 1) - sd * FR[b].iloc[e:x].sum() - 2 * COST
                legs = [ret(b, 1) for b in m.nlargest(N).index] + ([ret(b, -1) for b in m.nsmallest(N).index] if form == "ls" else [])
                book.append((idx[e], np.nanmean(legs)))
            bk = pd.Series(dict(book))
            for s in segs:
                a, b = SEG[s]; x = bk[(bk.index >= a) & (bk.index < b)]
                if len(x) > 3:
                    g, l = x[x > 0].sum(), -x[x < 0].sum()
                    res[s].append((x.mean(), x.mean() / x.std() * np.sqrt(len(x)), g / l if l else np.inf, x.sum(), len(x)))
        row = dict(L=L, N=N, form=form)
        for s in segs:
            v = np.array(res[s])
            row.update({f"{s}:weekly_mean_bp": round(v[:, 0].mean() * 1e4, 1), f"{s}:t_med": round(float(np.median(v[:, 1])), 2),
                        f"{s}:pf_med": round(float(np.median(v[:, 2])), 2), f"{s}:sum_pct": round(v[:, 3].mean() * 100, 1),
                        f"{s}:offsets_pos": int((v[:, 0] > 0).sum()), f"{s}:weeks": int(v[:, 4].mean())})
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    segs = ["DEV", "VALID-C"] + (["HOLDOUT"] if "--holdout" in sys.argv else [])
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    d1 = direction1(segs)
    d1["PASS"] = (d1["DEV:mean_bp"] > 0) & (d1["DEV:pf"] >= 1.2) & (d1["DEV:t"] >= 2) & (d1["VALID-C:mean_bp"] > 0) & (d1["VALID-C:pf"] >= 1.2)
    print("== Direction 1: leader follow"); print(d1.to_string(index=False))
    d2 = direction2(segs)
    d2["PASS"] = (d2["DEV:weekly_mean_bp"] > 0) & (d2["DEV:pf_med"] >= 1.2) & (d2["DEV:t_med"] >= 2) & (d2["VALID-C:weekly_mean_bp"] > 0) & (d2["VALID-C:pf_med"] >= 1.2)
    print("\n== Direction 2: theme rotation"); print(d2.to_string(index=False))
    d1.to_csv("dir1.csv", index=False); d2.to_csv("dir2.csv", index=False)
