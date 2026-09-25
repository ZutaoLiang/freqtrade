"""r5 R47 fresh-stage gate v2: funding72 + premium agree, 2023-24, binance-hist freqtrade feathers."""
import sys, os
import numpy as np
import pandas as pd
import pyarrow.feather as ft

sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r5")

HIST = "/root/freqtrade/user_data/data/binance-hist/futures"
rows = []

def cost_bps(base):
    return 7.5 if base in ("BTC", "ETH") else 10.0

def sim_stage(stage, theta=0.0012, hold=96, cum_w=72, rank_lo=31, rank_hi=90, theta_lo=-0.0012):
    i0 = pd.Timestamp(stage[0], tz="UTC"); i1 = pd.Timestamp(stage[1], tz="UTC")
    syms = sorted(set(f.split("-")[0] for f in os.listdir(HIST) if f.endswith("-1h-futures.feather")))
    qv_med = {}
    for s in syms:
        try:
            d = ft.read_feather(f"{HIST}/{s}-1h-futures.feather", columns=["date", "close", "volume"])
        except Exception:
            continue
        d = d[(d.date >= i0) & (d.date < i1)]
        if len(d) < 24 * 60: continue
        qv_med[s] = (d.volume * d.close).median()
    ranked = sorted(qv_med, key=lambda s: -qv_med[s])
    uni = ranked[max(rank_lo - 1, 0):rank_hi]
    trades = []
    for s in uni:
        base = s.split("_")[0]
        try:
            d = ft.read_feather(f"{HIST}/{s}-1h-futures.feather", columns=["date", "open", "close", "volume"])
            mk = ft.read_feather(f"{HIST}/{s}-1h-mark.feather", columns=["date", "close"]).rename(columns={"close": "mark"})
            fu = ft.read_feather(f"{HIST}/{s}-1h-funding_rate.feather", columns=["date", "open"]).rename(columns={"open": "rate"})
        except Exception:
            continue
        d = d[(d.date >= i0) & (d.date < i1)].reset_index(drop=True)
        if len(d) < cum_w + 120: continue
        d = d.merge(mk, on="date", how="left").merge(fu, on="date", how="left")
        if d.mark.notna().sum() < len(d) * 0.8: continue
        fr = np.where(np.abs(d.rate.fillna(0)) > 1e-12, d.rate.fillna(0), 0.0)
        cum = pd.Series(fr).rolling(cum_w, min_periods=cum_w // 6).sum().to_numpy()
        o = d.open.to_numpy(np.float64); c = d.close.to_numpy(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            prem = np.where((d.mark > 0) & (c > 0), c / d.mark - 1.0, np.nan)
        nn = len(d)
        i = cum_w + 9; busy = -1
        while i < nn - hold - 1:
            if i % 8 != 1 or i <= busy:
                i += 1; continue
            f, pr = cum[i - 1], prem[i - 1]
            side = 0
            if np.isfinite(f) and np.isfinite(pr):
                if f >= theta and pr > 0: side = -1
                elif f <= theta_lo and pr < 0: side = 1
            if side == 0:
                i += 1; continue
            e = i + 1
            x = min(e + hold - 1, nn - 1)
            ret = side * (c[x] / o[e] - 1.0) - 2 * cost_bps(base) / 1e4
            trades.append({"ret": ret, "t": d.date.iloc[e], "sym": s})
            busy = x; i = x + 1
    if not trades:
        return {"n": 0}
    df = pd.DataFrame(trades)
    df["day"] = df.t.dt.floor("D")
    n = len(df); r = df.ret
    g, b = r[r > 0].sum(), -r[r < 0].sum()
    dm = df.groupby("day").ret.mean()
    t = dm.mean() / dm.std() * np.sqrt(len(dm)) if len(dm) > 2 and dm.std() > 0 else float("nan")
    tot = r.sum()
    return {"n": int(n), "per_day": round(n / max((df.day.max() - df.day.min()).days, 1), 2),
            "mean_bp": round(r.mean() * 1e4, 2), "win": round((r > 0).mean(), 3),
            "pf": round(g / b, 3) if b > 0 else 99.0, "days": int(len(dm)),
            "day_mean_bp": round(dm.mean() * 1e4, 2), "day_t": round(t, 2),
            "coin_share": round(df.groupby("sym").ret.sum().abs().max() / abs(tot), 2) if tot else 0}

if __name__ == "__main__":
    for stage in (("2023-01-01", "2024-06-01"), ("2024-06-01", "2025-01-01")):
        for hold in (72, 96):
            res = sim_stage(stage, hold=hold)
            rows.append({"stage": f"{stage[0]}..{stage[1]}", "hold": hold, **res})
    out = pd.DataFrame(rows)
    out.to_csv("/root/freqtrade/user_data/minute_research/r5/r47_gate.csv", index=False)
    print(out.to_string())
