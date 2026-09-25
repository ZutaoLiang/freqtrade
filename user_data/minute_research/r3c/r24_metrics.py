"""SKILL §4 metrics (relaxed B/E per user 2026-09-25) from a freqtrade backtest export."""
import glob, json, sys, zipfile
import numpy as np, pandas as pd


def load(d):
    z = zipfile.ZipFile(sorted(glob.glob(f"{d}/*.zip"))[-1])
    n = [x for x in z.namelist() if x.endswith(".json") and "config" not in x and "meta" not in x][0]
    s = json.loads(z.read(n))["strategy"]; t = pd.DataFrame(next(iter(s.values()))["trades"])
    t["t"] = pd.to_datetime(t.open_date, utc=True); t["base"] = t.pair.str.split("/").str[0]
    return t


def metrics(t, days):
    r = t.profit_ratio
    dm = r.groupby(t.t.dt.floor("D")).mean()
    mon = t.groupby(t.t.dt.strftime("%Y-%m")).profit_abs.sum()
    tot = t.profit_abs.sum()
    return {"n": len(t), "per_day": round(len(t) / days, 2), "trade_days": int(dm.size), "mean_bp": round(r.mean() * 1e4, 1),
            "pf": round(t.profit_abs[t.profit_abs > 0].sum() / -t.profit_abs[t.profit_abs < 0].sum(), 3),
            "day_t": round(dm.mean() / dm.std() * np.sqrt(len(dm)), 2), "months_pos": f"{(mon > 0).sum()}/{len(mon)}",
            "top_coin": f"{t.groupby('base').profit_abs.sum().idxmax()} {t.groupby('base').profit_abs.sum().max() / tot:.0%}" if tot > 0 else "n/a",
            "top_day": f"{t.groupby(t.t.dt.floor('D')).profit_abs.sum().max() / tot:.0%}" if tot > 0 else "n/a",
            "funding_bp": round((t.funding_fees / t.stake_amount).mean() * 1e4, 1), "monthly_usdt": mon.round(1).to_dict()}


if __name__ == "__main__":
    d, days = sys.argv[1], float(sys.argv[2])
    for k, v in metrics(load(d), days).items():
        print(f"{k}: {v}")
