"""R18 equity-perp off-hours overreaction (pre-registered). SEG env: TRAIN|VALID|HOLD."""
import itertools, os
import numpy as np, pandas as pd
import r3lib as L

SYMS = "NVDA TSLA MSTR AAPL COIN HOOD AMZN GOOGL META MSFT CRCL INTC PLTR QQQ SPY AMD BABA TSM".split()
SEGS = {"TRAIN": ("2026-01-01", "2026-06-01"), "VALID": ("2026-06-01", "2026-08-01"), "HOLD": ("2026-08-01", "2026-09-01")}
SEG = SEGS[os.environ.get("SEG", "TRAIN")]
CELLS = list(itertools.product([0.01, 0.02], ["60m", "close"], ["fade", "follow"]))


def events(base):
    d = pd.read_parquet(f"{L.ROOT}/klines_1m/{base}USDT.parquet", columns=["date", "open", "close"])
    d = d[(d.date >= SEG[0]) & (d.date < SEG[1])].reset_index(drop=True)
    et = d.date.dt.tz_convert("America/New_York")
    d["day"] = et.dt.date; d["hm"] = et.dt.hour * 100 + et.dt.minute; d["wd"] = et.dt.weekday
    rows = []
    days = sorted(d.day[(d.wd < 5)].unique())
    closes = {}
    for day in days:
        g = d[d.day == day]
        c16 = g[g.hm == 1559]
        o930 = g[g.hm == 930]; p929 = g[g.hm == 929]
        if len(o930) and len(p929) and closes:
            prev_close = closes[max(closes)]
            M = p929.close.iloc[0] / prev_close - 1
            i = o930.index[0]
            e = d.open[i]
            x60 = d.open.get(i + 60, np.nan)
            xc = g[g.hm == 1559].close.iloc[0] if len(c16) else np.nan
            rows.append((base, day, M, x60 / e - 1, xc / e - 1))
        if len(c16):
            closes[day] = c16.close.iloc[0]
    return pd.DataFrame(rows, columns=["base", "day", "M", "r60", "rclose"])


if __name__ == "__main__":
    E = pd.concat([events(b) for b in SYMS])
    E = E.dropna()
    ndays = E.day.nunique()
    out = []
    for thr, ex, arm in CELLS:
        s = E[E.M.abs() >= thr]
        r = (np.sign(s.M) * (-1 if arm == "fade" else 1)) * (s.r60 if ex == "60m" else s.rclose) - 20e-4
        dm = r.groupby(s.day.to_numpy()).mean()
        g, b = r[r > 0].sum(), -r[r < 0].sum()
        out.append({"thr": thr, "exit": ex, "arm": arm, "n": len(s), "net_bp": r.mean() * 1e4, "med_bp": r.median() * 1e4,
                    "pf": g / b, "days": len(dm), "day_t": dm.mean() / dm.std() * np.sqrt(len(dm)), "per_day": len(s) / ndays})
    print("trading days", ndays, "instruments", E.base.nunique())
    print(pd.DataFrame(out).round(2).to_string())
