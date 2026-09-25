"""R12 pre-settlement drift, flat before T (pre-registered). TRAIN."""
import itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import r3lib as L
from r11_presettle import U162, U60

def one(base):
    k = pd.read_parquet(f"{L.ROOT}/klines_1m/{base}USDT.parquet", columns=["date", "open"])
    k = k[k.date < L.TRAIN[1]]
    op = pd.Series(k.open.to_numpy(), index=pd.DatetimeIndex(k.date))
    f = pd.read_parquet(f"{L.ROOT}/funding/{base}USDT.parquet"); f["date"] = f.date.dt.round("min")
    f = f.drop_duplicates("date").set_index("date").sort_index(); f["prev"] = f.funding_rate.shift(1)
    f = f[f.index < L.TRAIN[1]].dropna(subset=["prev"]); f = f[f.prev.abs() >= 0.002]
    c = (10.0 if base in U60 else 15.0) / 1e4
    rows = []
    for T, r in f.iterrows():
        side = 1 if r.prev < 0 else -1
        for H, X in itertools.product((15, 30, 60), (1, 3)):
            a, b = T - pd.Timedelta(minutes=H), T - pd.Timedelta(minutes=X)
            if a in op.index and b in op.index:
                rows.append((base, T, H, X, abs(r.prev), side * (op[b] / op[a] - 1) - 2 * c))
    return pd.DataFrame(rows, columns=["base", "T", "H", "X", "absprev", "net"])

if __name__ == "__main__":
    with ProcessPoolExecutor(16) as ex:
        d = pd.concat(list(ex.map(one, U162)))
    out = []
    for F, H, X in itertools.product((0.002, 0.003, 0.005), (15, 30, 60), (1, 3)):
        s = d[(d.absprev >= F) & (d.H == H) & (d.X == X)]
        dm = s.groupby(s["T"].dt.floor("D")).net.mean(); g, b = s.net[s.net > 0].sum(), -s.net[s.net < 0].sum()
        out.append({"F": F, "H": H, "X": X, "n": len(s), "net_bp": s.net.mean() * 1e4, "pf": g / b,
                    "day_t": dm.mean() / dm.std() * np.sqrt(len(dm)), "per_day": len(s) / 273})
    print(pd.DataFrame(out).round(2).to_string())
