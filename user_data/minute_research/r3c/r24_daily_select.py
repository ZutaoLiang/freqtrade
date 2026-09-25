"""R24 + causal daily coin pre-selection (pre-registered in this docstring).
At each UTC day D 00:00 rank U162 by
  A: mean funding over settlements in [D-7d, D)        B: share of settlements >= 0.03% in [D-3d, D)
keep the top N (N in 5/10/20); R24 trades entering on day D are kept only if their coin is selected.
Trades: fixed-stake R24 backtests (no compounding interaction). Choose on TRAIN, confirm on VALID-C, report HOLDOUT."""
import itertools
import numpy as np, pandas as pd
import r24_metrics as M

B = "/root/freqtrade/user_data/data/binance_public/funding"
U = pd.read_csv("universe_rank.csv"); U162 = U[U.med_qv >= 1e7].base.tolist()
F = []
for b in U162:
    f = pd.read_parquet(f"{B}/{b}USDT.parquet", columns=["date", "funding_rate"]); f["date"] = f.date.dt.round("h"); f["base"] = b
    F.append(f)
F = pd.concat(F).drop_duplicates(["base", "date"])
days = pd.date_range("2025-01-01", "2026-09-01", freq="D", tz="UTC")

def scores(kind):
    out = {}
    for D in days:
        if kind == "A":
            w = F[(F.date >= D - pd.Timedelta(days=7)) & (F.date < D)]; s = w.groupby("base").funding_rate.mean()
        else:
            w = F[(F.date >= D - pd.Timedelta(days=3)) & (F.date < D)]; s = w.groupby("base").funding_rate.apply(lambda x: (x >= 3e-4).mean())
        out[D] = s
    return out

SC = {k: scores(k) for k in ("A", "B")}
SEG = {"TRAIN": ("../r3/bt_r24_train", 273), "VALID-C": ("bt_r24_validC", 90), "HOLDOUT": ("../r3/bt_r24_holdout", 184)}
T = {s: M.load(p) for s, (p, _) in SEG.items()}

def stats(t, d):
    if len(t) < 3: return dict(n=len(t))
    g, l = t.profit_abs[t.profit_abs > 0].sum(), -t.profit_abs[t.profit_abs < 0].sum()
    dm = t.groupby(t.t.dt.floor("D")).profit_ratio.mean()
    return dict(n=len(t), per_day=round(len(t) / d, 2), usdt=round(t.profit_abs.sum(), 1), pf=round(g / l, 2) if l else np.inf,
                t=round(dm.mean() / dm.std() * np.sqrt(len(dm)), 2) if len(dm) > 2 else np.nan)

rows = []
for s, t in T.items():
    d = SEG[s][1]; full = stats(t, d)
    rows.append(dict(seg=s, rule="full U162 (R24 as is)", **full, kept_profit="100%", ARC_win="-", SWARMS_win="-"))
    for kind, N in itertools.product(("A", "B"), (5, 10, 20)):
        day = t.t.dt.floor("D")
        sel = [b in set(SC[kind][D].nlargest(N).index) for b, D in zip(t.base, day)]
        k = t[np.array(sel)]
        st = stats(k, d)
        cov = lambda c: f"{int(((k.base == c) & (k.profit_abs > 0)).sum())}/{int(((t.base == c) & (t.profit_abs > 0)).sum())}"
        rows.append(dict(seg=s, rule=f"{kind} top{N}", **st, kept_profit=f"{k.profit_abs.sum() / t.profit_abs.sum():.0%}", ARC_win=cov("ARC"), SWARMS_win=cov("SWARMS")))
print(pd.DataFrame(rows).to_string(index=False))
