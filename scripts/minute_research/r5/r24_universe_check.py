"""R24 on the union panel (their U162 and my U160), fixed 2026 funding. Lean: only open prices + settled funding."""
import json, os, sys
sys.path.insert(0, "scripts/minute_research/r5")
os.environ.setdefault("P7_OUT", "user_data/minute_research/r5/p7_panel_union.npz")
os.environ.setdefault("P7_UNIV", "user_data/minute_research/r5/u_union.json")
import numpy as np, pandas as pd
import p7_panel as P, recheck_schemeC as R
D = P.build()
o = D["open"].astype(np.float64); fr = D["fr"].astype(np.float64); syms = list(D["syms"])
dates = pd.to_datetime(D["dates"], unit="us", utc=True); T, N = o.shape
cost = np.array([7.5e-4 if s in ("BTC", "ETH") else 10e-4 for s in syms])
chain = np.zeros((T, N), bool)
for j in range(N):
    idx = np.where(~np.isnan(fr[:, j]))[0]
    if len(idx) < 3: continue
    f = fr[idx, j]; ok = (f[2:] >= 3e-4) & (f[1:-1] >= 3e-4) & (f[:-2] >= 3e-4)
    chain[idx[2:][ok], j] = True
m26 = dates >= "2026-01-01"
print("coins with 2026 funding rows:", int(((~np.isnan(fr[m26])).sum(axis=0) > 100).sum()), "/", N)
for label, uf in (("their U162", "user_data/minute_research/r5/u162_theirs.json"), ("my U160", "user_data/minute_research/r5/u160.json")):
    keep = set(json.load(open(uf)))
    sig = chain.copy(); sig[:, [k for k, s in enumerate(syms) if s not in keep]] = False
    e, j, sd, r, fu = P.sim(o, sig, np.full((T, N), -1, np.int64), 7, cost, fr, True)
    t = pd.DataFrame({"t": dates[e], "ret": r, "side": sd, "pair": [syms[k] for k in j], "cost": cost[j]}); t["seg"] = R.segC(t.t)
    print(f"\n== R24 on {label}")
    for s in ("TRAIN", "VALID", "HOLDOUT"):
        x = t[t.seg == s]; st = R.stats(x)
        print(f"  {s:7s} n{st['n']} {st['net_bp']}bp PF{st['pf']} t{st['t']} mon+{st['mon+']} top_pair{st['top_pair']}  top3:", (x.groupby('pair').ret.sum().sort_values().tail(3) * 100).round(0).to_dict())
    vh = t[t.seg.isin(["VALID", "HOLDOUT"])]; dm = vh.groupby(vh.t.dt.floor("D")).ret.sum()
    y = vh[vh.pair != "ARC"]; dm2 = y.groupby(y.t.dt.floor("D")).ret.sum()
    print(f"  V+HO t {dm.mean()/dm.std()*np.sqrt(len(dm)):.2f} | ex-ARC V+HO n{len(y)} {y.ret.mean()*1e4:.1f}bp t {dm2.mean()/dm2.std()*np.sqrt(len(dm2)):.2f}")
    t.to_parquet(f"user_data/minute_research/r5/r24_{label.split()[1]}_trades.parquet")
