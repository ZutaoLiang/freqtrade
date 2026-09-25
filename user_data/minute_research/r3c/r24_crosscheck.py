"""Why does another machine get 1452 R24 TRAIN trades (PF 0.90) vs 244-338 here? Test signal-timing hypotheses."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, "/root/freqtrade/.claude/skills/binance-minute-strategy-research/scripts")
import harness as H

B = "/root/freqtrade/user_data/data/binance_public"
U = pd.read_csv("universe_rank.csv"); U162 = U[U.med_qv >= 1e7].base.tolist()

def one(args):
    base, mode = args
    k = pd.read_parquet(f"{B}/klines_1m/{base}USDT.parquet", columns=["date", "open", "high", "low", "close", "volume"])
    k = k[(k.date >= "2025-01-01") & (k.date < "2025-10-01")]
    h = k.set_index("date").resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    f = pd.read_parquet(f"{B}/funding/{base}USDT.parquet"); f["date"] = f.date.dt.round("h")
    f = f.drop_duplicates("date", keep="last").set_index("date").funding_rate.sort_index()
    chain = (f >= 3e-4) & (f.shift(1) >= 3e-4) & (f.shift(2) >= 3e-4)
    if mode == "settlement":                       # signal only on the hour of the 3rd settlement (this repo's R24)
        sig = chain.reindex(h.index).fillna(False)
    else:                                          # 'hourly': the chain state forward-filled, re-checked every hour
        sig = chain.reindex(h.index, method="ffill").fillna(False)
    d = {c: h[c].to_numpy() for c in ("open", "high", "low", "close", "volume")}; d["date"] = h.index.to_series()
    idx = np.where(sig.to_numpy())[0]
    tr = H.run(d, idx, -1, hold=8, cost_bps=10.0)
    iv = f.index.to_series().diff().dt.total_seconds().div(3600).reindex(h.index, method="ffill").to_numpy()
    tr["interval"] = iv[tr.entry.to_numpy()] if len(tr) else []
    tr["base"] = base
    return tr

if __name__ == "__main__":
    for mode in ("settlement", "hourly"):
        with ProcessPoolExecutor(16) as ex:
            t = pd.concat([x for x in ex.map(one, [(b, mode) for b in U162]) if len(x)])
        g, l = t.ret[t.ret > 0].sum(), -t.ret[t.ret < 0].sum()
        by = t.groupby(t.interval.round()).ret.agg(["count", "mean"])
        print(f"{mode:10s} n={len(t):5d} mean={t.ret.mean()*1e4:6.1f}bp pf={g/l:4.2f} | by interval: " +
              ", ".join(f"{int(i)}h n={int(r['count'])} {r['mean']*1e4:+.0f}bp" for i, r in by.iterrows()))
