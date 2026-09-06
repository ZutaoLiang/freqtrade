"""Replay the frozen delisting-short event study on the 2024 notices (data from scripts/fetch_delist_2024.py)."""
import importlib.util, json, sys
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("rds", "/root/freqtrade/scripts/research_delist_short.py"); rds = importlib.util.module_from_spec(spec); spec.loader.exec_module(rds)
rds.B = __import__("pathlib").Path("/root/freqtrade/user_data/delist2024/public")
OUT = __import__("pathlib").Path("/root/freqtrade/user_data/delist2024"); rds.OUT = OUT
ev = pd.DataFrame(json.load(open(OUT / "events.json")))
ev["release"] = pd.to_datetime(ev.release, utc=True); ev["settle"] = pd.to_datetime(ev.settle, utc=True); ev["title"] = ev.notice
ev = ev.sort_values("release").reset_index(drop=True)
# BTC control: 2024 BTC not in this dir -> use an empty frame (control reported as NaN)
btc = pd.DataFrame(columns=["open"], index=pd.DatetimeIndex([], tz="UTC"))
recs, excl, paths = [], [], []
for _, e in ev.iterrows():
    d = rds.load_symbol(e.symbol)
    if d is None:
        excl.append({"symbol": e.symbol, "reason": "no data"}); continue
    for delay in rds.DELAYS:
        r = rds.run_event(e, d, btc, delay)
        (excl if r.get("excluded") else recs).append(r if not r.get("excluded") else {"symbol": e.symbol, "delay": delay, "reason": r["excluded"]})
    p = rds.cum_path(e, d)
    if p is not None: paths.append(p.rename(e.symbol))
df = pd.DataFrame(recs); df["period"] = "2024"; df["btc_short"] = np.nan
df.to_parquet(OUT / "trades_all_delays.parquet")
pd.set_option("display.width", 250)
print("exclusions:", excl)
for delay in rds.DELAYS:
    g = df[df.delay == delay]
    print(f"\n=== 2024, delay {delay}: n={len(g)}, exits {g.exit_reason.value_counts().to_dict()}, hold median {g.hold_h.median():.1f}h")
    print(rds.summarize(g).round(3).to_string())
    print("fixed horizons mean%/median%:", {f"h{h}": f"{g[f'h{h}'].mean()*100:+.2f}/{g[f'h{h}'].median()*100:+.2f}" for h in rds.HORIZONS})
P = pd.concat(paths, axis=1)
print("\nmean cum short % at minute k:", (P.mean(axis=1) * 100).loc[[1, 5, 15, 30, 60, 120, 180]].round(2).to_dict())
g = df[df.delay == 5].sort_values("release")
print(g[["symbol", "notice", "entry_t", "hold_h", "exit_reason", "gross", "funding", "net10", "pre24h_ret", "qv30_median"]].assign(gross=lambda x: x.gross * 100, funding=lambda x: x.funding * 100, net10=lambda x: x.net10 * 100, pre24h_ret=lambda x: x.pre24h_ret * 100).round(2).to_string(index=False))
