"""Placebo for the delisting short: same symbol, same hold length, same 15% close stop, same costs,
entry shifted to 7 / 14 / 21 days BEFORE the announcement (same time of day). If the real event
mean is not clearly above the placebo mean, the 'edge' is just these coins' pre-existing drift."""
import numpy as np, pandas as pd, importlib.util
spec = importlib.util.spec_from_file_location("rds", "/root/freqtrade/scripts/research_delist_short.py"); rds = importlib.util.module_from_spec(spec); spec.loader.exec_module(rds)
OUT = rds.OUT
ev = pd.read_csv(OUT / "events_parsed.csv"); ev["release"] = pd.to_datetime(ev.release, utc=True); ev["settle"] = pd.to_datetime(ev.settle, utc=True)
ev = ev[ev.release >= rds.DATA_START].sort_values("release").reset_index(drop=True)
cache = {s: rds.load_symbol(s) for s in ev.symbol.unique()}
real = pd.read_parquet(OUT / "trades_all_delays.parquet"); real = real[real.delay == 5].set_index("symbol")

def sim(k, m, f, entry_t, force_t, stop=0.15):
    if entry_t not in k.index or force_t not in k.index or entry_t >= force_t: return None
    entry = k.at[entry_t, "open"]; path = k.loc[entry_t:force_t]
    exit_t, reason = force_t, "force"
    hit = path.index[path.close >= entry * (1 + stop)]
    if len(hit):
        nxt = path.index[path.index > hit[0]]
        if len(nxt): exit_t, reason = nxt[0], "stop"
    gross = 1 - k.at[exit_t, "open"] / entry
    fs = f[(f.index > entry_t) & (f.index <= exit_t)]
    fund = sum(r * m.at[t] / entry for t, r in fs.items() if t in m.index)
    return gross + fund - 0.002, reason

rows = []
for _, e in ev.iterrows():
    d = cache[e.symbol]
    if d is None or e.symbol not in real.index: continue
    k, m, f = d
    r0 = real.loc[e.symbol]
    hold = pd.Timedelta(hours=float(r0.hold_h)) if r0.exit_reason == "force" else (r0.exit_t - r0.entry_t)
    # placebo hold = the real arm's forced-exit horizon (settlement-60m minus entry), not the stopped length
    hold = (e.settle - pd.Timedelta(minutes=60)).floor("1min") - r0.entry_t
    for shift in [7, 14, 21]:
        et = (r0.entry_t - pd.Timedelta(days=shift)); ft = et + hold
        res = sim(k, m, f, et, ft)
        if res: rows.append(dict(symbol=e.symbol, notice=e.title, shift=shift, net=res[0], reason=res[1]))
    rows.append(dict(symbol=e.symbol, notice=e.title, shift=0, net=r0.net10, reason=r0.exit_reason))
df = pd.DataFrame(rows)
print(df.groupby("shift").net.agg(n="count", mean=lambda x: x.mean() * 100, median=lambda x: x.median() * 100, win=lambda x: (x > 0).mean() * 100).round(2).to_string())
# paired difference real - placebo, cluster bootstrap by notice
for shift in [7, 14, 21]:
    p = df[df["shift"] == shift].set_index("symbol").net; r = df[df["shift"] == 0].set_index("symbol")
    j = r.join(p.rename("plc"), how="inner"); j["diff"] = j.net - j.plc
    ci, pv = rds.cluster_boot(j.reset_index(), "diff", n=5000)
    print(f"shift -{shift}d: paired n={len(j)}, real-placebo mean diff {j['diff'].mean()*100:+.2f}%, CI95 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}], P(diff<=0)={pv:.4f}")
df.to_csv(OUT / "placebo.csv", index=False)
