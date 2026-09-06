"""Sensitivity table for the delisting-short event study (reported in full, nothing selected).
Grid: entry delay {5, 15, 60, 240, 1440} min x stop {10%, 15%, 25%, none} x exit {24h, 48h, settle-6h, settle-60m}.
Also: mean short return vs hours-to-settlement (path), max adverse excursion, simple fixed-stake ledger.
"""
import numpy as np, pandas as pd
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location("rds", "/root/freqtrade/scripts/research_delist_short.py"); rds = importlib.util.module_from_spec(spec); spec.loader.exec_module(rds)
OUT = rds.OUT; B = rds.B
ev = pd.read_csv(OUT / "events_parsed.csv"); ev["release"] = pd.to_datetime(ev.release, utc=True); ev["settle"] = pd.to_datetime(ev.settle, utc=True)
ev = ev[ev.release >= rds.DATA_START].sort_values("release").reset_index(drop=True)
cache = {s: rds.load_symbol(s) for s in ev.symbol.unique()}

def one(e, delay, stop, exit_rule):
    d = cache[e.symbol]
    if d is None: return None
    k, m, f = d
    entry_t = (e.release + pd.Timedelta(minutes=delay)).ceil("1min")
    if entry_t not in k.index: return None
    if exit_rule == "settle-60m": force_t = (e.settle - pd.Timedelta(minutes=60)).floor("1min")
    elif exit_rule == "settle-6h": force_t = (e.settle - pd.Timedelta(hours=6)).floor("1min")
    elif exit_rule == "24h": force_t = min(entry_t + pd.Timedelta(hours=24), (e.settle - pd.Timedelta(minutes=60)).floor("1min"))
    elif exit_rule == "48h": force_t = min(entry_t + pd.Timedelta(hours=48), (e.settle - pd.Timedelta(minutes=60)).floor("1min"))
    if entry_t >= force_t: return None
    entry = k.at[entry_t, "open"]; path = k.loc[entry_t:force_t]
    exit_t, reason = force_t, "force"
    if stop is not None:
        hit = path.index[path.close >= entry * (1 + stop)]
        if len(hit):
            nxt = path.index[path.index > hit[0]]
            if len(nxt): exit_t, reason = nxt[0], "stop"
    if exit_t not in k.index: exit_t, reason = path.index[-1], "data_end"
    exit_px = k.at[exit_t, "open"]
    gross = 1 - exit_px / entry
    fs = f[(f.index > entry_t) & (f.index <= exit_t)]
    fund = sum(r * m.at[t] / entry for t, r in fs.items() if t in m.index)
    mae = path.loc[:exit_t].high.max() / entry - 1  # worst intrabar rise against the short
    return dict(symbol=e.symbol, notice=e.title, period=rds.period_of(e.release), delay=delay, stop=stop, exit=exit_rule,
                net=gross + fund - 0.002, reason=reason, hold_h=(exit_t - entry_t).total_seconds() / 3600, mae=mae, entry_t=entry_t, exit_t=exit_t)

rows = []
for delay in [5, 15, 60, 240, 1440]:
    for stop in [0.10, 0.15, 0.25, None]:
        for ex in ["24h", "48h", "settle-6h", "settle-60m"]:
            for _, e in ev.iterrows():
                r = one(e, delay, stop, ex)
                if r: rows.append(r)
df = pd.DataFrame(rows); df["stop"] = df.stop.fillna(0).astype(float)
df.to_parquet(OUT / "sensitivity_grid.parquet")
pd.set_option("display.width", 250)
def agg(g):
    ci, p = rds.cluster_boot(g, "net", n=2000)
    return pd.Series(dict(n=len(g), mean=g.net.mean() * 100, median=g.net.median() * 100, win=(g.net > 0).mean() * 100,
                          lo=ci[0] * 100, hi=ci[1] * 100, drop3=g.net.sort_values().iloc[:-3].mean() * 100,
                          train=g[g.period == "2025_train"].net.mean() * 100, valid=g[g.period == "2026H1_valid"].net.mean() * 100,
                          stop_pct=(g.reason == "stop").mean() * 100, hold_med_h=g.hold_h.median(), mae_max=g.mae.max() * 100))
print("=== sensitivity (net %, 10bp/side + funding):")
print(df.groupby(["delay", "stop", "exit"]).apply(agg).round(1).to_string())

# path vs hours-to-settlement, delay 5 entries, no stop
print("\n=== mean/median cumulative short return (%) by hours after delay-5 entry (no stop, price only), and by hours before settlement:")
paths = []
for _, e in ev.iterrows():
    d = cache[e.symbol]
    if d is None: continue
    k = d[0]; entry_t = (e.release + pd.Timedelta(minutes=5)).ceil("1min")
    force_t = (e.settle - pd.Timedelta(minutes=60)).floor("1min")
    if entry_t not in k.index or entry_t >= force_t: continue
    seg = k.loc[entry_t:force_t].open; entry = seg.iloc[0]
    s = 1 - seg / entry
    h_after = ((seg.index - entry_t).total_seconds() / 3600)
    h_before = ((e.settle - seg.index).total_seconds() / 3600)
    paths.append(pd.DataFrame({"sym": e.symbol, "h_after": h_after.round(0), "h_before": h_before.round(0), "r": s.values}))
P = pd.concat(paths)
a = P.groupby("h_after").r.agg(["mean", "median", "count"]); print("hours after entry:"); print((a.loc[[0, 1, 2, 4, 8, 12, 24, 36, 48, 72, 96, 110]].assign(mean=lambda x: x["mean"] * 100, median=lambda x: x["median"] * 100)).round(2).to_string())
b = P.groupby("h_before").r.agg(["mean", "median", "count"]); print("hours before settlement:"); print((b.loc[[96, 72, 48, 24, 12, 6, 3, 2, 1]].assign(mean=lambda x: x["mean"] * 100, median=lambda x: x["median"] * 100)).round(2).to_string())

# fixed-stake ledger: delay 5, stop 15, settle-60m; 150 USDT per event, all events taken (max concurrent?)
g = df[(df.delay == 5) & (df.stop == 0.15) & (df.exit == "settle-60m")].sort_values("entry_t")
conc = max(((g.entry_t <= t) & (g.exit_t > t)).sum() for t in g.entry_t)
print(f"\nledger delay5/stop15/settle-60m: {len(g)} trades, 150 USDT each: total P&L {(g.net * 150).sum():.1f} USDT, max concurrent {conc}, worst trade {(g.net * 150).min():.1f}, best {(g.net * 150).max():.1f}")
print("by notice (sum P&L USDT):"); print((g.groupby("notice").net.sum() * 150).round(1).sort_values().to_string())
