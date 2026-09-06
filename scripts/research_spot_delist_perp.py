"""Event study: short the USDT perpetual after a Binance SPOT delisting notice
("Binance Will Delist A, B, C on DATE"; also the 2025-04 "Vote to Delist" batch).

The perp keeps trading after the spot delisting; the futures contract is usually delisted
weeks later by a separate notice (that event is the DelistShort1m strategy).

Frozen design (mirrors the futures-notice study):
- Entry: short at the first 1m open >= notice + 5 min (also 15, 60 min reported).
- Arms: (a) 15% close-stop, exit at spot-delisting time - 60 min;
        (b) fixed 24h / 72h / 168h holds with the same stop.
- Costs 10 bp/side, actual funding, arithmetic short return.
- Cluster bootstrap by notice, drop best 3, split 2025 / 2026H1 / 2026Jun+.
- Only perps with non-zero-volume 1m data at the notice time are used.
"""
import json
import importlib.util
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("rds", "/root/freqtrade/scripts/research_delist_short.py")
rds = importlib.util.module_from_spec(spec); spec.loader.exec_module(rds)
OUT = rds.OUT
notices = json.load(open(OUT / "spot_delist_notices.json"))
rows = []
for n in notices:
    if not n["delist_utc"]:
        continue
    for c in n["perps_local"]:
        rows.append({"symbol": f"{c}USDT", "release": pd.Timestamp(n["release"]), "delist": pd.Timestamp(n["delist_utc"], tz="UTC"), "title": n["title"]})
ev = pd.DataFrame(rows).sort_values("release").reset_index(drop=True)
ev["settle"] = ev.delist   # reuse run_event's force-exit logic: settle - 60 min
cache = {s: rds.load_symbol(s) for s in ev.symbol.unique()}
btc = pd.read_parquet(rds.B / "klines_1m/BTCUSDT.parquet").set_index("date")


def fixed_hold(e, data, delay, hours, stop=0.15):
    k, m, f = data
    t0 = (e.release + pd.Timedelta(minutes=delay)).ceil("1min"); t1 = t0 + pd.Timedelta(hours=hours)
    if t0 not in k.index or t1 not in k.index:
        return None
    entry = k.at[t0, "open"]; p = k.loc[t0:t1]; xt, rs = t1, "hold"
    hit = p.index[p.close >= entry * (1 + stop)]
    if len(hit):
        nx = p.index[p.index > hit[0]]
        if len(nx): xt, rs = nx[0], "stop"
    g = 1 - k.at[xt, "open"] / entry
    fu = sum(r * m.at[t] / entry for t, r in f[(f.index > t0) & (f.index <= xt)].items() if t in m.index)
    return g + fu - 0.002, rs


recs, fixed, excl, paths = [], [], [], []
for _, e in ev.iterrows():
    d = cache[e.symbol]
    if d is None:
        excl.append((e.symbol, "no data")); continue
    for delay in [5, 15, 60]:
        r = rds.run_event(e, d, btc, delay)
        if r.get("excluded"):
            excl.append((e.symbol, delay, r["excluded"]))
        else:
            recs.append(r)
        for h in [24, 72, 168]:
            x = fixed_hold(e, d, delay, h)
            if x: fixed.append({"symbol": e.symbol, "notice": e.title, "delay": delay, "hours": h, "net": x[0], "reason": x[1], "period": rds.period_of(e.release)})
    p = rds.cum_path(e, d)
    if p is not None: paths.append(p.rename(e.symbol))
df = pd.DataFrame(recs); df.to_parquet(OUT / "spot_delist_trades.parquet")
fx = pd.DataFrame(fixed); fx.to_parquet(OUT / "spot_delist_fixed.parquet")
pd.set_option("display.width", 250)
print(f"spot-delist events with perp data: {df[df.delay == 5].symbol.nunique()} symbols, {df.notice.nunique()} notices; excluded {len(excl)}: {excl[:6]}")
for delay in [5, 15, 60]:
    g = df[df.delay == delay]
    print(f"\n=== delay {delay}: hold to spot delisting - 60m, 15% stop: n={len(g)}, exits {g.exit_reason.value_counts().to_dict()}, hold median {g.hold_h.median():.0f}h")
    print(rds.summarize(g).round(2).to_string())
    print("fixed horizons (price only) mean/median %:", {f"h{h}": f"{g[f'h{h}'].mean()*100:+.2f}/{g[f'h{h}'].median()*100:+.2f}" for h in rds.HORIZONS})
print("\n=== fixed holds with 15% stop, net % (mean / median / win / n) by delay x hours:")
print(fx.groupby(["delay", "hours"]).net.agg(n="count", mean=lambda x: x.mean() * 100, median=lambda x: x.median() * 100, win=lambda x: (x > 0).mean() * 100, stop_pct=lambda x: np.nan).round(1).to_string())
print(fx[fx.delay == 5].groupby(["hours", "period"]).net.agg(n="count", mean=lambda x: x.mean() * 100).round(1).to_string())
P = pd.concat(paths, axis=1)
print("\nmean/median cum short % at minute k after notice:", (P.mean(axis=1) * 100).loc[[1, 5, 15, 30, 60, 120, 180]].round(2).to_dict(), (P.median(axis=1) * 100).loc[[5, 30, 60, 180]].round(2).to_dict())
g = df[df.delay == 5].sort_values("release")
print(g[["symbol", "notice", "entry_t", "hold_h", "exit_reason", "gross", "funding", "net10", "pre24h_ret", "qv30_median"]].assign(gross=lambda x: x.gross * 100, funding=lambda x: x.funding * 100, net10=lambda x: x.net10 * 100, pre24h_ret=lambda x: x.pre24h_ret * 100, notice=lambda x: x.notice.str[:45]).round(2).to_string(index=False))
