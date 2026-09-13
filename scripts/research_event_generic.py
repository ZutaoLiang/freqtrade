"""Generic announcement event study on Binance USDT perps (shared by the 2026-09-12 batch).
Input: DataFrame of events (symbol, release [UTC], title, optional 'end' = hard exit time).
Arms: side {long, short} x delay {1,5,15,60} min x hold {15,60,240,1440} min, close-based stop
(10% for long... we use the same 15% for both sides), 10 bp/side, real funding, BTC control,
cluster bootstrap by notice hour, drop best 3, period split.
"""
import importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("rds", "/root/freqtrade/scripts/research_delist_short.py"); rds = importlib.util.module_from_spec(spec); spec.loader.exec_module(rds)
DELAYS = [1, 5, 15, 60]; HOLDS = [15, 60, 240, 1440]; STOP = 0.15

def run(ev: pd.DataFrame, label: str, cache=None, sides=("long", "short"), min_n=5):
    cache = cache or {}
    btc = pd.read_parquet(rds.B / "klines_1m/BTCUSDT.parquet").set_index("date").open
    rows = []
    for _, e in ev.iterrows():
        if e.symbol not in cache:
            try:
                cache[e.symbol] = rds.load_symbol(e.symbol)
            except FileNotFoundError:
                cache[e.symbol] = None
        d = cache[e.symbol]
        if d is None: continue
        k, m, f = d
        for delay in DELAYS:
            t0 = (e.release + pd.Timedelta(minutes=delay)).ceil("1min")
            if t0 not in k.index: continue
            entry = k.at[t0, "open"]
            for hold in HOLDS:
                t1 = t0 + pd.Timedelta(minutes=hold)
                if "end" in ev.columns and pd.notna(e.get("end")): t1 = min(t1, pd.Timestamp(e["end"]))
                if t1 not in k.index or t1 <= t0: continue
                p = k.loc[t0:t1]
                for side in sides:
                    sgn = 1 if side == "long" else -1
                    xt, rs = t1, "hold"
                    hit = p.index[(p.close <= entry * (1 - STOP))] if side == "long" else p.index[(p.close >= entry * (1 + STOP))]
                    if len(hit):
                        nx = p.index[p.index > hit[0]]
                        if len(nx): xt, rs = nx[0], "stop"
                    g = sgn * (k.at[xt, "open"] / entry - 1)
                    fu = -sgn * sum(r * m.at[t] / entry for t, r in f[(f.index > t0) & (f.index <= xt)].items() if t in m.index)
                    bt = sgn * (btc.get(xt, np.nan) / btc.get(t0, np.nan) - 1) if t0 in btc.index and xt in btc.index else np.nan
                    rows.append(dict(symbol=e.symbol, notice=e.title, release=e.release, side=side, delay=delay, hold=hold, net=g + fu - 0.002, reason=rs, btc=bt, period=rds.period_of(e.release)))
    df = pd.DataFrame(rows)
    df.to_parquet(f"/root/freqtrade/user_data/events_20260912/{label}_arms.parquet")
    pd.set_option("display.width", 250)
    out = []
    for (side, delay, hold), g in df.groupby(["side", "delay", "hold"]):
        if len(g) < min_n: continue
        gg = g.rename(columns={"notice": "notice"}); gg["notice"] = gg.release.dt.floor("h").astype(str)
        ci, p = rds.cluster_boot(gg, "net", n=2000)
        out.append(dict(side=side, delay=delay, hold=hold, n=len(g), mean=g.net.mean() * 100, median=g.net.median() * 100, win=(g.net > 0).mean() * 100, lo=ci[0] * 100, hi=ci[1] * 100,
                        drop3=g.net.sort_values().iloc[:-3].mean() * 100 if len(g) > 3 else np.nan, btc=g.btc.mean() * 100, stop=(g.reason == "stop").mean() * 100,
                        p25=g[g.period == "2025_train"].net.mean() * 100, p26h1=g[g.period == "2026H1_valid"].net.mean() * 100, p26jun=g[g.period == "2026Jun+_holdout"].net.mean() * 100))
    res = pd.DataFrame(out).round(2)
    print(f"\n===== {label}: {df.symbol.nunique()} symbols, {df.release.nunique()} notices; net % after 10bp/side; CI by notice-hour cluster")
    print(res.to_string(index=False))
    return df, res
