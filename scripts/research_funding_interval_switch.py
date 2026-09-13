"""Funding-interval switch events on Binance perps (8h->4h->1h = exchange flags extreme crowding; back = calm).
Event = first settlement at the new interval. Side of the crowd = sign of the funding rate at the switch.
Arms: follow the crowd (long if funding>0) vs fade, delay 1/5/15/60, holds 15..1440 min."""
import glob, os, pandas as pd, numpy as np, importlib.util
from multiprocessing import Pool
spec = importlib.util.spec_from_file_location("g", "/root/freqtrade/scripts/research_event_generic.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
B = "/root/freqtrade/user_data/data/binance_public"
def events_for(f):
    s = os.path.basename(f)[:-8]
    d = pd.read_parquet(f)
    d = d[d.funding_interval_hours.notna()].sort_values("date")
    ch = d[d.funding_interval_hours != d.funding_interval_hours.shift(1)].iloc[1:]
    return [dict(symbol=s, release=r.date, title=f"{s} {int(prev)}h->{int(r.funding_interval_hours)}h", prev=int(prev), new=int(r.funding_interval_hours), fr=r.funding_rate)
            for prev, r in zip(d.funding_interval_hours.shift(1).loc[ch.index], ch.itertuples())]
with Pool(16) as p:
    ev = pd.DataFrame([e for lst in p.map(events_for, sorted(glob.glob(f"{B}/funding/*USDT.parquet"))) for e in lst])
ev["release"] = pd.to_datetime(ev.release, utc=True)
ev["dir"] = np.where(ev.new < ev.prev, "tighten", "loosen")
print(ev.groupby(["dir", "prev", "new"]).size().to_string()); print("funding sign at tighten:", np.sign(ev[ev.dir == "tighten"].fr).value_counts().to_dict())
# liquidity filter via qv30 inside run? keep all; report both directions separately. Crowd side = sign(fr).
ev.to_parquet("/root/freqtrade/user_data/events_20260912/funding_switch_events.parquet")
for dr in ["tighten", "loosen"]:
    e = ev[(ev.dir == dr) & (ev.fr != 0)].copy()
    df, res = g.run(e, f"funding_switch_{dr}", min_n=20)
    # relabel arms relative to the crowd: crowd = long if funding > 0 (longs pay), short if funding < 0
    key = dict(zip(zip(e.symbol, e.release), e.fr))
    df["fr"] = [key[(s, r)] for s, r in zip(df.symbol, df.release)]
    df["vs_crowd"] = np.where(((df.side == "long") & (df.fr > 0)) | ((df.side == "short") & (df.fr < 0)), "with_crowd(pay side)", "against_crowd(receive side)")
    print(f"--- {dr}: net % by crowd relation x delay x hold (n / mean / median / win / drop3)")
    t = df.groupby(["vs_crowd", "delay", "hold"]).net.agg(n="count", mean=lambda x: x.mean() * 100, median=lambda x: x.median() * 100, win=lambda x: (x > 0).mean() * 100, drop3=lambda x: x.sort_values().iloc[:-3].mean() * 100)
    print(t.round(2).to_string())
    print("by funding magnitude at switch (|fr| bps buckets), with-crowd side, delay 5 hold 240:")
    w = df[(df.vs_crowd.str.startswith("with")) & (df.delay == 5) & (df.hold == 240)]
    print(w.groupby(pd.cut(w.fr.abs() * 1e4, [0, 10, 30, 50, 100, 1000])).net.agg(n="count", mean=lambda x: x.mean() * 100, median=lambda x: x.median() * 100).round(2).to_string())
