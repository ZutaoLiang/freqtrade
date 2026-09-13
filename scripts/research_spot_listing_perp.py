"""Binance 'Will List X (Seed Tag)' spot listing notices for coins whose Binance perp already trades."""
import json, re, os, pandas as pd, importlib.util
spec = importlib.util.spec_from_file_location("g", "/root/freqtrade/scripts/research_event_generic.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
B = "/root/freqtrade/user_data/data/binance_public/klines_1m"
arts = json.load(open("/root/freqtrade/user_data/xdelist_20260912/binance_catalog48.json"))
rows = []
for x in arts:
    t = x["title"]
    if not re.match(r"Binance Will List ", t): continue
    ts = pd.Timestamp(x["releaseDate"], unit="ms", tz="UTC")
    for c in re.findall(r"\(([A-Z0-9]{2,12})\)", t):
        s = f"{c}USDT"
        if os.path.exists(f"{B}/{s}.parquet"):
            k = pd.read_parquet(f"{B}/{s}.parquet", columns=["date", "volume"]); nz = k[k.volume > 0]
            if len(nz) and nz.date.min() <= ts - pd.Timedelta(days=1):
                rows.append(dict(symbol=s, release=ts, title=t))
ev = pd.DataFrame(rows)
print("spot-listing events with pre-existing perp:", len(ev))
df, res = g.run(ev, "binance_spot_listing")
# path around the notice
paths = []
for _, e in ev.iterrows():
    d = g.rds.load_symbol(e.symbol); k = d[0]; t0 = e.release.floor("1min")
    seg = k.loc[t0 - pd.Timedelta(minutes=60): t0 + pd.Timedelta(minutes=240)].open
    if t0 in seg.index: paths.append(((seg / seg[t0] - 1) * 100).rename(e.symbol).set_axis(((seg.index - t0).total_seconds() // 60).astype(int)))
P = pd.concat(paths, axis=1); print("price path % vs notice minute (mean/median):"); print(pd.DataFrame({"mean": P.mean(axis=1), "median": P.median(axis=1)}).loc[[-60, -30, -15, -5, 0, 1, 2, 5, 10, 15, 30, 60, 120, 240]].round(2).T.to_string())
