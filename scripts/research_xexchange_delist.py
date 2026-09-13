"""Bybit / OKX delisting notices -> short the Binance perp of the same coin (no Binance settlement; fixed holds)."""
import pandas as pd, importlib.util
spec = importlib.util.spec_from_file_location("g", "/root/freqtrade/scripts/research_event_generic.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
ev = pd.read_csv("/root/freqtrade/user_data/xdelist_20260912/events_bybit_okx.csv"); ev["release"] = pd.to_datetime(ev.release, utc=True, format="ISO8601")
ev = ev[ev.perp_ok].copy()
# exclude coins Binance itself delisted within +-10 days (those are the Binance event, not a cross-exchange one)
bn = pd.read_csv("/root/freqtrade/user_data/delist_short_20260906/events_parsed.csv"); bn["release"] = pd.to_datetime(bn.release, utc=True)
import json
sp = json.load(open("/root/freqtrade/user_data/delist_short_20260906/spot_delist_notices.json"))
bn_ev = [(f"{c}USDT", pd.Timestamp(n["release"])) for n in sp for c in n["coins"]] + list(zip(bn.symbol, bn.release))
def overlaps(s, t): return any(s == bs and abs((t - bt).total_seconds()) < 10 * 86400 for bs, bt in bn_ev)
ev["binance_overlap"] = [overlaps(s, t) for s, t in zip(ev.symbol, ev.release)]
print("cross-exchange events:", len(ev), "of which overlapping a Binance delisting notice within 10 days:", int(ev.binance_overlap.sum()))
df, res = g.run(ev, "xexchange_delist_all", sides=("short",))
df2, res2 = g.run(ev[~ev.binance_overlap], "xexchange_delist_noBinance", sides=("short",))
