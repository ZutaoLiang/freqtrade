"""Scheme-C / §4 report for AllWeatherRegimeAdaptive chunk backtests, split by entry tag (engine)."""
import glob, json, sys, zipfile
sys.path.insert(0, "scripts/minute_research/r5")
import numpy as np, pandas as pd
import recheck_schemeC as R
fee = sys.argv[1]
rows, wallets = [], []
for z in sorted(glob.glob(f"user_data/minute_research/r5/aw_bt/{fee}/*.zip")):
    with zipfile.ZipFile(z) as zf:
        js = [n for n in zf.namelist() if n.endswith(".json") and "config" not in n and "meta" not in n]
        res = json.loads(zf.read(js[0]))
    st = res["strategy"]["AllWeatherRegimeAdaptive"]
    rows.append(pd.DataFrame(st["trades"])); wallets.append((st["backtest_start"], st["backtest_end"], st["starting_balance"], st["final_balance"]))
t = pd.concat(rows, ignore_index=True)
t["t"] = pd.to_datetime(t.open_date, utc=True); t["ret"] = t.profit_ratio; t["side"] = np.where(t.is_short, -1, 1)
t["pair"] = t.pair.str.split("/").str[0]; t["cost"] = float(fee) * 1.5; t["seg"] = R.segC(t.t)
print(f"fee {fee}: trades {len(t)}  funding USDT {t.funding_fees.sum():.2f}")
print("chunk wallet returns (reset to 100 each chunk):", [(w[0][:10], round((w[3] / w[2] - 1) * 100, 1)) for w in wallets])
for s in ("TRAIN", "EXCL", "VALID", "HOLDOUT"):
    x = t[t.seg == s]; stt = R.stats(x)
    print(f"  {s:7s} ALL n{stt.get('n')} {stt.get('net_bp')}bp PF{stt.get('pf')} t{stt.get('t')} mon+{stt.get('mon+')} top_pair{stt.get('top_pair')} top_day{stt.get('top_day')}")
    for tag, g in x.groupby("enter_tag"):
        gs = R.stats(g); print(f"          {tag:14s} n{gs.get('n')} {gs.get('net_bp')}bp PF{gs.get('pf')} t{gs.get('t')}")
vh = t[t.seg.isin(["VALID", "HOLDOUT"])]; dm = vh.groupby(vh.t.dt.floor("D")).ret.sum()
mon = vh.groupby(vh.t.dt.tz_localize(None).dt.to_period("M")).ret.sum()
print(f"  V+HO ALL: t {dm.mean()/dm.std()*np.sqrt(len(dm)):.2f}  months+ {(mon>0).sum()}/{len(mon)}")
for tag, g in vh.groupby("enter_tag"):
    d = g.groupby(g.t.dt.floor("D")).ret.sum(); print(f"    V+HO {tag:14s} n{len(g)} {g.ret.mean()*1e4:.1f}bp t {d.mean()/d.std()*np.sqrt(len(d)):.2f}")
print("  exit reasons:", t.exit_reason.value_counts().head(10).to_dict())
t.to_parquet(f"user_data/minute_research/r5/aw_trades_{fee}.parquet")
