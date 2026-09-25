"""Scheme-C / §4 report for the DualSqueezeBtcTrend1h quarterly-chunk backtests (per-trade profit_ratio incl. leverage 1.5)."""
import glob, json, sys, zipfile
sys.path.insert(0, "scripts/minute_research/r5")
import numpy as np, pandas as pd
import recheck_schemeC as R
fee = sys.argv[1]
rows = []
for z in sorted(glob.glob(f"user_data/minute_research/r5/dualsq_bt/{fee}/*.zip")):
    with zipfile.ZipFile(z) as zf:
        js = [n for n in zf.namelist() if n.endswith(".json") and "config" not in n and "meta" not in n]
        res = json.loads(zf.read(js[0]))
    rows.append(pd.DataFrame(res["strategy"]["DualSqueezeBtcTrend1h"]["trades"]))
t = pd.concat(rows, ignore_index=True)
t["t"] = pd.to_datetime(t.open_date, utc=True); t["ret"] = t.profit_ratio; t["side"] = np.where(t.is_short, -1, 1)
t["pair"] = t.pair.str.split("/").str[0]; t["cost"] = float(fee.split("-")[-1]) * 1.5; t["seg"] = R.segC(t.t)
print(f"fee {fee}: trades {len(t)}  exits {t.exit_reason.value_counts().to_dict()}  funding USDT {t.funding_fees.sum():.2f}")
for s in ("TRAIN", "EXCL", "VALID", "HOLDOUT"):
    x = t[t.seg == s]; st = R.stats(x)
    usd = x.profit_abs.sum()
    print(f"  {s:7s} n{st.get('n')} days{st.get('days')} pd{st.get('per_day')} {st.get('net_bp')}bp PF{st.get('pf')} t{st.get('t')} mon+{st.get('mon+')} top_pair{st.get('top_pair')} top_day{st.get('top_day')} L{st.get('L_bp')} S{st.get('S_bp')}  USDT {usd:.1f}")
vh = t[t.seg.isin(["VALID", "HOLDOUT"])]; dm = vh.groupby(vh.t.dt.floor("D")).ret.sum()
mon = vh.groupby(vh.t.dt.tz_localize(None).dt.to_period("M")).ret.sum()
print(f"  V+HO: t {dm.mean()/dm.std()*np.sqrt(len(dm)):.2f}  months+ {(mon>0).sum()}/{len(mon)}  | TRAIN/VALID/HO verdict: {R.verdict(R.stats(t[t.seg=='TRAIN']), R.stats(t[t.seg=='VALID']))}")
t.to_parquet(f"user_data/minute_research/r5/dualsq_trades_{fee}.parquet")
