"""Summarise the two-source DelistShort1m backtest by enter_tag and reconcile each leg with its event study."""
import glob, os, importlib.util
import numpy as np, pandas as pd
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats
RES = "/root/freqtrade/user_data/delist/backtest_results"
spec = importlib.util.spec_from_file_location("rds", "/root/freqtrade/scripts/research_delist_short.py"); rds = importlib.util.module_from_spec(spec); spec.loader.exec_module(rds)
path = sorted(glob.glob(f"{RES}/*.zip"), key=os.path.getmtime)[-1]
st = load_backtest_stats(path)["strategy"]["DelistShort1m"]; tr = load_backtest_data(path).sort_values("open_date")
print(f"file {path}\ntrades {len(tr)}, profit {st['profit_total']*100:+.2f}% ({st['profit_total_abs']:+.2f} USDT), PF {st['profit_factor']:.3f}, max dd {st['max_drawdown_account']*100:.2f}%, funding {tr.funding_fees.sum():+.2f} USDT, rejected {st.get('rejected_signals')}")
pd.set_option("display.width", 250)
print(tr.groupby("enter_tag").agg(n=("pair", "count"), mean_ret=("profit_ratio", lambda x: x.mean() * 100), median_ret=("profit_ratio", lambda x: x.median() * 100), win=("profit_ratio", lambda x: (x > 0).mean() * 100), pnl=("profit_abs", "sum"), hold_h=("trade_duration", lambda x: x.median() / 60)).round(2).to_string())
print(pd.crosstab(tr.enter_tag, tr.exit_reason))
tr["q"] = pd.to_datetime(tr.open_date, utc=True).dt.to_period("Q").astype(str)
print(tr.pivot_table(index="q", columns="enter_tag", values="profit_abs", aggfunc=["count", "sum"]).round(1).to_string())
# reconcile futures leg with research delay-5 / stop-15
r = pd.read_parquet(rds.OUT / "trades_all_delays.parquet"); r = r[r.delay == 5]; r["pair"] = r.symbol.str[:-4] + "/USDT:USDT"
f = tr[tr.enter_tag == "delist_futures"].merge(r[["pair", "entry_t", "net10"]], on="pair", how="left")
f["dt"] = (pd.to_datetime(f.open_date, utc=True) - f.entry_t).dt.total_seconds() / 60
print(f"\nfutures leg vs research: n {len(f)}, engine mean {f.profit_ratio.mean()*100:+.2f}% vs research {f.net10.mean()*100:+.2f}%, entry offset min/max {f.dt.min()}/{f.dt.max()} min, unmatched {f.net10.isna().sum()}")
# reconcile spot leg with a research replay: delay 15, stop 25, exit min(spot delist, data end) - 60m
import json
notices = json.load(open(rds.OUT / "spot_delist_notices.json"))
rows = []
for n in notices:
    if not n["delist_utc"]: continue
    for c in n["perps_local"]:
        s = f"{c}USDT"; d = rds.load_symbol(s)
        if d is None: continue
        k, m, fu = d
        rel = pd.Timestamp(n["release"]); t0 = (rel + pd.Timedelta(minutes=15)).ceil("1min"); t1 = (pd.Timestamp(n["delist_utc"], tz="UTC") - pd.Timedelta(minutes=60)).floor("1min")
        if t0 not in k.index: continue
        if t1 not in k.index: t1 = k.index[-1]
        if t0 >= t1: continue
        e = k.at[t0, "open"]; p = k.loc[t0:t1]; xt = t1
        hit = p.index[p.close >= e * 1.25]
        if len(hit):
            nx = p.index[p.index > hit[0]]
            if len(nx): xt = nx[0]
        g = 1 - k.at[xt, "open"] / e; fund = sum(rr * m.at[t] / e for t, rr in fu[(fu.index > t0) & (fu.index <= xt)].items() if t in m.index)
        rows.append({"pair": f"{c}/USDT:USDT", "res_entry": t0, "res_exit": xt, "res_net": g + fund - 0.002})
rs = pd.DataFrame(rows)
s = tr[tr.enter_tag == "delist_spot"].merge(rs, on="pair", how="outer", indicator=True)
s["dt_entry"] = (pd.to_datetime(s.open_date, utc=True) - s.res_entry).dt.total_seconds() / 60
s["dt_exit"] = (pd.to_datetime(s.close_date, utc=True) - s.res_exit).dt.total_seconds() / 60
both = s[s._merge == "both"]
print(f"\nspot leg vs research replay (delay 15, stop 25): engine n {int((s._merge!='right_only').sum())}, research n {len(rs)}, matched {len(both)}; engine mean {both.profit_ratio.mean()*100:+.2f}% vs research {both.res_net.mean()*100:+.2f}%; diff bps mean {((both.profit_ratio-both.res_net)*1e4).mean():+.1f} max {((both.profit_ratio-both.res_net)*1e4).abs().max():.1f}; exit offsets >1min: {(both.dt_exit.abs()>1).sum()}")
print("research-only (no engine trade):", s[s._merge == "right_only"].pair.tolist())
print("engine-only:", s[s._merge == "left_only"].pair.tolist())
print(s[s._merge == "both"][["pair", "open_date", "dt_entry", "dt_exit", "exit_reason", "profit_ratio", "res_net"]].assign(profit_ratio=lambda x: (x.profit_ratio * 100).round(1), res_net=lambda x: (x.res_net * 100).round(1)).sort_values("open_date").to_string(index=False))
