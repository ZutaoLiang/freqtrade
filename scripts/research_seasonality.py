"""C4: calendar seasonality on Binance USDT perps, 1h panel 2025-01-01..2026-08-16.

Two pre-registered rules (published elsewhere), tested as stated; the 168-cell table is descriptive only.
R1 'Monday Asia open': long from Sunday 23:00 UTC open to Monday 23:00 UTC open (24h), every week.
R2 'weekend momentum': sign of trailing 7-day return, position held Saturday 00:00 -> Monday 00:00 UTC
    (48h) vs the same rule on weekdays (each weekday 00:00 -> next 00:00), long/short.
Groups: BTC, ETH, 'majors' = top-20 by trailing-30d median daily quote volume re-ranked monthly
    (prior data only), 'alts' = ranks 21-120 by the same ranking.
Cost: 10 bp per side per position change; funding charged hourly as rate * 1/interval (short receives).
Return = simple return on notional, equal-weight across the group's names, reported per holding period.
"""
import glob, os
import numpy as np, pandas as pd

B = "/root/freqtrade/user_data/data/binance_public"
OUT = "/root/freqtrade/user_data/seasonality_20260906"
os.makedirs(OUT, exist_ok=True)
COST = 0.001

def load():
    closes, qvs, funds = {}, {}, {}
    for f in sorted(glob.glob(f"{B}/resampled/1h/*USDT.parquet")):
        s = os.path.basename(f)[:-8]
        d = pd.read_parquet(f).set_index("date")
        d = d[d.volume > 0]
        if len(d) < 24 * 60:
            continue
        closes[s] = d.close; qvs[s] = d.quote_volume
        af = f"{B}/aux_resampled/1h/{s}.parquet"
        if os.path.exists(af):
            a = pd.read_parquet(af).set_index("date")
            fr = a.funding_rate.fillna(0.0)
            # hourly accrual: rate / interval, spread over the interval preceding the settlement
            iv = a.funding_interval_hours.bfill().fillna(8.0)
            acc = (a.funding_rate / iv).bfill(limit=7).fillna(0.0)
            funds[s] = acc
    C = pd.DataFrame(closes); Q = pd.DataFrame(qvs); F = pd.DataFrame(funds).reindex(C.index).reindex(columns=C.columns).fillna(0.0)
    return C, Q, F

def groups(C, Q):
    qd = Q.resample("1D").sum()
    med30 = qd.rolling(30, min_periods=20).median().shift(1)
    # monthly ranking using data before the month
    ranks = {}
    for m in pd.period_range(C.index[0], C.index[-1], freq="M"):
        t = m.start_time.tz_localize("UTC")
        row = med30.loc[:t].iloc[-1] if (med30.index <= t).any() else None
        if row is None or row.notna().sum() < 50:
            continue
        order = row.dropna().sort_values(ascending=False).index.tolist()
        ranks[m] = {"majors": order[:20], "alts": order[20:120]}
    return ranks

def hold_return(C, F, names, t0, t1, side=1):
    """simple return of an equal-weight basket from t0 open (= close of previous bar; we use close at t0) to t1, incl funding, cost both sides."""
    names = [n for n in names if n in C.columns and t0 in C.index and t1 in C.index and np.isfinite(C.at[t0, n]) and np.isfinite(C.at[t1, n])]
    if not names:
        return np.nan
    px = C.loc[t0:t1, names]
    r = side * (px.iloc[-1] / px.iloc[0] - 1)
    fund = side * F.loc[C.index[C.index > t0][0]:t1, names].sum()   # short receives positive rate
    return (r - fund - 2 * COST).mean()   # long pays funding when rate>0

def main():
    C, Q, F = load()
    ranks = groups(C, Q)
    idx = C.index
    # R1: Sunday 23:00 -> Monday 23:00
    rows = []
    for t0 in idx[(idx.dayofweek == 6) & (idx.hour == 23)]:
        t1 = t0 + pd.Timedelta(hours=24)
        if t1 not in idx: continue
        m = t0.to_period("M")
        if m not in ranks: continue
        rows.append(dict(t0=t0, BTC=hold_return(C, F, ["BTCUSDT"], t0, t1), ETH=hold_return(C, F, ["ETHUSDT"], t0, t1),
                         majors=hold_return(C, F, ranks[m]["majors"], t0, t1), alts=hold_return(C, F, ranks[m]["alts"], t0, t1)))
    r1 = pd.DataFrame(rows).set_index("t0")
    # same window every other weekday as descriptive comparison
    comp = {}
    for dow in range(7):
        rr = []
        for t0 in idx[(idx.dayofweek == dow) & (idx.hour == 23)]:
            t1 = t0 + pd.Timedelta(hours=24)
            m = t0.to_period("M")
            if t1 not in idx or m not in ranks: continue
            rr.append(dict(BTC=hold_return(C, F, ["BTCUSDT"], t0, t1), majors=hold_return(C, F, ranks[m]["majors"], t0, t1), alts=hold_return(C, F, ranks[m]["alts"], t0, t1)))
        d = pd.DataFrame(rr)
        comp[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow] + " 23:00 +24h"] = pd.Series({f"{c}_mean%": d[c].mean() * 100 for c in d} | {f"{c}_t": d[c].mean() / d[c].std() * np.sqrt(len(d)) for c in d} | {"n": len(d)})
    pd.set_option("display.width", 250)
    print("=== R1: long from Sunday 23:00 UTC for 24h (net of 10bp/side + funding), mean % per week, t-stat, n")
    print(pd.DataFrame(comp).T.round(2).to_string())
    print("R1 by period (BTC / majors / alts mean %):")
    for name, g in r1.groupby(np.where(r1.index.year == 2025, "2025", "2026")):
        print(name, {c: f"{g[c].mean()*100:+.2f} (t {g[c].mean()/g[c].std()*np.sqrt(len(g)):+.2f}, n {len(g)})" for c in ["BTC", "ETH", "majors", "alts"]})
    r1.to_csv(f"{OUT}/r1_weekly.csv")

    # R2: 7-day momentum sign, weekend (Sat 00:00 -> Mon 00:00) vs each weekday (00:00 -> +24h)
    rows = []
    for t0 in idx[idx.hour == 0]:
        m = t0.to_period("M")
        if m not in ranks: continue
        if t0.dayofweek == 5:
            t1 = t0 + pd.Timedelta(hours=48); kind = "weekend"
        elif t0.dayofweek == 6:
            continue
        else:
            t1 = t0 + pd.Timedelta(hours=24); kind = "weekday"
        tb = t0 - pd.Timedelta(days=7)
        if t1 not in idx or tb not in idx: continue
        for grp, names in [("BTC", ["BTCUSDT"]), ("ETH", ["ETHUSDT"]), ("majors", ranks[m]["majors"]), ("alts", ranks[m]["alts"])]:
            names = [n for n in names if n in C.columns]
            mom = np.sign(C.loc[t0, names] / C.loc[tb, names] - 1).fillna(0)
            px = C.loc[t0:t1, names]
            r = mom * (px.iloc[-1] / px.iloc[0] - 1)
            fund = mom * F.loc[idx[idx > t0][0]:t1, names].sum()
            net = (r - fund - 2 * COST * (mom != 0)).mean()
            rows.append(dict(t0=t0, kind=kind, group=grp, net=net, hours=(t1 - t0).total_seconds() / 3600))
    r2 = pd.DataFrame(rows)
    r2["net_per_day"] = r2.net / (r2.hours / 24)
    print("\n=== R2: 7-day momentum sign, net % per DAY (10bp/side per position, funding): weekend (48h hold) vs weekday (24h hold)")
    t = r2.groupby(["group", "kind"]).net_per_day.agg(n="count", mean=lambda x: x.mean() * 100, t=lambda x: x.mean() / x.std() * np.sqrt(len(x)), win=lambda x: (x > 0).mean() * 100)
    print(t.round(2).to_string())
    print("R2 by year:")
    print(r2.groupby([r2.t0.dt.year, "group", "kind"]).net_per_day.agg(n="count", mean=lambda x: x.mean() * 100, t=lambda x: x.mean() / x.std() * np.sqrt(len(x))).round(2).to_string())
    r2.to_csv(f"{OUT}/r2_momentum.csv", index=False)

    # descriptive 168-cell: mean hourly return (bps) by weekday x hour, BTC and alts basket
    ret = C.pct_change()
    alt_ret = pd.Series(index=idx, dtype=float)
    for m, r in ranks.items():
        sl = (idx.to_period("M") == m)
        alt_ret[sl] = ret.loc[sl, [n for n in r["alts"] if n in ret.columns]].mean(axis=1)
    for name, s in [("BTC", ret["BTCUSDT"]), ("alts", alt_ret)]:
        tab = (s.groupby([idx.dayofweek, idx.hour]).mean() * 1e4).unstack(0)
        tab.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        tab.to_csv(f"{OUT}/hour_dow_{name}.csv")
        print(f"\n=== descriptive: {name} mean hourly return (bps) by hour (rows) x weekday; row 'sum' = daily")
        print(pd.concat([tab, tab.sum().rename("sum").to_frame().T]).round(1).to_string())

if __name__ == "__main__":
    main()
