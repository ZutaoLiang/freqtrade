"""Should the hold shorten on hourly-funded pairs?

Binance moves a perp to 1h funding exactly when it is in the squeeze this strategy trades,
so the 225-minute hold -- which was sized to clear the 4h schedule -- crosses three hourly
settlements and pays the same extreme rate that triggered the entry. This prices the
trade-off directly: price outcome and funding cost, per hold length, split by the pair's
settlement interval at the time of the event.

Entry is at T+5m and the stop is checked on 5m closes, matching FundingSkewMomentum5m.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

KL = "/root/freqtrade/user_data/data/binance_public/freqtrade/futures"
FUND = "/root/freqtrade/user_data/data/binance_public/funding"
EVENTS = "/root/freqtrade/user_data/niche_work/universe_events.parquet"
OUT = "/root/freqtrade/user_data/niche_work/hold_sweep.parquet"
HOLDS = (225, 340, 470, 700)
STOP = 0.15
COST = 0.0020
LIQ = 0.99
ENTRY_OFFSET = 5


def _run(args):
    sym, ev = args
    fp = f"{FUND}/{sym.replace('_USDT_USDT', 'USDT')}.parquet"
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_feather(f"{KL}/{sym}-1m-futures.feather").set_index("date")
        df = df[~df.index.duplicated()]
        fr = pd.read_parquet(fp).set_index("date")["funding_rate"]
        fr = fr[~fr.index.duplicated()].sort_index()
    except Exception:
        return None
    opens, closes, idx = df["open"].values, df["close"].values, df.index

    rows = []
    for e in ev.itertuples():
        stamp = e.date - pd.Timedelta(minutes=1)          # events store the T+1m entry bar
        p = idx.searchsorted(stamp + pd.Timedelta(minutes=ENTRY_OFFSET))
        if p >= len(idx) - max(HOLDS) - 2 or idx[p] != stamp + pd.Timedelta(minutes=ENTRY_OFFSET):
            continue
        # settlement interval in force, from the gap to the neighbouring settlements
        nxt = fr.index[fr.index > stamp]
        if len(nxt) == 0:
            continue
        interval_h = (nxt[0] - stamp).total_seconds() / 3600
        entry = opens[p]
        seg = e.side * (closes[p:p + max(HOLDS) + 1] / entry - 1.0)
        nx = e.side * (opens[p + 1:p + max(HOLDS) + 2] / entry - 1.0)
        rec = dict(sym=sym, date=e.date, side=e.side, fr_bps=e.fr_bps,
                   interval_h=interval_h, trail_qv=e.trail_qv)
        for hold in HOLDS:
            checks = np.arange(4, hold + 1, 5)            # 5m closes
            hit = checks[seg[checks] <= -STOP]
            stopped = len(hit) > 0
            price = nx[hit[0]] if stopped else seg[hold]
            exit_at = stamp + pd.Timedelta(minutes=ENTRY_OFFSET + (hit[0] + 1 if stopped else hold))
            paid = fr[(fr.index > stamp) & (fr.index <= exit_at)]
            # short receives a positive rate and pays a negative one; long is the mirror
            funding = -e.side * paid.sum()
            rec[f"h{hold}"] = max(price, -LIQ) + funding - COST
            rec[f"f{hold}"] = funding
        rows.append(rec)
    return pd.DataFrame(rows) if rows else None


def main():
    d = pd.read_parquet(EVENTS)
    d = d[(d.trail_qv >= 1e7) & (d.fr_bps.abs() >= 40) & (d.streak_40 >= 1)]
    out = []
    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as ex:
        for r in ex.map(_run, [(s, g) for s, g in d.groupby("sym")], chunksize=2):
            if r is not None:
                out.append(r)
    r = pd.concat(out, ignore_index=True)
    r.to_parquet(OUT)
    r["bucket"] = np.where(r.interval_h <= 1.01, "1h 结算", "4h/8h 结算")
    r["q"] = pd.PeriodIndex(r.date, freq="Q").astype(str)
    print(f"n={len(r)}   1h 结算 {int((r.bucket == '1h 结算').sum())}   "
          f"4h/8h {int((r.bucket != '1h 结算').sum())}\n")
    for b, g in r.groupby("bucket"):
        print(f"=== {b}  n={len(g)} ===")
        print(f"{'持仓':>6} {'净单笔':>10} {'其中资金费':>11} {'中位':>9} {'为正季度':>9}")
        for hold in HOLDS:
            byq = g.groupby("q")[f"h{hold}"].sum()
            print(f"{hold:>4}m  {1e4*g[f'h{hold}'].mean():>+9.1f}bps {1e4*g[f'f{hold}'].mean():>+10.1f}bps "
                  f"{1e4*g[f'h{hold}'].median():>+8.1f} {(byq > 0).sum():>6}/{len(byq)}")
        print()
    print("=== 组合:1h 用短持仓、其余用 225m ===")
    one = r[r.bucket == "1h 结算"]
    rest = r[r.bucket != "1h 结算"]
    for hold in HOLDS:
        tot = one[f"h{hold}"].sum() + rest["h225"].sum()
        n = len(r)
        print(f"  1h 段用 {hold:>3}m: 合计 {100*tot:>+8.1f}%   单笔均值 {1e4*tot/n:>+7.1f}bps")


if __name__ == "__main__":
    main()
