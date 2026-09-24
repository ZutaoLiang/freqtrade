"""Stricter multi-indicator BTC bear regimes: grid over stops, with funding broken out."""
import itertools, os
import numpy as np
import pandas as pd
from backtest import load, run, period_table, regime_series, PERIODS, HERE

REGIMES = ["sma50", "s_ma3", "s_ma3_slope", "s_ma3_slope_ret", "s_6of8", "s_7of8", "s_all8"]
STOPS = [None, 0.10, 0.15, 0.20, 0.30]

d = load()
lo, hi = d["grid"][0].normalize(), d["grid"][-1].normalize()
print("regime days on / episodes (2022-11..2026-08):")
for r in REGIMES:
    on = regime_series(d["btc"], r)[lo:hi].astype(bool)
    eps = int((on & ~on.shift(1, fill_value=False)).sum())
    by = on.groupby(on.index.year).sum().to_dict()
    print(f"  {r:16s} {int(on.sum()):4d} days, {eps:3d} episodes, by year {by}")

rows = []
for reg, st in itertools.product(REGIMES, STOPS):
    res = run(d, reg, st)
    res_nf = run(d, reg, st, use_funding=0.0)
    tab, tab_nf = period_table(res), period_table(res_nf)
    for p in tab.index:
        rows.append(dict(regime=reg, stop=st or "none", period=p, **tab.loc[p].to_dict(),
                         ret_no_funding=tab_nf.loc[p, "ret"], stops_hit=res["stops"]))
    a = tab.loc["ALL"]
    yr = " ".join(f"{p[:4]}:{tab.loc[p, 'ret']:+.1%}" for p in PERIODS)
    print(f"{reg:16s} stop={str(st or 'none'):5s} ret {a.ret:+7.1%} (no-fund {tab_nf.loc['ALL','ret']:+7.1%}) "
          f"funding {a.funding:+.1%} fees {a.fees:+.1%} sharpe {a.sharpe:+.2f} maxDD {a.maxdd:.1%} "
          f"in_mkt {a.in_mkt:.0%} stops {res['stops']:4d} | {yr}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "grid_strict.csv"), index=False)
