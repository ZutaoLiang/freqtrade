"""Which variants survive a bull market as well as a bear one?

The base rule earns its whole record in 2025 and loses in 2023 and 2024, so the
selection criterion here is not the total: it is whether a variant is positive
in every year. Anything that only lifts the total is just re-fitting the regime
that already worked.

Hypotheses under test, in the order the diagnosis suggested them:

  funding side   in a bull the whole market pays positive funding, so a gate on
                 |funding| picks the most crowded longs and the breakout then
                 buys the side that pays. Conditioning direction on the funding
                 sign should matter far more in 2023-24 than in 2025.
  exit width     tight exits suited 2025's chop; a trending alt pulls back hard
                 and a 12h channel or a 2xATR trail may cut every winner.
  regime gate    align the tradable side with BTC's own trend, or against it.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/scripts/ma_harness")
import trend_lab as tl  # noqa: E402


def variants():
    base = dict(kind="donchian", entry=96, exit=12, atr_mult=2.0,
                vol_target=0.6, fund_abs_pctile=0.8)
    v = [("基准 96/12+2ATR+门槛80", base)]
    v.append(("+ 只逆资金费方向", {**base, "fund_side": "contrarian"}))
    v.append(("去掉资金费门槛", {k: x for k, x in base.items() if k != "fund_abs_pctile"}))
    v.append(("去门槛 + 逆资金费", {**{k: x for k, x in base.items()
                                      if k != "fund_abs_pctile"},
                                   "fund_side": "contrarian"}))
    v.append(("放宽出场 24 + 3ATR", {**base, "exit": 24, "atr_mult": 3.0}))
    v.append(("放宽出场 48 + 4ATR", {**base, "exit": 48, "atr_mult": 4.0}))
    v.append(("慢入场 168/24 + 3ATR", {**base, "entry": 168, "exit": 24, "atr_mult": 3.0}))
    v.append(("只做多", {**base, "long_only": True}))
    v.append(("顺 BTC 200h 均线", {**base, "btc_filter": 200}))
    v.append(("顺 BTC 500h 均线", {**base, "btc_filter": 500}))
    v.append(("逆资金费 + 放宽出场", {**base, "fund_side": "contrarian",
                                    "exit": 24, "atr_mult": 3.0}))
    v.append(("逆资金费 + 顺 BTC 500h", {**base, "fund_side": "contrarian",
                                       "btc_filter": 500}))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--years", default="2023,2024,2025")
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(",")]

    sel = tl.selection_mask(3, 3, top_vol=30)
    idx = pd.DatetimeIndex(tl.load()["p"].index)
    print(f"{'变体':<26}" + "".join(f"{y:>10}" for y in years)
          + f"{'三年合计':>10}{'最差年':>9}{'全正':>6}")
    rows = []
    for name, cfg in variants():
        port = tl.run({**cfg, "label": name}, sel, a.fee)
        per = []
        for y in years:
            m = idx.year == y
            per.append(float(np.prod(1.0 + port[m]) - 1.0) if m.sum() > 24 * 30 else np.nan)
        allm = np.isin(idx.year, years)
        tot = float(np.prod(1.0 + port[allm]) - 1.0)
        ok = "是" if all(p > 0 for p in per) else ""
        rows.append((name, per, tot, min(per), ok))
        print(f"{name:<26}" + "".join(f"{100*p:>+9.1f}%" for p in per)
              + f"{100*tot:>+9.1f}%{100*min(per):>+8.1f}%{ok:>6}")
    return rows


if __name__ == "__main__":
    main()
