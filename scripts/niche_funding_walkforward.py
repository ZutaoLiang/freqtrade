"""Walk-forward the funding-skew rule: select on 3 months, test on the next 1, repeat.

Runs the whole grid offline on the path simulator, which reproduces what Freqtrade can
actually take -- one open trade per pair, `cap` concurrent slots -- so the selected fold
configs are worth spending a real backtest on. Reports each selection
rule against two baselines: the fixed default config, and the best config chosen with
hindsight on the test window itself (the ceiling any selection rule could reach). Two
selection metrics are carried because selecting on total return just buys trade count.

Both `allow_long` settings are carried through every fold so the long side is judged on
out-of-sample folds rather than on the one pooled split.
"""
import json
import sys
from itertools import product

import numpy as np
import pandas as pd

WORK = "/root/freqtrade/user_data/niche_work"
FUND = "/root/freqtrade/user_data/data/binance_public/funding"
CAP = 6
MIN_TRAIN_TRADES = 25
DEFAULT = dict(fr=40.0, streak=1, stop=15, hold=235)
GRID = [dict(fr=fr, streak=st, stop=sp, hold=h)
        for fr, st, sp, h in product((30.0, 40.0, 60.0, 80.0), (0, 1, 2), (10, 15, 25), (120, 235))]
FOLDS = [("2026-02", 3), ("2026-03", 3), ("2026-04", 3), ("2026-05", 3),
         ("2026-06", 3), ("2026-07", 3), ("2026-08", 3)]


def month_range(month, back):
    end = pd.Timestamp(month, tz="UTC") + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
    start = pd.Timestamp(month, tz="UTC")
    return start, min(end, pd.Timestamp("2026-08-17", tz="UTC")), start - pd.DateOffset(months=back)


def add_streaks(d):
    """Consecutive prior extreme settlements, one column per candidate threshold."""
    thresholds = sorted({c["fr"] for c in GRID})
    out = []
    for sym, g in d.groupby("sym", sort=False):
        fr = pd.read_parquet(f"{FUND}/{sym.replace('_USDT_USDT', 'USDT')}.parquet")
        fr = fr.set_index("date")["funding_rate"] * 1e4
        g = g.copy()
        for th in thresholds:
            state = np.sign(fr) * (fr.abs() >= th)
            run = state.groupby((state != state.shift()).cumsum()).cumcount()
            g[f"streak_{th:g}"] = g.stamp.map(run)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def portfolio(df, hold, cap=CAP):
    """Chronological fill: one open trade per pair, at most `cap` at once."""
    open_until, keep = {}, []
    for r in df.sort_values("date").itertuples():
        for k in [k for k, v in open_until.items() if v <= r.date]:
            del open_until[k]
        if len(open_until) >= cap or r.sym in open_until:
            continue
        open_until[r.sym] = r.date + pd.Timedelta(minutes=hold + 1)
        keep.append(r.Index)
    return df.loc[keep]


def evaluate(d, cfg, allow_long, lo, hi):
    col = f"h{cfg['hold']}_s{cfg['stop']}"
    sel = d[(d.date >= lo) & (d.date < hi)
            & (d.fr_bps.abs() >= cfg["fr"])
            & (d[f"streak_{cfg['fr']:g}"] >= cfg["streak"])]
    if not allow_long:
        sel = sel[sel.side < 0]
    t = portfolio(sel, cfg["hold"])
    if len(t) == 0:
        return 0, 0.0, 0.0
    return len(t), t[col].sum(), t[col].mean()


def main():
    d = pd.read_parquet(f"{WORK}/path_sim.parquet")
    d["stamp"] = d.date - pd.Timedelta(minutes=1)
    d = add_streaks(d)
    d = d[d.date >= "2025-08-01"].reset_index(drop=True)
    print(f"events: {len(d)}  {d.date.min():%Y-%m-%d} .. {d.date.max():%Y-%m-%d}\n")

    rows = []
    for allow_long in (False, True):
        tag = "long+short" if allow_long else "short only"
        print(f"===== allow_long = {allow_long}  ({tag}) =====")
        for month, back in FOLDS:
            lo, hi, tr_lo = month_range(month, back)
            scored = [(evaluate(d, c, allow_long, tr_lo, lo), c) for c in GRID]
            ok = [(s, c) for s, c in scored if s[0] >= MIN_TRAIN_TRADES]
            picks = {
                "by_total": max(ok, key=lambda x: x[0][1])[1] if ok else DEFAULT,
                "by_mean": max(ok, key=lambda x: x[0][2])[1] if ok else DEFAULT,
            }
            res = {k: evaluate(d, c, allow_long, lo, hi) for k, c in picks.items()}
            dn, dtot, dmean = evaluate(d, DEFAULT, allow_long, lo, hi)
            best = max((evaluate(d, c, allow_long, lo, hi) + (c,) for c in GRID),
                       key=lambda x: x[1])
            rows.append(dict(allow_long=allow_long, month=month,
                             pick_total=json.dumps(picks["by_total"]),
                             pick_mean=json.dumps(picks["by_mean"]),
                             n=res["by_total"][0], tot=res["by_total"][1],
                             mean_n=res["by_mean"][0], mean_tot=res["by_mean"][1],
                             def_n=dn, def_tot=dtot, def_mean_bps=1e4 * dmean,
                             ceil_tot=best[1]))
            pm = picks["by_mean"]
            print(f"  {month} | by_total n={res['by_total'][0]:3d} tot={100*res['by_total'][1]:+7.1f}%"
                  f" | by_mean fr={pm['fr']:.0f} s={pm['streak']} st=-{pm['stop']}% h={pm['hold']}m"
                  f" n={res['by_mean'][0]:3d} tot={100*res['by_mean'][1]:+7.1f}%"
                  f" | default n={dn:3d} tot={100*dtot:+7.1f}% | ceiling {100*best[1]:+7.1f}%")
        sub = [r for r in rows if r["allow_long"] == allow_long]
        for lab, tk, nk in (("select by total", "tot", "n"), ("select by mean ", "mean_tot", "mean_n"),
                            ("fixed default  ", "def_tot", "def_n"), ("hindsight ceil ", "ceil_tot", None)):
            print(f"  ---- {lab}: {100*sum(r[tk] for r in sub):+8.1f}% of one stake"
                  f"   months positive {sum(r[tk] > 0 for r in sub)}/{len(sub)}"
                  + (f"   trades {sum(r[nk] for r in sub)}" if nk else ""))
        print()

    pd.DataFrame(rows).to_csv(f"{WORK}/walkforward.csv", index=False)


if __name__ == "__main__":
    main()
