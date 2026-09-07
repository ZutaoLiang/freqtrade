"""Score the walk-forward output of the MA screening harness.

The top-K whitelist is compared with the controls that decide whether the two
layer race adds anything: a random draw of the same size from the same
universe, the bottom of the same ranking, the whole universe, and BTC.
"""
import argparse
import json
import numpy as np
from scipy import stats

BARS_PER_DAY = {"5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}


def bucket_period_returns(steps, k, which="top", field="oos_total"):
    out = []
    for st in steps:
        rows = sorted(st["rows"], key=lambda r: r["rank"])
        sel = rows[:k] if which == "top" else rows[-k:]
        out.append(np.mean([r[field] for r in sel]))
    return np.array(out)


def curve_stats(period_ret, days_per_step, label):
    eq = np.cumprod(1.0 + period_ret)
    total = eq[-1] - 1.0
    years = len(period_ret) * days_per_step / 365.0
    cagr = eq[-1] ** (1.0 / years) - 1.0
    per_year = 365.0 / days_per_step
    sharpe = (period_ret.mean() / period_ret.std(ddof=1) * np.sqrt(per_year)
              if period_ret.std(ddof=1) > 0 else 0.0)
    mdd = float((1.0 - eq / np.maximum.accumulate(eq)).max())
    t = period_ret.mean() / (period_ret.std(ddof=1) / np.sqrt(len(period_ret)))
    return dict(label=label, total=float(total), cagr=float(cagr),
                sharpe=float(sharpe), mdd=mdd, t_stat=float(t),
                periods=len(period_ret), mean_period=float(period_ret.mean()),
                win_periods=float((period_ret > 0).mean()))


def fmt(s):
    return (f"{s['label']:<28} total {s['total']*100:8.1f}%  CAGR {s['cagr']*100:7.1f}%  "
            f"Sharpe {s['sharpe']:6.2f}  maxDD {s['mdd']*100:5.1f}%  "
            f"t {s['t_stat']:5.2f}  win {s['win_periods']*100:4.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    d = json.load(open(a.wf))
    cfg, steps = d["config"], d["steps"]
    k, tf = cfg["top_k"], cfg["tf"]
    dps = cfg["oos_days"]
    print(f"{tf}  {len(steps)} rebalances x {dps}d out of sample  "
          f"lookback {cfg['lookback_days']}d  universe {cfg['top_n']}  top-K {k}  "
          f"fee {cfg['fee']*100:.3f}%/side")

    res = []
    top = bucket_period_returns(steps, k, "top")
    res.append(curve_stats(top, dps, f"whitelist top-{k}"))
    res.append(curve_stats(bucket_period_returns(steps, k, "bottom"), dps,
                           f"bottom-{k} of same ranking"))
    allc = np.array([np.mean([r["oos_total"] for r in st["rows"]]) for st in steps])
    res.append(curve_stats(allc, dps, "whole universe, own params"))
    if "oos_total_rand" in steps[0]["rows"][0]:
        res.append(curve_stats(bucket_period_returns(steps, k, "top", "oos_total_rand"),
                               dps, f"top-{k} coins, random params"))
    if "oos_total_fixed" in steps[0]["rows"][0]:
        res.append(curve_stats(bucket_period_returns(steps, k, "top", "oos_total_fixed"),
                               dps, f"top-{k} coins, fixed 50/200"))
        allf = np.array([np.mean([r["oos_total_fixed"] for r in st["rows"]])
                         for st in steps])
        res.append(curve_stats(allf, dps, "whole universe, fixed 50/200"))

    # random draw of K coins from the same universe, using the parameters the
    # screen picked for them: isolates coin selection from parameter selection
    rng = np.random.default_rng(0)
    draws = np.empty((a.draws, len(steps)))
    for i in range(a.draws):
        for j, st in enumerate(steps):
            tot = np.array([r["oos_total"] for r in st["rows"]])
            draws[i, j] = tot[rng.choice(tot.size, k, replace=False)].mean()
    rand_total = np.prod(1.0 + draws, axis=1) - 1.0
    res.append(curve_stats(draws.mean(axis=0), dps, f"random {k} (mean of draws)"))
    for s in res:
        print(fmt(s))
    pct = float((rand_total < (np.prod(1.0 + top) - 1.0)).mean())
    print(f"\nwhitelist total return sits at the {pct*100:.1f}th percentile of "
          f"{a.draws} random draws of {k} coins from the same universe")

    # buy and hold BTC over exactly the same bars
    import sys
    sys.path.insert(0, "/root/freqtrade/user_data/factor_research")
    from research.panel import Panel
    p = Panel(tf, mmap=True)
    btc = p.symbols.index("BTCUSDT")
    bpd = BARS_PER_DAY[tf]
    per = []
    for st in steps:
        t0 = st["t"]
        t1 = min(t0 + dps * bpd, p.shape[0] - 1)
        c = np.asarray(p["close"][[t0, t1], btc], dtype=float)
        per.append(c[1] / c[0] - 1.0)
    print(fmt(curve_stats(np.array(per), dps, "buy and hold BTC")))

    # does the in-sample score rank predict the out-of-sample return at all?
    ics = []
    for st in steps:
        sc = [r["score"] for r in st["rows"]]
        ot = [r["oos_total"] for r in st["rows"]]
        if len(sc) > 10:
            ics.append(stats.spearmanr(sc, ot).statistic)
    ics = np.array(ics)
    tt = ics.mean() / (ics.std(ddof=1) / np.sqrt(ics.size))
    print(f"rank IC of score vs next-{dps}d return: mean {ics.mean():+.4f}  "
          f"t {tt:+.2f}  positive {100*(ics>0).mean():.0f}% of {ics.size} rebalances")

    half = len(top) // 2
    print(fmt(curve_stats(top[:half], dps, "  whitelist, first half")))
    print(fmt(curve_stats(top[half:], dps, "  whitelist, second half")))

    # bar level curve of the whitelist, for drawdown and Sharpe at bar scale
    bars = []
    for st in steps:
        rows = sorted(st["rows"], key=lambda r: r["rank"])[:k]
        mats = [r["oos_ret"] for r in rows if r["oos_ret"]]
        if not mats:
            continue
        n = min(len(m) for m in mats)
        bars.append(np.mean([m[:n] for m in mats], axis=0))
    if bars:
        b = np.concatenate(bars)
        eq = np.cumprod(1 + b)
        ann = np.sqrt(BARS_PER_DAY[tf] * 365)
        print(f"bar level: {len(b)} bars  total {100*(eq[-1]-1):.1f}%  "
              f"Sharpe {b.mean()/b.std(ddof=1)*ann:.2f}  "
              f"maxDD {100*(1-eq/np.maximum.accumulate(eq)).max():.1f}%")

    # how often does the winning parameter set survive to the next rebalance
    keep_c, keep_p = [], []
    for prev, cur in zip(steps, steps[1:]):
        pw = {r["symbol"]: (r["fast"], r["slow"])
              for r in sorted(prev["rows"], key=lambda r: r["rank"])[:k]}
        cw = {r["symbol"]: (r["fast"], r["slow"])
              for r in sorted(cur["rows"], key=lambda r: r["rank"])[:k]}
        keep_c.append(len(set(pw) & set(cw)) / k)
        same = [s for s in set(pw) & set(cw) if pw[s] == cw[s]]
        keep_p.append(len(same) / max(1, len(set(pw) & set(cw))))
    print(f"whitelist carry-over {100*np.mean(keep_c):.0f}% of names, "
          f"parameters unchanged on {100*np.mean(keep_p):.0f}% of the survivors")


if __name__ == "__main__":
    main()
