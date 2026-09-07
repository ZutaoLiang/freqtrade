"""Would the tuning have been available in advance?

Round three found a plateau of breakout settings on the whole sample. That is
hindsight. This picks the setting the same way a live operator would: every
month, score the grid on the trailing window only, trade next month with the
winner, and never look forward. The result is compared with the single best
in-sample config and with the median of the grid, which is what an operator who
refused to tune would have got.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/root/freqtrade/scripts/ma_harness")
import trend_lab as tl  # noqa: E402

BPD = 24


def grid():
    out = []
    for entry in (48, 72, 96, 120, 144, 168):
        for ex in (6, 12, 24):
            for m in (2.0, 2.5):
                out.append(dict(kind="donchian", entry=entry, exit=ex,
                                atr_mult=m, vol_target=0.6,
                                label=f"{entry}/{ex}+{m:g}ATR"))
    return out


def sharpe(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(24 * 365)) if x.std(ddof=1) else 0.0


def stats(x, label):
    eq = np.cumprod(1 + x)
    years = len(x) / (24 * 365)
    mdd = float((1 - eq / np.maximum.accumulate(eq)).max())
    return (f"{label:<28}{100*(eq[-1]-1):>+8.1f}%{100*(eq[-1]**(1/years)-1):>+8.1f}%"
            f"{sharpe(x):>8.2f}{100*mdd:>7.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=180)
    ap.add_argument("--trade-days", type=int, default=30)
    ap.add_argument("--fee", type=float, default=0.0005)
    a = ap.parse_args()

    sel = tl.selection_mask(3, 3, top_vol=30)
    cfgs = grid()
    curves = np.vstack([tl.run(c, sel, a.fee) for c in cfgs])
    live = np.flatnonzero(sel.any(axis=1))
    t0, t1 = int(live[0]), int(live[-1])
    train, trade = a.train_days * BPD, a.trade_days * BPD

    picked, chosen = [], []
    for t in range(t0 + train, t1, trade):
        sh = np.array([sharpe(curves[i, t - train:t]) for i in range(len(cfgs))])
        best = int(np.argmax(sh))
        picked.append(curves[best, t:t + trade])
        chosen.append(cfgs[best]["label"])
    wf = np.concatenate(picked)
    span = slice(t0 + train, t0 + train + len(wf))

    print(f"{len(cfgs)} configs, retune every {a.trade_days}d on the trailing "
          f"{a.train_days}d, {len(chosen)} decisions, fee {a.fee*100:.3f}%/side")
    print(f"{'':<28}{'total':>9}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}")
    print(stats(wf, "walk-forward pick"))
    fixed = [i for i, c in enumerate(cfgs) if c["label"] == "96/12+2ATR"][0]
    print(stats(curves[fixed, span], "fixed 96/12+2ATR"))
    med = np.median(curves[:, span], axis=0)
    print(stats(med, "median of the grid"))
    print(stats(curves[:, span].mean(axis=0), "equal weight over grid"))
    uniq = {}
    for c in chosen:
        uniq[c] = uniq.get(c, 0) + 1
    print("\npicks:", ", ".join(f"{k} x{v}" for k, v in
                                sorted(uniq.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
