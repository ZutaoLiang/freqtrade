"""Purged block k-fold on the vol-screen Donchian family.

The parameters in `skills/vol-screen-trend-lab-20260906.md` were chosen while
looking at the whole sample. This asks the only question that matters about
that: had the choice been made without seeing a block of time, would the config
it picked still have worked on that block?

Random k-fold would leak: the screen looks back 3 days and the entry channel 96
hours, so a training bar adjacent to a test bar shares its information. Blocks
are therefore contiguous and an embargo is cut out of the training set on both
sides of the test block.

Selection uses the panel engine because it scores a config in seconds. That
engine is optimistic on shorts -- it holds constant notional, so a short in a
collapsing coin compounds past the 100% a fixed short can earn -- but every
config in the comparison carries the same bias, which is what a selection study
needs. Absolute numbers still belong to freqtrade.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/scripts/ma_harness")
import trend_lab as tl  # noqa: E402

BPD = 24


def grid():
    out = []
    for entry in (48, 72, 96, 120, 144, 168):
        for ex in (6, 12, 24):
            for m in (2.0, 2.5):
                out.append(dict(kind="donchian", entry=entry, exit=ex, atr_mult=m,
                                vol_target=0.6, fund_abs_pctile=0.8,
                                label=f"{entry}/{ex}+{m:g}ATR"))
    return out


def _curve(args):
    cfg, sel, fee = args
    return cfg["label"], tl.run(cfg, sel, fee)


def sharpe(x):
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(24 * 365)) if s > 0 else 0.0


def total(x):
    return float(np.prod(1.0 + x) - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo-days", type=int, default=7)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="/root/freqtrade/user_data/research/ma_harness/kfold.json")
    a = ap.parse_args()

    sel = tl.selection_mask(3, 3, top_vol=30)
    cfgs = grid()
    print(f"{len(cfgs)} configs, {a.folds} folds, embargo {a.embargo_days}d", flush=True)
    curves = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for label, c in ex.map(_curve, [(c, sel, a.fee) for c in cfgs]):
            curves[label] = c
    labels = [c["label"] for c in cfgs]
    mat = np.vstack([curves[l] for l in labels])

    live = np.flatnonzero(sel.any(axis=1))
    lo, hi = int(live[0]), int(live[-1]) + 1
    bounds = np.linspace(lo, hi, a.folds + 1).astype(int)
    emb = a.embargo_days * BPD
    p = tl.load()["p"]

    rows = []
    for i in range(a.folds):
        t0, t1 = bounds[i], bounds[i + 1]
        test = np.zeros(mat.shape[1], dtype=bool)
        test[t0:t1] = True
        train = np.zeros_like(test)
        train[lo:hi] = True
        train[max(lo, t0 - emb):min(hi, t1 + emb)] = False

        tr_sharpe = np.array([sharpe(mat[j][train]) for j in range(len(labels))])
        pick = int(np.argmax(tr_sharpe))
        te = {l: sharpe(mat[j][test]) for j, l in enumerate(labels)}
        te_tot = {l: total(mat[j][test]) for j, l in enumerate(labels)}
        rows.append(dict(
            fold=i + 1,
            start=str(p.index[t0].date()), end=str(p.index[t1 - 1].date()),
            days=int((t1 - t0) / BPD),
            picked=labels[pick],
            train_sharpe=float(tr_sharpe[pick]),
            test_sharpe=float(te[labels[pick]]),
            test_total=float(te_tot[labels[pick]]),
            fixed_sharpe=float(te["96/12+2ATR"]),
            fixed_total=float(te_tot["96/12+2ATR"]),
            median_sharpe=float(np.median([te[l] for l in labels])),
            best_possible=float(max(te.values())),
            worst_possible=float(min(te.values())),
            rank_of_pick=int(1 + sum(1 for l in labels if te[l] > te[labels[pick]])),
        ))
        print(f"  fold {i+1} {rows[-1]['start']}..{rows[-1]['end']} "
              f"picked {rows[-1]['picked']}", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n{'fold':<5}{'区间':<24}{'训练选出':<16}{'训练Sharpe':>11}"
          f"{'测试Sharpe':>11}{'测试收益':>10}{'固定配置':>10}{'网格中位':>10}{'排名':>6}")
    for r in rows:
        print(f"{r['fold']:<5}{r['start']+'..'+r['end']:<24}{r['picked']:<16}"
              f"{r['train_sharpe']:>11.2f}{r['test_sharpe']:>11.2f}"
              f"{100*r['test_total']:>9.1f}%{r['fixed_sharpe']:>10.2f}"
              f"{r['median_sharpe']:>10.2f}{r['rank_of_pick']:>4}/{len(labels)}")
    print(f"\n{'平均':<29}{'':<16}{df.train_sharpe.mean():>11.2f}"
          f"{df.test_sharpe.mean():>11.2f}{100*df.test_total.mean():>9.1f}%"
          f"{df.fixed_sharpe.mean():>10.2f}{df.median_sharpe.mean():>10.2f}")
    print(f"\n测试集为正的折数: 选出的配置 {int((df.test_sharpe>0).sum())}/{a.folds}，"
          f"固定配置 {int((df.fixed_sharpe>0).sum())}/{a.folds}，"
          f"网格中位 {int((df.median_sharpe>0).sum())}/{a.folds}")
    print(f"选出的配置在测试集里的平均排名 {df.rank_of_pick.mean():.1f} / {len(labels)}"
          f"（随机猜测的期望是 {(len(labels)+1)/2:.1f}）")
    json.dump(rows, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()


def variance_report(mat, labels, bounds, folds):
    """How much of the out-of-sample spread is the period, and how much the config?

    If the period dominates, tuning is beside the point: the same parameters
    would have been judged brilliant or broken depending only on which block
    they were tested in.
    """
    rows = []
    for i in range(folds):
        t0, t1 = bounds[i], bounds[i + 1]
        for j, lab in enumerate(labels):
            x = mat[j][t0:t1]
            s = x.std(ddof=1)
            rows.append(dict(fold=i + 1, config=lab,
                             sharpe=float(x.mean() / s * np.sqrt(24 * 365)) if s > 0 else 0.0))
    df = pd.DataFrame(rows)
    grand = df.sharpe.mean()
    ss_total = ((df.sharpe - grand) ** 2).sum()
    ss_fold = df.groupby("fold").sharpe.transform("mean").sub(grand).pow(2).sum()
    ss_cfg = df.groupby("config").sharpe.transform("mean").sub(grand).pow(2).sum()
    print(f"\n测试集 Sharpe 的方差分解（{len(labels)} 配置 x {folds} 折 = {len(df)} 个观测）")
    print(f"  时间段（折）解释 {100*ss_fold/ss_total:.1f}%")
    print(f"  参数配置解释   {100*ss_cfg/ss_total:.1f}%")
    print(f"  残差           {100*(ss_total-ss_fold-ss_cfg)/ss_total:.1f}%")
    print(f"  每折内部的配置间 Sharpe 极差 中位 "
          f"{df.groupby('fold').sharpe.agg(lambda x: x.max()-x.min()).median():.2f}")
    print(f"  每个配置跨折的 Sharpe 极差 中位 "
          f"{df.groupby('config').sharpe.agg(lambda x: x.max()-x.min()).median():.2f}")
    return df
