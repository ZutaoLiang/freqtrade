"""The multi-timeframe moving-average score from the FMZ rotation strategy.

Source: https://zhuanlan.zhihu.com/p/1999433049788654396 (2026-01-27), an
"AI rotation" workflow whose first layer ranks every liquid perpetual by a
composite of three moving-average readings, then longs the top five and shorts
the bottom five. Its news layer and its large-model layer cannot be replayed
offline, so only the ranking layer is tested here -- which is also the layer
that claims to know direction.

The article's own formulas, kept verbatim:

  arrangement  four MAs from four timeframes; perfect bull order +4, perfect
               bear order -4, two adjacent agreements +3, two scattered +2
  timeSeries   (rising MAs) - (falling MAs), so +4 down to -4
  gap          signed widening of the ribbon; positive is upward expansion
  score        gap * arrangement * timeSeries          when gap > 0
               gap * |arrangement| * |timeSeries|      when gap < 0

Assumptions it does not state: the four timeframes are wheelPeriod/4, /2, *2,
*4 as in its code, realised on the 1h panel as scaled bar counts; the ribbon
width is (short MA - long MA) / long MA and the gap is its change over
`--gap-bars`; a moving average counts as rising if it is above its value one
bar earlier.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/root/freqtrade/user_data/factor_research")
from research.panel import Panel  # noqa: E402


def sma(a, w):
    """Column-wise simple moving average of a (time, symbol) matrix."""
    c = np.cumsum(np.nan_to_num(a), axis=0)
    n = np.cumsum(np.isfinite(a), axis=0)
    out = np.full_like(a, np.nan)
    out[w - 1:] = (c[w - 1:] - np.vstack([np.zeros((1, a.shape[1])), c[:-w]])) / w
    valid = np.full_like(a, False, dtype=bool)
    valid[w - 1:] = (n[w - 1:] - np.vstack([np.zeros((1, a.shape[1])), n[:-w]])) == w
    out[~valid] = np.nan
    return out


def score_matrix(close, mean_period, wheel, gap_bars):
    """The article's composite score for every bar and symbol."""
    # wheelPeriod/4, /2, *2, *4 expressed in 1h bars
    mults = (wheel / 4.0, wheel / 2.0, wheel * 2.0, wheel * 4.0)
    wins = [max(2, int(round(mean_period * m))) for m in mults]
    mas = [sma(close, w) for w in wins]
    s, ms, ml, ls = mas

    c1, c2, c3 = s - ms, ms - ml, ml - ls
    bull = (c1 > 0).astype(int) + (c2 > 0).astype(int) + (c3 > 0).astype(int)
    adj_bull = ((c1 > 0) & (c2 > 0)) | ((c2 > 0) & (c3 > 0))
    adj_bear = ((c1 < 0) & (c2 < 0)) | ((c2 < 0) & (c3 < 0))
    arrangement = np.select(
        [bull == 3, bull == 0, bull == 2, bull == 1],
        [4.0, -4.0, np.where(adj_bull, 3.0, 2.0), np.where(adj_bear, -3.0, -2.0)],
        default=0.0)

    rising = np.zeros_like(close)
    for m in mas:
        prev = np.vstack([np.full((1, close.shape[1]), np.nan), m[:-1]])
        rising += np.where(m > prev, 1.0, np.where(m < prev, -1.0, 0.0))
    time_series = rising                                   # +4 .. -4

    spread = (s - ls) / ls
    prev_spread = np.vstack([np.full((gap_bars, close.shape[1]), np.nan),
                             spread[:-gap_bars]])
    gap = spread - prev_spread

    score = np.where(gap > 0, gap * arrangement * time_series,
                     gap * np.abs(arrangement) * np.abs(time_series))
    # As published, the gap > 0 branch multiplies two signed terms, so a
    # perfectly bearish ribbon (-4 x -4) scores the same as a perfectly bullish
    # one and lands in the long basket. The corrected variant keeps the same
    # three readings but signs the score by the arrangement.
    strength = np.abs(gap) * np.abs(arrangement) * np.abs(time_series)
    fixed = np.sign(arrangement) * strength
    bad = ~np.isfinite(arrangement) | ~np.isfinite(gap) | ~np.isfinite(ls)
    score[bad] = np.nan
    fixed[bad] = np.nan
    return score, arrangement, time_series, gap, fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--mean-period", type=int, default=20)
    ap.add_argument("--wheel", type=float, default=1.0,
                    help="wheelPeriod in units of the panel timeframe")
    ap.add_argument("--gap-bars", type=int, default=20)
    ap.add_argument("--rebalance-hours", type=int, default=4)
    ap.add_argument("--top-n", type=int, default=150, help="liquidity universe")
    ap.add_argument("--k", type=int, default=5, help="longs, and shorts")
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--sign-fix", action="store_true",
                    help="sign the score by the arrangement, as it was meant")
    ap.add_argument("--reverse", action="store_true",
                    help="long the bottom and short the top instead")
    ap.add_argument("--trail", type=float, default=0.0,
                    help="trailing stop on peak profit, e.g. 0.015")
    ap.add_argument("--out", default="/root/freqtrade/user_data/research/ma_harness/rotation.parquet")
    a = ap.parse_args()

    p = Panel(a.tf, mmap=True)
    close = np.asarray(p["close"], dtype=np.float64)
    qv, mask, fr = p["quote_volume"], p["universe_mask"], p["funding_rate"]
    score, arrangement, ts, gap, fixed = score_matrix(close, a.mean_period,
                                                      a.wheel, a.gap_bars)
    if a.sign_fix:
        score = fixed
    bpd = 24
    step = a.rebalance_hours
    start = 90 * bpd

    rows, leg_ret = [], []
    held_long, held_short = set(), set()
    for t in range(start, close.shape[0] - step, step):
        dollar = np.nansum(qv[t - 30 * bpd:t], axis=0)
        ok = np.isfinite(close[t - 90 * bpd:t]).all(axis=0) & mask[t - 1] \
            & np.isfinite(score[t - 1])
        dollar = np.where(ok, dollar, -1.0)
        liq = np.argsort(-dollar)[:a.top_n]
        liq = liq[dollar[liq] > 0]
        if liq.size < 4 * a.k:
            continue
        sc = score[t - 1, liq]                    # decided on the closed bar
        if a.reverse:
            sc = -sc
        order = np.argsort(-sc)
        longs, shorts = liq[order[:a.k]], liq[order[-a.k:]]
        fwd = close[t:t + step + 1]
        seg_f = np.nan_to_num(np.asarray(fr[t:t + step + 1], dtype=np.float64))

        def leg(cols, side, held):
            """Gross returns per name, plus the names still held at the end.

            A coin carried from the previous rebalance pays no fee; only the
            names that change hands do, which is what the rotation actually
            costs. A trailing stop that fires ends that name early and it has
            to be re-entered if the screen still wants it.
            """
            out, survive = [], set()
            for c in cols:
                path = fwd[:, c]
                if not np.isfinite(path).all():
                    continue
                r = side * (path / path[0] - 1.0)
                if a.trail > 0:
                    peak = np.maximum.accumulate(r)
                    hit = np.flatnonzero((peak - r) >= a.trail)
                    i = hit[0] if hit.size else len(r) - 1
                else:
                    i = len(r) - 1
                stopped = a.trail > 0 and i < len(r) - 1
                fund = side * seg_f[:i + 1, c].sum() / 8.0
                fee = a.fee * ((1 if int(c) not in held else 0)
                               + (1 if stopped else 0))
                out.append(float(r[i] - fee - fund))
                if not stopped:
                    survive.add(int(c))
            return out, survive

        lr, new_long = leg(longs, 1.0, held_long)
        sr, new_short = leg(shorts, -1.0, held_short)
        if not lr or not sr:
            continue
        # names dropped at this rebalance pay their exit fee here
        drop = (len(held_long - new_long) + len(held_short - new_short))
        exit_cost = a.fee * drop / (2.0 * a.k)
        held_long, held_short = new_long, new_short
        rows.append(dict(t=t, long=float(np.mean(lr)), short=float(np.mean(sr)),
                         both=float(0.5 * np.mean(lr) + 0.5 * np.mean(sr))
                         - exit_cost,
                         n_long=len(lr), n_short=len(sr)))
        # rank IC of the score against the return it is supposed to predict
        fut = fwd[-1, liq] / fwd[0, liq] - 1.0
        m = np.isfinite(sc) & np.isfinite(fut)
        if m.sum() > 20:
            leg_ret.append(stats.spearmanr(sc[m], fut[m]).statistic)

    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    per_year = 365 * 24 / step
    print(f"{len(df)} rebalances every {step}h, universe {a.top_n}, "
          f"k={a.k} per side, mean period {a.mean_period}, wheel {a.wheel}, "
          f"gap {a.gap_bars} bars, fee {a.fee*100:.3f}%/side"
          + (f", trailing stop {a.trail*100:.1f}%" if a.trail else ""))
    print(f"{'leg':<12}{'total':>10}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>8}"
          f"{'win%':>7}{'t':>7}")
    for col in ("long", "short", "both"):
        x = df[col].to_numpy()
        eq = np.cumprod(1 + x)
        years = len(x) / per_year
        mdd = float((1 - eq / np.maximum.accumulate(eq)).max())
        sh = x.mean() / x.std(ddof=1) * np.sqrt(per_year)
        tt = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        print(f"{col:<12}{100*(eq[-1]-1):>+9.1f}%{100*(eq[-1]**(1/years)-1):>+8.1f}%"
              f"{sh:>8.2f}{100*mdd:>7.1f}%{100*(x>0).mean():>6.0f}%{tt:>+7.2f}")
    ic = np.array(leg_ret)
    print(f"rank IC of the score vs the next {step}h return: {ic.mean():+.4f}  "
          f"t {ic.mean()/(ic.std(ddof=1)/np.sqrt(ic.size)):+.2f}  "
          f"positive {100*(ic>0).mean():.0f}% of {ic.size} rebalances")


if __name__ == "__main__":
    main()
