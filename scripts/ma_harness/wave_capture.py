"""Can a rule ride the wave once the right coin has been picked?

The screening test answered the selection half: realised volatility predicts
which coins travel, with a rank IC around +0.6. This asks the other half -- on
exactly those coins, how much of the move does an exit rule actually keep, and
does any common trend rule keep more than the crossover the article uses.

Selection here is deliberately the strongest predictor found, not the article's
score, so that only the signal is under test.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/factor_research")
from research.panel import Panel  # noqa: E402

BPD = {"15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}


def sma(x, w):
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(x.size, np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def atr(high, low, close, w):
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    return pd.Series(tr).ewm(alpha=1.0 / w, adjust=False).mean().to_numpy()


def pnl(state, close, funding, fee, bar_hours, start):
    """Returns of a position path; `state` is decided at each bar's close."""
    pos = np.concatenate([[0.0], state[:-1]])
    r = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    f = np.nan_to_num(funding) * (bar_hours / 8.0)
    net = pos * r - turn * fee - pos * f
    return float(np.prod(1.0 + net[start:]) - 1.0)


def sig_cross(close, f, s):
    d = sma(close, f) - sma(close, s)
    out = np.sign(d)
    out[~np.isfinite(d)] = 0.0
    return out


def sig_donchian(high, low, close, entry, exit_):
    hh = pd.Series(high).rolling(entry).max().shift(1).to_numpy()
    ll = pd.Series(low).rolling(entry).min().shift(1).to_numpy()
    xh = pd.Series(high).rolling(exit_).max().shift(1).to_numpy()
    xl = pd.Series(low).rolling(exit_).min().shift(1).to_numpy()
    out = np.zeros(close.size)
    pos = 0.0
    for i in range(close.size):
        if np.isfinite(hh[i]):
            if close[i] > hh[i]:
                pos = 1.0
            elif close[i] < ll[i]:
                pos = -1.0
            elif pos > 0 and close[i] < xl[i]:
                pos = 0.0
            elif pos < 0 and close[i] > xh[i]:
                pos = 0.0
        out[i] = pos
    return out


def sig_atr_trail(close, high, low, entry_sig, mult, w):
    """Enter on the given signal, leave on a Chandelier-style ATR stop."""
    a = atr(high, low, close, w)
    out = np.zeros(close.size)
    pos, stop = 0.0, np.nan
    for i in range(close.size):
        if pos > 0:
            stop = max(stop, close[i] - mult * a[i]) if np.isfinite(a[i]) else stop
            if close[i] < stop:
                pos = 0.0
        elif pos < 0:
            stop = min(stop, close[i] + mult * a[i]) if np.isfinite(a[i]) else stop
            if close[i] > stop:
                pos = 0.0
        if pos == 0.0 and entry_sig[i] != 0 and (i == 0 or entry_sig[i - 1] != entry_sig[i]):
            pos = entry_sig[i]
            stop = (close[i] - mult * a[i]) if pos > 0 else (close[i] + mult * a[i])
        out[i] = pos
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--horizon-days", type=int, default=30)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--top-vol", type=int, default=30, help="movers kept")
    ap.add_argument("--fast-rules", action="store_true",
                    help="short-lookback versions of the same rule families")
    ap.add_argument("--mom-days", type=int, default=30,
                    help="lookback of the time-series momentum control")
    ap.add_argument("--sel-vol-days", type=int, default=7,
                    help="realised-volatility window used to rank movers")
    ap.add_argument("--top-n", type=int, default=150, help="liquidity universe")
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--out", default="/root/freqtrade/user_data/research/ma_harness/wave.parquet")
    a = ap.parse_args()

    p = Panel(a.tf, mmap=True)
    bpd = BPD[a.tf]
    close, high, low = p["close"], p["high"], p["low"]
    qv, fr, mask = p["quote_volume"], p["funding_rate"], p["universe_mask"]
    h = a.horizon_days * bpd
    warm = max(210, a.mom_days * bpd + 5)
    bar_hours = 24.0 / bpd

    recs = []
    for t in range(90 * bpd, p.shape[0] - h, a.step_days * bpd):
        dollar = np.nansum(qv[t - 30 * bpd:t], axis=0)
        ok = np.isfinite(close[t - 90 * bpd:t]).all(axis=0) & mask[t - 1]
        dollar = np.where(ok, dollar, -1.0)
        liq = np.argsort(-dollar)[:a.top_n]
        liq = liq[dollar[liq] > 0]
        c7 = np.asarray(close[t - a.sel_vol_days * bpd:t, liq], dtype=np.float64)
        v7 = np.nanstd(np.diff(np.log(c7), axis=0), axis=0)
        order = np.argsort(-v7)
        movers = liq[order[:a.top_vol]]
        vol_rank = {int(liq[c]): int(i) for i, c in enumerate(order)}
        for col in movers:
            cl = np.asarray(close[t - warm:t + h, col], dtype=np.float64)
            hi = np.asarray(high[t - warm:t + h, col], dtype=np.float64)
            lo = np.asarray(low[t - warm:t + h, col], dtype=np.float64)
            fu = np.asarray(fr[t - warm:t + h, col], dtype=np.float64)
            if not np.isfinite(cl).all():
                continue
            fwd = cl[warm:]
            # trend cleanliness on daily steps: at bar scale the ratio is
            # dominated by noise and reads near zero for every window
            daily = fwd[::bpd]
            r = np.diff(np.log(daily))
            up = float((fwd / np.minimum.accumulate(fwd)).max() - 1.0)
            dn = float(1.0 - (fwd / np.maximum.accumulate(fwd)).min())
            # features known at the decision bar, to ask whether the trending
            # windows can be picked in advance rather than only in hindsight
            past = cl[:warm]
            pd_daily = past[::bpd]
            pr = np.diff(np.log(pd_daily))
            past_eff = float(abs(pr.sum()) / np.abs(pr).sum()) if np.abs(pr).sum() > 0 else 0.0
            rec = dict(t=int(t), symbol=p.symbols[col],
                       vol_rank=vol_rank[int(col)],
                       past_eff=past_eff,
                       past_vol=float(np.std(np.diff(np.log(
                           past[-a.sel_vol_days * bpd:])))),
                       past_mom=float(past[-1] / past[0] - 1.0),
                       past_absfunding=float(np.nanmean(np.abs(fu[:warm]))),
                       ret=float(fwd[-1] / fwd[0] - 1.0),
                       best_swing=max(up, dn), best_up=up, best_dn=dn,
                       efficiency=float(abs(r.sum()) / np.abs(r).sum()))
            if a.fast_rules:
                # lookbacks sized for a move that is over in two or three days
                cross_slow = sig_cross(cl, 12, 48)
                rules = {
                    "cross_12_48": cross_slow,
                    "cross_6_24": sig_cross(cl, 6, 24),
                    "cross_4_12": sig_cross(cl, 4, 12),
                    "donchian_24_8": sig_donchian(cl, hi, lo, 24, 8),
                    "donchian_48_12": sig_donchian(cl, hi, lo, 48, 12),
                    "atr_trail_on_cross": sig_atr_trail(cl, hi, lo, cross_slow,
                                                        2.0, 12),
                    f"tsmom_{a.mom_days}d": np.sign(cl - np.concatenate(
                        [np.full(a.mom_days * bpd, np.nan),
                         cl[:-a.mom_days * bpd]])),
                    "long_only_cross": np.maximum(cross_slow, 0.0),
                    "buy_hold": np.ones(cl.size),
                }
                for name, st in rules.items():
                    rec[name] = pnl(np.nan_to_num(st), cl, fu, a.fee,
                                    bar_hours, warm)
                recs.append(rec)
                continue
            cross_slow = sig_cross(cl, 50, 200)
            rules = {
                "cross_50_200": cross_slow,
                "cross_20_60": sig_cross(cl, 20, 60),
                "cross_10_30": sig_cross(cl, 10, 30),
                "donchian_55_20": sig_donchian(cl, hi, lo, 55 * 1, 20),
                "atr_trail_on_cross": sig_atr_trail(cl, hi, lo, cross_slow, 3.0, 24),
                f"tsmom_{a.mom_days}d": np.sign(cl - np.concatenate(
                    [np.full(a.mom_days * bpd, np.nan),
                     cl[:-a.mom_days * bpd]])),
                "long_only_cross": np.maximum(cross_slow, 0.0),
                "buy_hold": np.ones(cl.size),
            }
            for name, st in rules.items():
                st = np.nan_to_num(st)
                rec[name] = pnl(st, cl, fu, a.fee, bar_hours, warm)
            recs.append(rec)

    df = pd.DataFrame(recs)
    df.to_parquet(a.out)
    rules = [c for c in df.columns if c not in
             ("t", "symbol", "ret", "best_swing", "best_up", "best_dn", "efficiency",
              "past_eff", "past_vol", "past_mom", "past_absfunding", "vol_rank")]
    n_t = df.t.nunique()
    print(f"{len(df)} coin-windows, {n_t} rebalances, horizon {a.horizon_days}d, "
          f"step {a.step_days}d, tf {a.tf}, movers {a.top_vol} of top {a.top_n} "
          f"by volume, ranked on {a.sel_vol_days}d volatility")
    print(f"available: mean best swing {100*df.best_swing.mean():.1f}%, "
          f"mean |ret| {100*df.ret.abs().mean():.1f}%, "
          f"mean trend efficiency {df.efficiency.mean():.3f}\n")
    print(f"{'rule':<22}{'mean':>9}{'median':>9}{'win%':>7}"
          f"{'capture':>9}{'t (by date)':>13}")
    for name in rules:
        x = df[name]
        per_date = df.groupby("t")[name].mean()
        tt = per_date.mean() / (per_date.std(ddof=1) / np.sqrt(len(per_date)))
        cap = (x / df.best_swing.replace(0, np.nan)).median()
        print(f"{name:<22}{100*x.mean():>+8.2f}%{100*x.median():>+8.2f}%"
              f"{100*(x > 0).mean():>6.0f}%{cap:>+9.3f}{tt:>+13.2f}")

    print(f"\nmedians: best swing {100*df.best_swing.median():.1f}%, "
          f"|ret| {100*df.ret.abs().median():.1f}%, "
          f"efficiency {df.efficiency.median():.3f}")
    # at a two or three day horizon the daily efficiency saturates at 1.0 and
    # the quartile edges collide, so ties are dropped rather than raising
    q = pd.qcut(df.efficiency, 4, duplicates="drop")
    q = q.cat.rename_categories(
        ["chop", "q2", "q3", "trend"][-len(q.cat.categories):])
    print("\nby trend cleanliness quartile (median return per coin-window)")
    print(f"{'rule':<22}" + "".join(f"{str(x):>10}" for x in q.cat.categories))
    for name in rules:
        line = f"{name:<22}"
        for lab in q.cat.categories:
            line += f"{100*df[q == lab][name].median():>+9.2f}%"
        print(line)
    tq = df[q == q.cat.categories[-1]]
    print(f"\ncleanest quartile ({len(tq)} windows): median best swing "
          f"{100*tq.best_swing.median():.1f}%, median |ret| "
          f"{100*tq.ret.abs().median():.1f}%, median efficiency "
          f"{tq.efficiency.median():.3f}")
    for name in rules:
        x = tq[name]
        per_date = tq.groupby("t")[name].mean()
        tt = per_date.mean() / (per_date.std(ddof=1) / np.sqrt(len(per_date)))
        print(f"  {name:<22} mean{100*x.mean():>+8.2f}%  median{100*x.median():>+8.2f}%"
              f"  win {100*(x > 0).mean():>3.0f}%  capture median "
              f"{(x / tq.best_swing).median():+.3f}  t {tt:+.2f}")
    for feat in ("past_eff", "past_vol", "past_mom", "past_absfunding"):
        by_feature(df, rules, feat)


def by_feature(df, rules, feat):
    """Do the rules pay more when a decision-time feature is high?"""
    q = pd.qcut(df[feat], 4, labels=["low", "q2", "q3", "high"], duplicates="drop")
    print(f"\nmedian return by {feat} quartile at the decision bar")
    print(f"{'rule':<22}" + "".join(f"{str(x):>10}" for x in q.cat.categories))
    for name in rules + ["efficiency", "best_swing"]:
        line = f"{name:<22}"
        for lab in q.cat.categories:
            line += f"{100 * df[q == lab][name].median():>+9.2f}%"
        print(line)


if __name__ == "__main__":
    main()
