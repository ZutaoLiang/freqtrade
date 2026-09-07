"""Once a breakout is live, can the wave be recognised while it runs?

Nothing known at the decision bar predicts whether a mover will trend or chop.
This asks the weaker but tradable question: after entry, does an open profit
say anything about what follows -- and which exit keeps the most of it.

Trades are Donchian(55) breakouts on the same movers the wave test used, one
position per coin at a time, and every exit rule is scored on the same entries.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/factor_research")
from research.panel import Panel  # noqa: E402

BPD = {"15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}


def exits(path, side, fee, funding, bar_hours, high, low, atr_series):
    """Return of one trade under each exit rule; `path` starts at entry bar."""
    r = side * (path / path[0] - 1.0)          # running paper profit, fraction
    peak = np.maximum.accumulate(r)
    n = path.size
    out = {}

    def finish(i):
        i = min(i, n - 1)
        gross = side * (path[i] / path[0] - 1.0)
        hold = np.arange(i + 1)
        f = np.nan_to_num(funding[:i + 1]).sum() * (bar_hours / 8.0) * side
        return float(gross - 2.0 * fee - f), int(i)

    out["hold_to_end"] = finish(n - 1)
    for tp in (0.10, 0.20, 0.50):
        hit = np.flatnonzero(r >= tp)
        out[f"take_profit_{int(tp*100)}"] = finish(hit[0] if hit.size else n - 1)
    for tr in (0.05, 0.10, 0.20):
        give = np.flatnonzero((peak >= tr) & (peak - r >= tr))
        out[f"trail_{int(tr*100)}"] = finish(give[0] if give.size else n - 1)
    # article's tiered trailing, in price terms
    tier = np.where(peak >= 0.07, 0.03, np.where(peak >= 0.04, 0.02, 0.015))
    give = np.flatnonzero((peak >= 0.02) & (peak - r >= tier))
    out["article_tiers"] = finish(give[0] if give.size else n - 1)
    ch = np.flatnonzero(np.arange(n) >= 20)
    if side > 0:
        xl = pd.Series(low).rolling(20).min().shift(1).to_numpy()
        stop = np.flatnonzero(path < xl)
    else:
        xh = pd.Series(high).rolling(20).max().shift(1).to_numpy()
        stop = np.flatnonzero(path > xh)
    stop = stop[stop > 0]
    out["channel_20"] = finish(stop[0] if stop.size else n - 1)
    a = atr_series
    trail_stop = np.nan
    hit = n - 1
    for i in range(n):
        if not np.isfinite(a[i]):
            continue
        lvl = path[i] - 3.0 * a[i] if side > 0 else path[i] + 3.0 * a[i]
        trail_stop = lvl if not np.isfinite(trail_stop) else (
            max(trail_stop, lvl) if side > 0 else min(trail_stop, lvl))
        if i > 0 and ((side > 0 and path[i] < trail_stop)
                      or (side < 0 and path[i] > trail_stop)):
            hit = i
            break
    out["atr_3x"] = finish(hit)

    # tradable conditioning: from the FIRST bar the trade is up X, what comes
    # after? `peak` above includes the future and cannot be traded on.
    cond = {}
    for x in (0.05, 0.10, 0.20, 0.30):
        hit_x = np.flatnonzero(r >= x)
        tag = f"c{int(x*100)}"
        if hit_x.size == 0:
            cond[tag + "_hit"] = False
            continue
        i0 = hit_x[0]
        fwd = r[i0:]
        cond[tag + "_hit"] = True
        cond[tag + "_bar"] = int(i0)
        cond[tag + "_fwd_max"] = float(fwd.max() - r[i0])
        cond[tag + "_fwd_min"] = float(fwd.min() - r[i0])
        cond[tag + "_fwd_end"] = float(fwd[-1] - r[i0])
    return out, float(peak[-1]), float(r[-1]), cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--hold-days", type=int, default=30)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--top-vol", type=int, default=30)
    ap.add_argument("--sel-vol-days", type=int, default=7)
    ap.add_argument("--max-trade-days", type=int, default=14)
    ap.add_argument("--entry-window", type=int, default=55,
                    help="Donchian breakout lookback in bars")
    ap.add_argument("--top-n", type=int, default=150)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--out", default="/root/freqtrade/user_data/research/ma_harness/breakouts.parquet")
    a = ap.parse_args()

    p = Panel(a.tf, mmap=True)
    bpd = BPD[a.tf]
    close, high, low = p["close"], p["high"], p["low"]
    qv, fr, mask = p["quote_volume"], p["funding_rate"], p["universe_mask"]
    hold = a.hold_days * bpd
    warm = 60 * 1 + 60
    bar_hours = 24.0 / bpd

    trades = []
    for t in range(90 * bpd, p.shape[0] - hold, a.step_days * bpd):
        dollar = np.nansum(qv[t - 30 * bpd:t], axis=0)
        ok = np.isfinite(close[t - 90 * bpd:t]).all(axis=0) & mask[t - 1]
        dollar = np.where(ok, dollar, -1.0)
        liq = np.argsort(-dollar)[:a.top_n]
        liq = liq[dollar[liq] > 0]
        c7 = np.asarray(close[t - a.sel_vol_days * bpd:t, liq], dtype=np.float64)
        v7 = np.nanstd(np.diff(np.log(c7), axis=0), axis=0)
        for col in liq[np.argsort(-v7)[:a.top_vol]]:
            cl = np.asarray(close[t - warm:t + hold + 1, col], dtype=np.float64)
            hi = np.asarray(high[t - warm:t + hold + 1, col], dtype=np.float64)
            lo = np.asarray(low[t - warm:t + hold + 1, col], dtype=np.float64)
            fu = np.asarray(fr[t - warm:t + hold + 1, col], dtype=np.float64)
            if not np.isfinite(cl).all():
                continue
            hh = pd.Series(hi).rolling(a.entry_window).max().shift(1).to_numpy()
            ll = pd.Series(lo).rolling(a.entry_window).min().shift(1).to_numpy()
            tr = np.concatenate([[np.nan], np.maximum(
                hi[1:] - lo[1:], np.maximum(np.abs(hi[1:] - cl[:-1]),
                                            np.abs(lo[1:] - cl[:-1])))])
            atr = pd.Series(tr).ewm(alpha=1 / 24, adjust=False).mean().to_numpy()
            i = warm
            end = cl.size - 1
            while i < end:
                side = 0
                if np.isfinite(hh[i]) and cl[i] > hh[i]:
                    side = 1
                elif np.isfinite(ll[i]) and cl[i] < ll[i]:
                    side = -1
                if side == 0:
                    i += 1
                    continue
                j = min(i + a.max_trade_days * bpd, end)
                res, peak, final, cond = exits(cl[i:j + 1], side, a.fee, fu[i:j + 1],
                                         bar_hours, hi[i:j + 1], lo[i:j + 1],
                                         atr[i:j + 1])
                rec = dict(t=int(t), symbol=p.symbols[col], side=side,
                           peak=peak, final=final, **cond)
                for k, (ret, bars) in res.items():
                    rec[k] = ret
                    rec[k + "_bars"] = bars
                trades.append(rec)
                i = i + res["channel_20"][1] + 1     # flat, then look again
    df = pd.DataFrame(trades)
    df.to_parquet(a.out)
    cond_cols = {c for c in df.columns
                 if c.split("_")[0] in ("c5", "c10", "c20", "c30")}
    rules = [c for c in df.columns
             if c not in ("t", "symbol", "side", "peak", "final")
             and not c.endswith("_bars") and c not in cond_cols]
    print(f"{len(df)} breakout trades on {df.symbol.nunique()} coins, "
          f"{df.t.nunique()} selection dates, {a.sel_vol_days}d volatility "
          f"screen, Donchian({a.entry_window}), {a.max_trade_days}d max hold, "
          f"fee {a.fee*100:.3f}%/side\n")
    print(f"{'exit rule':<20}{'mean':>9}{'median':>9}{'win%':>7}{'bars':>7}"
          f"{'t by date':>11}")
    for r in rules:
        per = df.groupby("t")[r].mean()
        tt = per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))
        print(f"{r:<20}{100*df[r].mean():>+8.2f}%{100*df[r].median():>+8.2f}%"
              f"{100*(df[r] > 0).mean():>6.0f}%{df[r + '_bars'].mean():>7.0f}"
              f"{tt:>+11.2f}")

    print("\ncontinuation from the FIRST bar the trade is up X (no look ahead)")
    print(f"  {'threshold':<12}{'trades':>8}{'share':>8}{'further up':>12}"
          f"{'further down':>14}{'ends at':>10}{'up>down':>9}")
    for x in (0.05, 0.10, 0.20, 0.30):
        tag = f"c{int(x*100)}"
        g = df[df.get(tag + "_hit", False) == True]  # noqa: E712
        if len(g) < 30:
            continue
        up, dn = g[tag + "_fwd_max"], g[tag + "_fwd_min"]
        print(f"  up {100*x:>3.0f}%{'':<6}{len(g):>8}{100*len(g)/len(df):>7.1f}%"
              f"{100*up.median():>+11.1f}%{100*dn.median():>+13.1f}%"
              f"{100*g[tag + '_fwd_end'].median():>+9.1f}%"
              f"{100*(up > -dn).mean():>8.0f}%")


if __name__ == "__main__":
    main()
