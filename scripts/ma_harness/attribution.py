"""Where does the tuned config's money actually come from?

Takes the settled setup -- 3d-volatility movers, Donchian 96/12 with a 2xATR
trail and inverse-volatility sizing -- turns the position matrix into a trade
ledger, and cuts the profit by everything that could carry a regularity: side,
hour, weekday, month, BTC regime, funding, breadth, entry context. Every cut is
reported on both halves of the sample, because a split that only works in one
half is a story, not a pattern.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/scripts/ma_harness")
import trend_lab as tl  # noqa: E402


def ledger(cfg, sel, fee):
    d = tl.load()
    close, fund = d["close"], d["fund"]
    p = d["p"]
    state = tl.build_state(cfg) * sel
    pos = np.vstack([np.zeros((1, close.shape[1]), np.float32), state[:-1]])
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / close[:-1] - 1.0
    ret[~np.isfinite(ret)] = 0.0
    turn = np.abs(np.diff(np.vstack([np.zeros((1, close.shape[1]), np.float32),
                                     pos]), axis=0))
    pnl = pos * ret - turn * fee - pos * fund / 8.0

    btc = close[:, p.symbols.index("BTCUSDT")].astype(np.float64)
    btc_ma = pd.Series(btc).rolling(200).mean().to_numpy()
    btc_r = np.diff(np.log(btc), prepend=np.log(btc[0]))
    btc_vol = pd.Series(btc_r).rolling(7 * 24).std().to_numpy()
    btc_vol_q = pd.Series(btc_vol).rolling(180 * 24, min_periods=500)\
        .rank(pct=True).to_numpy()
    breadth = (pos != 0).sum(axis=1)

    rows = []
    sgn = np.sign(pos)
    for c in range(close.shape[1]):
        s = sgn[:, c]
        idx = np.flatnonzero(s != 0)
        if idx.size == 0:
            continue
        brk = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate([[idx[0]], idx[brk + 1]])
        ends = np.concatenate([idx[brk], [idx[-1]]])
        for a, b in zip(starts, ends):
            seg = s[a:b + 1]
            for sub_a, sub_b in _sign_runs(seg, a):
                q = pnl[sub_a:sub_b + 1, c]
                if not np.isfinite(q).all():
                    continue
                px = close[sub_a:sub_b + 1, c].astype(np.float64)
                side = float(s[sub_a])
                r = side * (px / px[0] - 1.0)
                rows.append(dict(
                    coin=p.symbols[c], t=int(sub_a), side=side,
                    pnl=float(q.sum()), bars=int(sub_b - sub_a + 1),
                    mfe=float(r.max()), mae=float(r.min()),
                    size=float(np.abs(pos[sub_a:sub_b + 1, c]).mean()),
                    fund=float(np.mean(fund[sub_a:sub_b + 1, c])),
                    btc_up=bool(btc[sub_a] > btc_ma[sub_a])
                    if np.isfinite(btc_ma[sub_a]) else None,
                    btc_vol_q=float(btc_vol_q[sub_a])
                    if np.isfinite(btc_vol_q[sub_a]) else np.nan,
                    breadth=int(breadth[sub_a]),
                    date=p.index[sub_a]))
    df = pd.DataFrame(rows)
    df["hour"] = df.date.dt.hour
    df["weekday"] = df.date.dt.dayofweek
    df["month"] = df.date.dt.to_period("M").astype(str)
    df["half"] = np.where(df.t < df.t.max() / 2 + df.t.min() / 2, "h1", "h2")
    return df


def _sign_runs(seg, offset):
    out, start = [], 0
    for i in range(1, seg.size):
        if seg[i] != seg[i - 1]:
            out.append((offset + start, offset + i - 1))
            start = i
    out.append((offset + start, offset + seg.size - 1))
    return out


def cut(df, by, label, min_n=40):
    g = df.groupby(by, observed=True)
    tot = df.pnl.sum()
    print(f"\n{label}")
    print(f"  {'bucket':<16}{'trades':>8}{'sum P&L':>10}{'share':>8}"
          f"{'mean':>9}{'win%':>7}{'h1 sum':>9}{'h2 sum':>9}")
    for k, sub in g:
        if len(sub) < min_n:
            continue
        h1 = sub[sub.half == "h1"].pnl.sum()
        h2 = sub[sub.half == "h2"].pnl.sum()
        print(f"  {str(k):<16}{len(sub):>8}{sub.pnl.sum():>+10.2f}"
              f"{100*sub.pnl.sum()/tot:>7.0f}%{100*sub.pnl.mean():>+8.3f}%"
              f"{100*(sub.pnl > 0).mean():>6.0f}%{h1:>+9.2f}{h2:>+9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=0.0005)
    a = ap.parse_args()
    sel = tl.selection_mask(3, 3, top_vol=30)
    cfg = dict(kind="donchian", entry=96, exit=12, atr_mult=2.0,
               vol_target=0.6, label="tuned")
    df = ledger(cfg, sel, a.fee)
    print(f"{len(df)} trades on {df.coin.nunique()} coins, "
          f"total P&L {df.pnl.sum():+.2f} (portfolio units, 1/30 each)")
    print(f"win rate {100*(df.pnl > 0).mean():.1f}%  "
          f"median hold {df.bars.median():.0f}h  mean hold {df.bars.mean():.0f}h  "
          f"mean winner {100*df[df.pnl > 0].pnl.mean():+.2f}%  "
          f"mean loser {100*df[df.pnl < 0].pnl.mean():+.2f}%")
    x = df.pnl.sort_values()
    n5 = max(1, int(0.05 * len(x)))
    print(f"top 5% of trades contribute {100*x.tail(n5).sum()/x.sum():.0f}% "
          f"of the total; without them the rest sums to {x.head(len(x)-n5).sum():+.2f}")
    top = df.groupby("coin").pnl.sum().sort_values()
    print(f"best coin {top.index[-1]} {top.iloc[-1]:+.2f}, "
          f"worst {top.index[0]} {top.iloc[0]:+.2f}, "
          f"top 10 coins = {100*top.tail(10).sum()/df.pnl.sum():.0f}% of P&L "
          f"({df.coin.nunique()} coins traded)")

    cut(df, "side", "by side (+1 long, -1 short)")
    cut(df, df.hour // 4 * 4, "by entry hour (UTC, 4h buckets)")
    cut(df, "weekday", "by weekday of entry (0=Mon)")
    cut(df, "btc_up", "by BTC above/below its 200h average at entry")
    cut(df, pd.qcut(df.btc_vol_q, 3, labels=["btc calm", "btc mid", "btc wild"]),
        "by BTC volatility percentile at entry")
    cut(df, pd.qcut(df.breadth, 3, labels=["few open", "mid", "many open"]),
        "by how many positions were already open")
    cut(df, pd.qcut(df.bars, 4, labels=["<q1", "q2", "q3", "longest"]),
        "by holding time")
    cut(df, pd.qcut(df["fund"], 3, labels=["funding neg", "flat", "funding pos"]),
        "by average funding rate during the trade")
    cut(df, np.where(df.side > 0, np.where(df["fund"] > 0, "long pays",
                                           "long collects"),
                     np.where(df["fund"] > 0, "short collects", "short pays")),
        "by which side of funding the position sat on")
    m = df.groupby("month").pnl.sum()
    print("\nby month")
    for k, v in m.items():
        print(f"  {k}  {v:+.2f}  {'#' * int(max(0, v * 40))}"
              f"{'.' * int(max(0, -v * 40))}")


if __name__ == "__main__":
    main()
