"""How much of the tuned Sharpe survives a null that keeps the market's shape?

Forty-odd configurations were scored on the same bars, so the best of them is
biased upward. This rebuilds the same portfolio against circularly shifted
returns: the positions, the basket and the cost model are untouched, only the
link between a signal and the return it earned is broken. The shift preserves
each coin's volatility clustering and the cross-sectional correlation, which an
i.i.d. bootstrap would destroy.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/root/freqtrade/scripts/ma_harness")
import trend_lab as tl  # noqa: E402


def portfolio(state, sel, ret, fund, fee, k):
    st = state * sel
    pos = np.vstack([np.zeros((1, st.shape[1]), np.float32), st[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, st.shape[1]), np.float32),
                                     pos]), axis=0))
    pnl = pos * ret - turn * fee - pos * fund / 8.0
    return np.nansum(pnl, axis=1) / k


def sharpe(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(24 * 365))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=int, default=96)
    ap.add_argument("--exit", type=int, default=12)
    ap.add_argument("--atr-mult", type=float, default=2.0)
    ap.add_argument("--vol-target", type=float, default=0.6)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--min-shift-days", type=int, default=30)
    a = ap.parse_args()

    d = tl.load()
    close, fund = d["close"], d["fund"]
    sel = tl.selection_mask(3, 3, top_vol=30)
    cfg = dict(kind="donchian", entry=a.entry, exit=a.exit,
               atr_mult=a.atr_mult, vol_target=a.vol_target, label="tuned")
    state = tl.build_state(cfg)
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / close[:-1] - 1.0
    ret[~np.isfinite(ret)] = 0.0
    k = int(sel.sum(axis=1).max())

    real = sharpe(portfolio(state, sel, ret, fund, a.fee, k))
    T = close.shape[0]
    lo = a.min_shift_days * 24
    rng = np.random.default_rng(0)
    null = []
    for _ in range(a.draws):
        s = int(rng.integers(lo, T - lo))
        null.append(sharpe(portfolio(state, sel, np.roll(ret, s, axis=0),
                                     np.roll(fund, s, axis=0), a.fee, k)))
    null = np.array(null)
    p = float((null >= real).mean())
    print(f"donchian {a.entry}/{a.exit} +{a.atr_mult:g}ATR +vt{a.vol_target:g}")
    print(f"  realised Sharpe {real:.2f}")
    print(f"  null over {a.draws} circular shifts: mean {null.mean():+.2f}  "
          f"sd {null.std(ddof=1):.2f}  "
          f"95th pct {np.percentile(null, 95):+.2f}  "
          f"max {null.max():+.2f}")
    print(f"  p = {p:.3f}  (share of shifted worlds at least this good)")
    print(f"  z  = {(real - null.mean()) / null.std(ddof=1):+.2f}")


if __name__ == "__main__":
    main()
