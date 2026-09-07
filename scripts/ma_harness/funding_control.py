"""Is the funding gate information, or just trading less?

The |funding| filter cuts exposure hard, and any filter that cuts exposure
changes the Sharpe of what is left. The control here replaces it with a random
gate of the same duty cycle, drawn in blocks so it turns on and off at the same
rhythm, and asks where the real filter's Sharpe sits in that distribution.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/scripts/ma_harness")
import trend_lab as tl  # noqa: E402

BPD = 24


def curve(state, sel, fee):
    d = tl.load()
    close, fund = d["close"], d["fund"]
    st = state * sel
    pos = np.vstack([np.zeros((1, st.shape[1]), np.float32), st[:-1]])
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / close[:-1] - 1.0
    ret[~np.isfinite(ret)] = 0.0
    turn = np.abs(np.diff(np.vstack([np.zeros((1, st.shape[1]), np.float32),
                                     pos]), axis=0))
    pnl = pos * ret - turn * fee - pos * fund / 8.0
    k = int(sel.sum(axis=1).max())
    return np.nansum(pnl, axis=1) / k


def sharpe(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(24 * 365))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pctile", type=float, default=0.9)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--fee", type=float, default=0.0005)
    a = ap.parse_args()

    sel = tl.selection_mask(3, 3, top_vol=30)
    base = dict(kind="donchian", entry=96, exit=12, atr_mult=2.0,
                vol_target=0.6, label="base")
    raw = tl.build_state(base)
    gated = tl.build_state({**base, "fund_abs_pctile": a.pctile})

    d = tl.load()
    kept = float((gated != 0).sum() / max(1, (raw != 0).sum()))
    real = sharpe(curve(gated, sel, a.fee))
    print(f"|funding| gate at the {int(a.pctile*100)}th percentile keeps "
          f"{100*kept:.1f}% of the position-bars the ungated rule would hold")
    print(f"  ungated Sharpe {sharpe(curve(raw, sel, a.fee)):.2f}   "
          f"gated Sharpe {real:.2f}")

    # a random gate with the same duty cycle, in blocks of one funding period
    rng = np.random.default_rng(0)
    T, N = raw.shape
    blocks = int(np.ceil(T / 8))
    null = []
    for _ in range(a.draws):
        g = rng.random((blocks, N)) < kept
        g = np.repeat(g, 8, axis=0)[:T]
        null.append(sharpe(curve(raw * g, sel, a.fee)))
    null = np.array(null)
    print(f"  random gates of the same size: mean {null.mean():+.2f}  "
          f"sd {null.std(ddof=1):.2f}  95th {np.percentile(null, 95):+.2f}  "
          f"max {null.max():+.2f}")
    print(f"  p = {(null >= real).mean():.3f}   "
          f"z = {(real - null.mean()) / null.std(ddof=1):+.2f}")

    # and the same gate on shifted funding: keeps the shape, breaks the link
    f = pd.DataFrame(d["fund"]).rolling(3 * BPD).mean().shift(1).to_numpy()
    af = np.abs(f)
    shifted = []
    for _ in range(min(50, a.draws)):
        s = int(rng.integers(30 * 24, T - 30 * 24))
        aff = np.roll(af, s, axis=0)
        thr = np.nanquantile(np.where(np.isfinite(aff), aff, np.nan),
                             a.pctile, axis=1, keepdims=True)
        shifted.append(sharpe(curve(np.where(aff >= thr, raw, 0.0), sel, a.fee)))
    shifted = np.array(shifted)
    print(f"  gate built from time-shifted funding: mean {shifted.mean():+.2f}  "
          f"sd {shifted.std(ddof=1):.2f}  p = {(shifted >= real).mean():.3f}")


if __name__ == "__main__":
    main()
