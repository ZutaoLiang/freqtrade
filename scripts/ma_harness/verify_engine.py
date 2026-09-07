"""Cross-check the vectorised sleeve against an explicit event loop.

The walk-forward conclusions rest on `oos_returns`, so its bar returns are
compared here with a trade-by-trade simulation written independently: an
account that flips at each crossover, pays the fee on the notional it turns
over, and accrues funding on what it holds.
"""
import sys
import numpy as np

sys.path.insert(0, "/root/freqtrade/scripts/ma_harness")
sys.path.insert(0, "/root/freqtrade/user_data/factor_research")
import walkforward as wf
from research.panel import Panel


def event_loop(close, funding, f, s, fee, bar_hours):
    fast = np.convolve(close, np.ones(f) / f, mode="full")[:close.size]
    slow = np.convolve(close, np.ones(s) / s, mode="full")[:close.size]
    fast[:f - 1] = np.nan
    slow[:s - 1] = np.nan
    equity, pos = 1.0, 0.0
    curve = [1.0]
    for i in range(1, close.size):
        r = close[i] / close[i - 1] - 1.0
        equity *= 1.0 + pos * r
        equity -= equity * pos * funding[i] * (bar_hours / 8.0)
        d = fast[i] - slow[i]
        want = pos if not np.isfinite(d) else np.sign(d)
        if want != pos:
            equity -= equity * abs(want - pos) * fee
            pos = want
        curve.append(equity)
    return np.array(curve)


def main():
    p = Panel("1h", mmap=True)
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(20):
        c = rng.integers(len(p.symbols))
        t0 = rng.integers(2000, p.shape[0] - 1500)
        close = np.asarray(p["close"][t0:t0 + 1400, c], dtype=np.float64)
        fund = np.nan_to_num(np.asarray(p["funding_rate"][t0:t0 + 1400, c],
                                        dtype=np.float64))
        if not np.isfinite(close).all():
            continue
        f, s = wf.GRID[rng.integers(len(wf.GRID))]
        ret, _ = wf.oos_returns(close, fund, f, s, 0.0005, 1.0)
        vec = np.cumprod(1.0 + np.nan_to_num(ret))
        ev = event_loop(close, fund, f, s, 0.0005, 1.0)
        # compare only past the warm-up the harness itself skips
        w = max(wf.SLOW) + 5
        vec, ev = vec[w:] / vec[w], ev[w:] / ev[w]
        gap = abs(vec[-1] / ev[-1] - 1.0)
        worst = max(worst, gap)
        print(f"{p.symbols[c]:<16} {f:>3}/{s:<3} vectorised {vec[-1]:.6f}  "
              f"event loop {ev[-1]:.6f}  gap {gap:.2e}")
    print(f"worst relative gap {worst:.2e}")


if __name__ == "__main__":
    main()
