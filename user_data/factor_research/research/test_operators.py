"""Validate every operator against an independent pandas implementation.

pandas is the reference precisely because it is not how the operators are
implemented: bottleneck kernels and rolling-sum identities can agree with each
other while both being wrong, so the check has to come from a different route.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import operators as ops  # noqa: E402

T, N, W = 400, 7, 20
rng = np.random.default_rng(0)


def sample():
    """Random walks plus a leading NaN block per column, mimicking late listings."""
    x = rng.standard_normal((T, N)).cumsum(axis=0) + 100.0
    for j in range(N):
        x[: rng.integers(0, 40), j] = np.nan
    return x


X, Y = sample(), sample()
PX, PY = pd.DataFrame(X), pd.DataFrame(Y)
RX, RY = PX.rolling(W, min_periods=W), PY.rolling(W, min_periods=W)

CASES = [
    ("ts_mean", ops.ts_mean(X, W), RX.mean()),
    ("ts_std", ops.ts_std(X, W), RX.std()),
    ("ts_var", ops.ts_var(X, W), RX.var()),
    ("ts_sum", ops.ts_sum(X, W), RX.sum()),
    ("ts_median", ops.ts_median(X, W), RX.median()),
    ("ts_min", ops.ts_min(X, W), RX.min()),
    ("ts_max", ops.ts_max(X, W), RX.max()),
    ("ts_range", ops.ts_range(X, W), RX.max() - RX.min()),
    ("ts_skew", ops.ts_skew(X, W), RX.skew()),
    ("ts_kurt", ops.ts_kurt(X, W), RX.kurt()),
    ("ts_zscore", ops.ts_zscore(X, W), (PX - RX.mean()) / RX.std()),
    ("ts_delay", ops.ts_delay(X, W), PX.shift(W)),
    ("ts_delta", ops.ts_delta(X, W), PX - PX.shift(W)),
    ("ts_pct_change", ops.ts_pct_change(X, W), PX / PX.shift(W) - 1.0),
    ("ts_corr", ops.ts_corr(X, Y, W), RX.corr(PY)),
    ("ts_cov", ops.ts_cov(X, Y, W), RX.cov(PY)),
    ("ts_beta", ops.ts_beta(X, Y, W), RX.cov(PY) / RY.var()),
    ("ts_rank", ops.ts_rank(X, W),
     RX.apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False)),
    ("ts_argmin", ops.ts_argmin(X, W),
     RX.apply(lambda s: len(s) - 1 - np.argmin(s.values), raw=False)),
    ("ts_argmax", ops.ts_argmax(X, W),
     RX.apply(lambda s: len(s) - 1 - np.argmax(s.values), raw=False)),
    ("ts_count_above_mean", ops.ts_count_above_mean(X, W),
     (PX > RX.mean()).astype(float).rolling(W, min_periods=W).sum()),
    ("ts_slope", ops.ts_slope(X, W),
     RX.apply(lambda s: np.polyfit(np.arange(len(s)), s.values, 1)[0], raw=False)),
    ("decay_linear", ops.decay_linear(X, W),
     RX.apply(lambda s: np.average(s.values, weights=np.arange(1, len(s) + 1)), raw=False)),
    ("ts_autocorr", ops.ts_autocorr(X, W), RX.corr(PX.shift(1))),
    ("ts_ema", ops.ts_ema(X, 10.0), PX.ewm(halflife=10.0, adjust=False).mean()),
]


# Rolling power sums lose precision that no bottleneck kernel does; skew and
# kurtosis are held to a looser bar than the exactly-representable operators.
TOL = {"ts_skew": 1e-7, "ts_kurt": 1e-5}


def compare(name, got, ref):
    got = np.asarray(got, dtype="float64")
    ref = np.asarray(ref, dtype="float64")
    both = np.isfinite(got) & np.isfinite(ref)
    # NaN placement is part of the contract: a partial window must stay NaN.
    shape_ok = np.array_equal(np.isfinite(got), np.isfinite(ref))
    if both.sum() == 0:
        return name, False, np.nan, shape_ok, 0
    err = np.abs(got[both] - ref[both])
    scale = np.maximum(np.abs(ref[both]), 1.0)
    rel = (err / scale).max()
    return name, rel < TOL.get(name, 1e-8), rel, shape_ok, int(both.sum())


def price_scale_check():
    """Skew/kurt must not degrade as price magnitude grows."""
    print("\nprecision vs price level (ts_kurt max rel err):")
    for level in (1e0, 1e2, 1e5):
        z = rng.standard_normal((T, N)).cumsum(axis=0) * 0.01 * level + level
        ref = pd.DataFrame(z).rolling(W, min_periods=W).kurt()
        _, _, rel, _, _ = compare("ts_kurt", ops.ts_kurt(z, W), ref)
        print(f"  price ~{level:>7.0e} : {rel:.3e}")


if __name__ == "__main__":
    print(f"{'operator':22s} {'match':6s} {'max rel err':>12s}  {'nan layout':10s} {'n':>7s}")
    failures = 0
    for name, got, ref in CASES:
        name, ok, rel, shape_ok, n = compare(name, got, ref)
        if not ok or not shape_ok:
            failures += 1
        print(f"{name:22s} {'OK' if ok else 'FAIL':6s} {rel:12.3e}  "
              f"{'OK' if shape_ok else 'MISMATCH':10s} {n:7d}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} operators verified")
    price_scale_check()
    sys.exit(1 if failures else 0)
