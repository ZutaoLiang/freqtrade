"""Cross-check the IC evaluator against independent implementations.

Every check here re-derives the same number a different way -- scipy for the
rank correlations, statsmodels for the HAC standard errors, a closed form for
the effective count -- so a shared bug has to be made twice to slip through.
The last two checks are behavioural rather than numerical: a factor built to
carry known signal must be found, and a factor built from pure noise must not
pass the gate more often than the gate's nominal rate.
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ic as M  # noqa: E402

TOL = 1e-10
T, N = 600, 40


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, N))
    x[rng.random((T, N)) < 0.05] = np.nan          # scattered holes
    x[:20, :5] = np.nan                            # late listings
    return x


def _report(name, err, tol=TOL):
    ok = np.isfinite(err) and err <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} max err {err:.3e}")
    return ok


def check_rank_axis1():
    x = _panel(1)
    got = M._rank_axis1(x)
    want = np.full_like(x, np.nan)
    for i in range(T):
        row = x[i]
        ok = np.isfinite(row)
        if ok.sum():
            want[i, ok] = stats.rankdata(row[ok]) / ok.sum()
    both = np.isfinite(got) & np.isfinite(want)
    layout = np.array_equal(np.isfinite(got), np.isfinite(want))
    err = np.nanmax(np.abs(got[both] - want[both]))
    return _report("_rank_axis1 vs scipy.rankdata", err if layout else np.inf)


def check_ic_cross_section():
    f, r = _panel(2), _panel(3)
    got = M.rank_ic_cross_section(f, r, min_symbols=5)
    want = np.full(T, np.nan)
    for i in range(T):
        ok = np.isfinite(f[i]) & np.isfinite(r[i])
        if ok.sum() >= 5:
            want[i] = stats.spearmanr(f[i, ok], r[i, ok]).statistic
    layout = np.array_equal(np.isfinite(got), np.isfinite(want))
    both = np.isfinite(got) & np.isfinite(want)
    err = np.nanmax(np.abs(got[both] - want[both]))
    return _report("cross-sectional IC vs spearmanr", err if layout else np.inf)


def check_ic_per_symbol():
    f, r = _panel(4), _panel(5)
    got, obs = M.rank_ic_per_symbol(f, r, min_obs=50)
    want = np.full(N, np.nan)
    for j in range(N):
        ok = np.isfinite(f[:, j]) & np.isfinite(r[:, j])
        if ok.sum() >= 50:
            want[j] = stats.spearmanr(f[ok, j], r[ok, j]).statistic
    layout = np.array_equal(np.isfinite(got), np.isfinite(want))
    both = np.isfinite(got) & np.isfinite(want)
    err = np.nanmax(np.abs(got[both] - want[both]))
    counts = np.array_equal(obs, (np.isfinite(f) & np.isfinite(r)).sum(axis=0))
    return _report("per-symbol IC vs spearmanr",
                   err if (layout and counts) else np.inf)


def check_newey_west():
    """statsmodels regresses on a constant; its HAC se must match ours.

    maxlags=L in statsmodels uses the same Bartlett weights over lags 1..L and
    the same division by n, so with use_correction off the two must agree to
    floating-point precision rather than merely closely.
    """
    import statsmodels.api as sm
    rng = np.random.default_rng(6)
    worst = 0.0
    for band in (0, 3, 11, 23):
        e = rng.standard_normal(T + 40)
        v = np.convolve(e, np.ones(12) / 12.0, mode="valid")[:T]   # MA(11)
        mu, se, t, n = M.newey_west(v, band=band)
        res = sm.OLS(v, np.ones(n)).fit(cov_type="HAC",
                                        cov_kwds={"maxlags": band, "use_correction": False})
        want_se = float(res.bse[0])
        worst = max(worst, abs(se - want_se) / want_se,
                    abs(mu - v.mean()) / max(abs(v.mean()), 1e-12))
    return _report("Newey-West se vs statsmodels HAC", worst, 1e-9)


def check_effective_count():
    worst = 0.0
    for n, rho in ((40, 0.0), (40, 0.5), (800, 0.6), (1, 0.9)):
        got = M.effective_count(n, rho)
        want = n if n <= 1 else n / (1 + (n - 1) * rho)
        worst = max(worst, abs(got - want))
    # The independent case must return n exactly, or nothing downstream is right
    worst = max(worst, abs(M.effective_count(40, 0.0) - 40))
    return _report("effective_count closed form", worst)


def check_forward_return_and_lag():
    # No holes here on purpose: one NaN in a cumprod poisons the whole column,
    # which would hide an alignment error behind an all-NaN comparison.
    rng = np.random.default_rng(7)
    c = np.cumprod(1 + 0.01 * rng.standard_normal((T, 3)), axis=0)
    worst = 0.0
    for h in (1, 3, 10):
        got = M.forward_return(c, h)
        want = np.full_like(c, np.nan)
        want[:-h] = c[h:] / c[:-h] - 1
        worst = max(worst, np.nanmax(np.abs(got - want)))
        assert np.all(np.isnan(got[-h:])), "forward return must end in NaN"
    l = M.lag(c, 2)
    worst = max(worst, np.nanmax(np.abs(l[2:] - c[:-2])))
    assert np.all(np.isnan(l[:2])), "lag must open with NaN"
    return _report("forward_return / lag alignment", worst)


def check_no_lookahead():
    """A factor equal to the future return must score ~0 once lagged wrongly.

    The point of the one-bar lag is that perfect foresight at bar t cannot be
    scored against bar t. Here the factor IS the forward return, so with the
    lag applied the IC must fall far below the unlagged 1.0.
    """
    rng = np.random.default_rng(8)
    c = np.cumprod(1 + 0.01 * rng.standard_normal((T, N)), axis=0)
    r = M.forward_return(c, 1)
    unlagged = np.nanmean(M.rank_ic_cross_section(r, r, min_symbols=5))
    rows = M.evaluate(r, c, horizons=(1,), lag_bars=1, rho=0.0)
    lagged = abs(rows[0]["ic_cs"])
    print(f"  {'PASS' if unlagged > 0.99 and lagged < 0.15 else 'FAIL'}  "
          f"{'one-bar lag kills foresight':34s} "
          f"unlagged {unlagged:.3f} -> lagged {lagged:.3f}")
    return unlagged > 0.99 and lagged < 0.15


def check_power_and_size():
    """Known signal must be found; pure noise must not pass the t > 3 gate."""
    rng = np.random.default_rng(9)
    t_, n_ = 3000, 60
    common = rng.standard_normal((t_, 1)) * 0.01           # market beta
    idio = rng.standard_normal((t_, n_)) * 0.01
    ret = common + idio
    c = np.cumprod(1 + ret, axis=0)

    # Signal at bar t must foresee the return the evaluator will score it
    # against: evaluate lags one bar, so f[t] = sig[t-1] meets ret[t+1], and
    # sig[t] therefore has to carry idio[t+2]. Contaminated down to IC ~ 0.05.
    sig = np.full((t_, n_), np.nan)
    sig[:-2] = 0.05 * idio[2:] + 0.999 * rng.standard_normal((t_ - 2, n_)) * 0.01
    rows = M.evaluate(sig, c, horizons=(1,), lag_bars=1, rho=0.0)
    found = abs(rows[0]["ic_cs_t"]) > M.T_GATE

    hits = 0
    trials = 40
    for k in range(trials):
        noise = rng.standard_normal((t_, n_))
        r = M.evaluate(noise, c, horizons=(1,), lag_bars=1, rho=0.0)
        hits += bool(r[0]["passes"])
    rate = hits / trials
    ok = found and rate <= 0.05
    print(f"  {'PASS' if ok else 'FAIL'}  {'power / false-positive rate':34s} "
          f"signal t={rows[0]['ic_cs_t']:.2f}  noise pass rate {rate:.1%}")
    return ok


def check_shift_null_fft():
    """The one-pass null must equal the rotate-and-recompute it replaces.

    The whole null distribution comes from one cross-correlation, so if the
    transform convention were off by a sign or an index every z-score in the
    sweep would be wrong in a way nothing else would catch.
    """
    rng = np.random.default_rng(12)
    t_, n_ = 1200, 40
    x = rng.standard_normal((t_, n_))
    y = rng.standard_normal((t_, n_))
    x[rng.random(x.shape) < 0.2] = np.nan
    y[rng.random(y.shape) < 0.2] = np.nan
    xi, _ = M._impute_demean(x)
    yi, _ = M._impute_demean(y)
    den = np.sqrt((xi * xi).sum(0) * (yi * yi).sum(0))
    cross = np.fft.irfft(np.fft.rfft(xi, axis=0) * np.conj(np.fft.rfft(yi, axis=0)),
                         n=t_, axis=0)
    curve = (cross / den).mean(1)
    err = 0.0
    for k in (0, 1, 7, 123, 600, t_ - 1):
        direct = ((xi * np.roll(yi, k, axis=0)).sum(0) / den).mean()
        err = max(err, abs(direct - curve[k]))
    return _report("shift null vs explicit rotation", err, 1e-14)


def check_persistent_null():
    """A persistent factor built from the price itself must still test as noise.

    This is the regression guard for the 2026-08-21 bug: on pure random walks,
    a block-based per-symbol statistic reported IC -0.33 at t = -108 for
    ts_rank(close,720) with no signal present, and the cross-sectional t-stat
    reached 3.0 because the Newey-West bandwidth ignored how persistent the IC
    series was. Both statistics must now sit inside their nominal size.
    """
    import operators as ops

    rng = np.random.default_rng(11)
    t_, n_, trials = 2500, 60, 12
    cs_hits = ts_hits = 0
    worst_z = 0.0
    for _ in range(trials):
        ret = rng.standard_normal((t_, n_)) * 0.01
        c = np.cumprod(1 + ret, axis=0) * 100
        for f in (ops.ts_rank(c, 480), ops.ts_sum(ret, 72)):
            row = M.evaluate(f, c, horizons=(24,), lag_bars=1, rho=0.0,
                             n_draws=60)[0]
            cs_hits += bool(row["passes_cs"])
            ts_hits += bool(row["passes_ts"])
            if np.isfinite(row["ic_ts_z"]):
                worst_z = max(worst_z, abs(row["ic_ts_z"]))
    n = trials * 2
    ok = cs_hits / n <= 0.10 and ts_hits / n <= 0.10
    print(f"  {'PASS' if ok else 'FAIL'}  {'persistent-factor null size':34s} "
          f"cs {cs_hits}/{n}, ts {ts_hits}/{n}, worst |z| {worst_z:.2f}")
    return ok


def check_deflated_sharpe():
    """DSR must fall as the number of trials rises, and be high for a real edge."""
    rng = np.random.default_rng(10)
    trials = rng.standard_normal(500) * 0.05
    d_small, sr0_small = M.deflated_sharpe(0.20, 3000, sr_trials=trials[:20])
    d_big, sr0_big = M.deflated_sharpe(0.20, 3000, sr_trials=trials)
    mono = sr0_big > sr0_small and d_big < d_small
    strong, _ = M.deflated_sharpe(0.60, 3000, sr_trials=trials)
    ok = mono and strong > 0.99 and 0.0 <= d_big <= 1.0
    print(f"  {'PASS' if ok else 'FAIL'}  {'deflated Sharpe monotonicity':34s} "
          f"20 trials {d_small:.3f} -> 500 trials {d_big:.3f}")
    return ok


if __name__ == "__main__":
    print("IC evaluator checks")
    results = [
        check_rank_axis1(),
        check_ic_cross_section(),
        check_ic_per_symbol(),
        check_newey_west(),
        check_effective_count(),
        check_forward_return_and_lag(),
        check_no_lookahead(),
        check_power_and_size(),
        check_shift_null_fft(),
        check_persistent_null(),
        check_deflated_sharpe(),
    ]
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
