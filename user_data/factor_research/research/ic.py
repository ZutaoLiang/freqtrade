"""Information-coefficient evaluation for time-series factors.

Four properties decide whether a sweep's conclusions mean anything, and all
four are enforced here rather than left to the caller:

1. The factor is lagged one bar before it meets a forward return, so a value
   observed at a bar close is never scored against that same bar's move.
2. Overlapping h-bar forward returns make the per-bar IC series autocorrelated
   by construction; t-stats use a Newey-West HAC estimator with a bandwidth of
   at least h - 1, not a plain sqrt(T).
3. Pooling per-symbol ICs pretends the symbols are independent. Crypto pairs
   move together, so the pooled t-stat divides by an effective symbol count
   derived from the average pairwise return correlation instead of by N.
4. A sweep of thousands of factors produces large t-stats from noise alone, so
   selection uses a hard t > 3 gate and, where the trial distribution is known,
   a Deflated Sharpe Ratio.

Everything works on the (time, symbol) float32 panels from panel.py.
"""
import numpy as np

try:
    import bottleneck as bn
except ImportError:  # pragma: no cover - bottleneck is a hard dependency here
    bn = None

MIN_SYMBOLS = 20        # a cross-sectional IC on fewer pairs is mostly noise
MIN_OBS = 200           # per-symbol IC below this many bars is not interpretable
T_GATE = 3.0            # multiple-testing gate, deliberately above the usual 2


def _f8(x):
    return np.asarray(x, dtype="float64")


def forward_return(close, h):
    """Simple return from this bar's close to the close h bars later."""
    c = _f8(close)
    fwd = np.full_like(c, np.nan)
    fwd[:-h] = c[h:] / c[:-h] - 1.0
    return fwd


def lag(x, k=1):
    """Shift down the time axis so row t holds the value from row t - k."""
    out = np.full_like(_f8(x), np.nan)
    if k:
        out[k:] = _f8(x)[:-k]
    else:
        out[:] = _f8(x)
    return out


def apply_mask(x, mask):
    """Blank anything outside the tradeable universe at that timestamp."""
    if mask is None:
        return _f8(x)
    return np.where(mask, _f8(x), np.nan)


def _rank_axis1(x):
    """Average ranks across symbols per bar, scaled to (0, 1], NaN preserved."""
    r = bn.nanrankdata(x, axis=1)
    n = np.sum(np.isfinite(x), axis=1, keepdims=True).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, r / n, np.nan)


def _rank_axis0(x):
    """Average ranks down time per symbol, scaled to (0, 1], NaN preserved."""
    r = bn.nanrankdata(x, axis=0)
    n = np.sum(np.isfinite(x), axis=0, keepdims=True).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, r / n, np.nan)


def _corr_along(a, b, axis, min_count):
    """Pearson correlation of a against b, reduced along one axis.

    Reducing along the requested axis directly matters more than it looks: the
    column-wise case used to transpose and reduce along axis 1, which turns a
    (8500, 860) pass into a strided one. With 24 workers competing for memory
    bandwidth that made the shift null roughly an order of magnitude slower
    than the operator it was testing.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(axis=axis).astype("float64")
    x = np.where(ok, a, 0.0)
    y = np.where(ok, b, 0.0)
    sx, sy = x.sum(axis=axis), y.sum(axis=axis)
    sxx = np.einsum("ij,ij->" + ("j" if axis == 0 else "i"), x, x)
    syy = np.einsum("ij,ij->" + ("j" if axis == 0 else "i"), y, y)
    sxy = np.einsum("ij,ij->" + ("j" if axis == 0 else "i"), x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        out = cov / np.sqrt(vx * vy)
    return np.where(n >= min_count, out, np.nan)


def _corr_rowwise(a, b, min_count):
    """Pearson correlation of each row of a against the same row of b."""
    return _corr_along(a, b, 1, min_count)


def _corr_colwise(a, b, min_count):
    """Pearson correlation of each column of a against the same column of b."""
    return _corr_along(a, b, 0, min_count)


def rank_ic_cross_section(f, r, min_symbols=MIN_SYMBOLS):
    """Spearman IC across symbols at each bar -- one observation per bar."""
    both = np.isfinite(f) & np.isfinite(r)
    fa = np.where(both, f, np.nan)
    ra = np.where(both, r, np.nan)
    return _corr_rowwise(_rank_axis1(fa), _rank_axis1(ra), min_symbols)


def rank_ic_per_symbol(f, r, min_obs=MIN_OBS):
    """Spearman IC down time for each symbol -- the time-series view."""
    both = np.isfinite(f) & np.isfinite(r)
    fa = np.where(both, f, np.nan)
    ra = np.where(both, r, np.nan)
    ic = _corr_colwise(_rank_axis0(fa), _rank_axis0(ra), min_obs)
    return ic, both.sum(axis=0)


def andrews_band(x):
    """Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel.

    The fixed 4(T/100)^(2/9) rule is a function of sample size only, so it
    returns the same bandwidth for an IC series with AR(1) 0.05 and one with
    0.99. A factor built on a 720-bar window produces an IC series whose
    autocorrelation runs far past any such fixed lag, and under-correcting
    there is what let a pure random walk reach |t| = 3 in the null test.
    """
    v = _f8(x)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 10:
        return 0
    d = v - v.mean()
    denom = float(d @ d)
    rho = float(d[1:] @ d[:-1]) / denom if denom > 0 else 0.0
    rho = min(max(rho, -0.97), 0.97)
    alpha = 4.0 * rho ** 2 / (1.0 - rho ** 2) ** 2
    band = int(np.ceil(1.1447 * (alpha * n) ** (1.0 / 3.0)))
    return int(max(0, min(band, n // 4)))


def newey_west(x, band=None):
    """Mean of a serially correlated series with its HAC standard error.

    Bartlett kernel. Returns (mean, se, t, n). With overlapping forward returns
    the series is MA(h-1) by construction, so `band` should be at least h - 1;
    when band is None the Andrews plug-in picks it from the series itself.
    """
    v = _f8(x)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 10:
        return np.nan, np.nan, np.nan, n
    if band is None:
        band = andrews_band(v)
    band = int(max(0, min(band, n - 2)))
    mu = v.mean()
    d = v - mu
    gamma0 = float(d @ d) / n
    total = gamma0
    for l in range(1, band + 1):
        gl = float(d[l:] @ d[:-l]) / n
        total += 2.0 * (1.0 - l / (band + 1.0)) * gl
    # A HAC sum can go negative in small samples; fall back to the plain
    # variance rather than emitting a NaN t-stat that reads as "no result".
    total = total if total > 0 else gamma0
    se = np.sqrt(total / n)
    return mu, se, (mu / se if se > 0 else np.nan), n


def average_pair_correlation(returns, sample=4000, seed=0):
    """Mean off-diagonal correlation of 1-bar returns across symbols."""
    r = _f8(returns)
    t = r.shape[0]
    if t > sample:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(t, size=sample, replace=False))
        r = r[rows]
    ok = np.isfinite(r)
    keep = ok.sum(axis=0) >= max(30, 0.1 * r.shape[0])
    r = r[:, keep]
    if r.shape[1] < 2:
        return 0.0, int(r.shape[1])
    x = np.where(np.isfinite(r), r, np.nan)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = np.where(np.isfinite(x), x, 0.0)
    n = (np.isfinite(r)).astype("float64")
    cov = x.T @ x
    cnt = n.T @ n
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = cov / np.maximum(cnt, 1.0)
        sd = np.sqrt(np.diag(cov))
        c = cov / np.outer(sd, sd)
    c = c[np.triu_indices_from(c, k=1)]
    c = c[np.isfinite(c) & (cnt[np.triu_indices_from(cnt, k=1)] >= 30)]
    if not c.size:
        return 0.0, int(r.shape[1])
    return float(np.mean(c)), int(r.shape[1])


def effective_count(n, rho):
    """Independent-observation equivalent of n estimates correlated at rho.

    Variance of the mean of n unit-variance estimates with common correlation
    rho is (1 + (n-1)rho)/n, so the effective count is n / (1 + (n-1)rho).
    With 800 crypto pairs at rho = 0.6 this is under 2, which is the entire
    reason a naive pooled t-stat over pairs is meaningless.
    """
    rho = float(max(0.0, min(0.99, rho)))
    if n <= 1:
        return float(n)
    return float(n / (1.0 + (n - 1.0) * rho))


def deflated_sharpe(sr, n_obs, sr_trials=None, n_trials=None, skew=0.0, kurt=3.0):
    """Probability the observed Sharpe survives the selection it came from.

    `sr` and `n_obs` describe the winning series (for a factor, its per-bar IC
    series treated as a return stream, so sr == ICIR). Either pass the SRs of
    every trial in the sweep, or the trial count plus an assumed spread.
    """
    from scipy.stats import norm

    if sr_trials is not None:
        sr_trials = _f8(sr_trials)
        sr_trials = sr_trials[np.isfinite(sr_trials)]
        n_trials = sr_trials.size
        var = float(np.var(sr_trials, ddof=1)) if n_trials > 1 else 0.0
    elif n_trials is None:
        raise ValueError("pass sr_trials or n_trials")
    else:
        var = float(sr) ** 2 if sr else 1e-4
    if n_trials < 2 or var <= 0:
        return np.nan, np.nan
    e = 0.5772156649015329
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(var) * ((1.0 - e) * z1 + e * z2)
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    dsr = float(norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / denom))
    return dsr, float(sr0)


def _impute_demean(x):
    """Column-wise: demean over the valid entries, then set NaN to zero.

    Zero after demeaning means a missing bar contributes nothing to any sum
    rather than being dropped pairwise. That costs a little attenuation on
    sparse columns, and buys a mask that does not change when the series is
    rotated -- which is what makes the whole null distribution computable in
    one pass instead of one pass per draw.
    """
    ok = np.isfinite(x)
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(n > 0, np.nansum(np.where(ok, x, 0.0), axis=0) / np.maximum(n, 1), 0.0)
    return np.where(ok, x - mu, 0.0), n


def prep_series(x):
    """Rank down time, then mean-impute and demean -- the reusable half."""
    return _impute_demean(_rank_axis0(np.where(np.isfinite(x), x, np.nan)))


def ts_null_test(f, r, min_obs=MIN_OBS, min_shift_frac=0.05, max_draws=4000,
                 seed=0, n_draws=None, f_prep=None):
    """Significance of the pooled per-symbol IC against a circular-shift null.

    An earlier version cut the sample into blocks and treated the block means
    as observations. That statistic is unusable: measured on pure random walks
    on 2026-08-21, ts_rank(close,720) scored a block IC of -0.33 at t = -108
    with no signal present, because the block was shorter than the factor
    window and the factor barely moved inside it. The full-sample per-symbol IC
    has no such bias -- the same null gave -0.012 -- so that is the statistic,
    and the only thing left to establish is its spread.

    The null rotates the forward-return panel in time by one offset applied to
    every symbol at once. That destroys the factor-return link while leaving
    intact the three things that would otherwise inflate significance: each
    series' own persistence, the overlap between forward returns, and the
    correlation between pairs.

    Every rotation is evaluated at once rather than sampled. Once the ranks are
    mean-imputed the denominators stop depending on the offset, so the whole
    null is one circular cross-correlation per symbol -- an FFT pair instead of
    a loop. `n_draws` is accepted and ignored; it is now the full set.
    """
    x, nx = f_prep if f_prep is not None else prep_series(f)
    y, ny = prep_series(r)

    keep = (nx >= min_obs) & (ny >= min_obs)
    if keep.sum() < 2:
        return {"ic_ts_shift": np.nan, "null_mean": np.nan, "null_sd": np.nan,
                "z": np.nan, "p": np.nan, "n_draws": 0}
    x, y = x[:, keep], y[:, keep]
    t = x.shape[0]

    denom = np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0))
    good = denom > 0
    if good.sum() < 2:
        return {"ic_ts_shift": np.nan, "null_mean": np.nan, "null_sd": np.nan,
                "z": np.nan, "p": np.nan, "n_draws": 0}
    x, y, denom = x[:, good], y[:, good], denom[good]

    # cross[k, j] = sum_t x[t, j] * y[(t - k) mod T, j] -- every rotation of y
    # against x, for every symbol, from one forward and one inverse transform.
    cross = np.fft.irfft(np.fft.rfft(x, axis=0) * np.conj(np.fft.rfft(y, axis=0)),
                         n=t, axis=0)
    curve = (cross / denom).mean(axis=1)          # pooled IC at each rotation
    obs = float(curve[0])

    lo = max(1, int(min_shift_frac * t))
    idx = np.arange(t)
    null = curve[(idx >= lo) & (idx <= t - lo)]
    if null.size > max_draws:                     # thin evenly, keep the spread
        null = null[np.linspace(0, null.size - 1, max_draws).astype(int)]
    if null.size < 10:
        return {"ic_ts_shift": obs, "null_mean": np.nan, "null_sd": np.nan,
                "z": np.nan, "p": np.nan, "n_draws": int(null.size)}
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    z = (obs - mu) / sd if sd > 0 else np.nan
    p = float(np.mean(np.abs(null - mu) >= abs(obs - mu)))
    return {"ic_ts_shift": obs, "null_mean": mu, "null_sd": sd, "z": z, "p": p,
            "n_draws": int(null.size)}


def turnover_proxy(f, k=1):
    """Autocorrelation of the factor at lag k, averaged over symbols.

    High autocorrelation means a slow signal and little turnover; a factor at
    0.05 will be eaten by fees regardless of its IC.
    """
    a = _f8(f)
    ac = _corr_colwise(a[k:], a[:-k], MIN_OBS)
    return float(np.nanmean(ac))


def evaluate(factor, close, mask=None, horizons=(1, 4, 24), lag_bars=1,
             returns_1bar=None, rho=None, name="", n_draws=0, seed=0):
    """Full IC report for one factor across several forward horizons."""
    f = apply_mask(lag(factor, lag_bars), mask)
    c = _f8(close)

    if rho is None:
        r1 = returns_1bar if returns_1bar is not None else forward_return(c, 1)
        rho, _ = average_pair_correlation(apply_mask(r1, mask))

    f_prep = prep_series(f) if n_draws else None
    rows = []
    for h in horizons:
        r = apply_mask(forward_return(c, h), mask)
        ic_cs = rank_ic_cross_section(f, r)
        # Floor at h-1 for the mechanical overlap, then let the data ask for
        # more when the factor itself is persistent.
        band = max(h - 1, andrews_band(ic_cs))
        mu, se, t, n = newey_west(ic_cs, band=band)
        sd = float(np.nanstd(ic_cs, ddof=1)) if n > 1 else np.nan
        icir = mu / sd if sd and np.isfinite(sd) and sd > 0 else np.nan

        null = (ts_null_test(f, r, f_prep=f_prep) if n_draws
                else {"z": np.nan, "p": np.nan, "null_sd": np.nan,
                      "ic_ts_shift": np.nan, "n_draws": 0})

        ic_ts, obs = rank_ic_per_symbol(f, r)
        valid = np.isfinite(ic_ts)
        n_sym = int(valid.sum())
        ts_mu = float(np.nanmean(ic_ts)) if n_sym else np.nan
        ts_sd = float(np.nanstd(ic_ts, ddof=1)) if n_sym > 1 else np.nan
        n_eff = effective_count(n_sym, rho)
        ts_t = (ts_mu / (ts_sd / np.sqrt(n_eff))
                if n_sym > 1 and ts_sd and ts_sd > 0 else np.nan)
        pos = float(np.mean(ic_ts[valid] > 0)) if n_sym else np.nan

        rows.append({
            "name": name, "horizon": h,
            "ic_cs": mu, "ic_cs_se": se, "ic_cs_t": t, "icir": icir, "n_bars": n,
            "ic_ts": ts_mu, "ic_ts_t": ts_t, "n_symbols": n_sym,
            "n_eff_symbols": n_eff, "frac_positive": pos,
            "ic_ts_shift": null["ic_ts_shift"],
            "ic_ts_z": null["z"], "ic_ts_p": null["p"],
            "ic_ts_null_sd": null["null_sd"], "n_draws": null["n_draws"],
            "rho_pairs": rho, "nw_band": band,
            # Two gates, because they answer different questions and a factor
            # can pass one while failing the other.
            #
            # passes_ts: the pooled per-symbol IC judged against its own
            # circular-shift null. This is the time-series question -- does the
            # factor predict a pair's own next move -- and it only exists when
            # the null was actually simulated (n_draws > 0).
            #
            # passes_cs: one observation per bar, Newey-West with an Andrews
            # bandwidth. A factor can score a huge t here and nothing on
            # passes_ts; that is a ranking effect (the low-volatility premium
            # does exactly this), useful for picking a whitelist but not for
            # timing a single pair.
            #
            # The pooled per-symbol t computed from n_eff is reported and never
            # gates: with rho near 0.45 its effective sample size is about two.
            "passes_ts": bool(np.isfinite(null["z"]) and abs(null["z"]) > T_GATE),
            "passes_cs": bool(np.isfinite(t) and abs(t) > T_GATE
                              and np.isfinite(ts_mu) and np.sign(t) == np.sign(ts_mu)),
            "passes": bool((np.isfinite(null["z"]) and abs(null["z"]) > T_GATE)
                           or (np.isfinite(t) and abs(t) > T_GATE
                               and np.isfinite(ts_mu)
                               and np.sign(t) == np.sign(ts_mu))),
        })
    return rows
