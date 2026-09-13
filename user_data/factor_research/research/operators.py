"""Time-series operators over (time, symbol) panels.

Every operator takes a 2-D float array shaped (T, N) and returns the same
shape, computing down axis=0 so one call covers the whole universe. That is the
reason for the panel layout: a per-pair Python loop over 860 symbols costs
roughly three orders of magnitude more than one vectorised call.

Windows require full history (`min_count == window`), so the warm-up region and
any pair not yet listed stay NaN rather than silently producing a value from a
partial window -- a partial-window factor is a different factor.

Backed by bottleneck's C move_* kernels where they exist; the rest are derived
from rolling sums, which keeps everything vectorised.
"""
import bottleneck as bn
import warnings

import numpy as np
from scipy.ndimage import convolve1d

EPS = 1e-12


def _f8(x):
    """Accumulate in float64. Rolling sums of float32 lose precision fast."""
    return np.asarray(x, dtype="float64")


def _full(x, w):
    return bn.move_sum(x, window=w, min_count=w, axis=0)


# --- shifts ---------------------------------------------------------------

def ts_delay(x, w):
    out = np.full_like(_f8(x), np.nan)
    if w < len(x):
        out[w:] = _f8(x)[:-w]
    return out


def ts_delta(x, w):
    return _f8(x) - ts_delay(x, w)


def ts_pct_change(x, w):
    prev = ts_delay(x, w)
    return _f8(x) / np.where(np.abs(prev) < EPS, np.nan, prev) - 1.0


# --- moments --------------------------------------------------------------

def ts_sum(x, w):
    return _full(_f8(x), w)


def ts_mean(x, w):
    return bn.move_mean(_f8(x), window=w, min_count=w, axis=0)


def ts_std(x, w):
    return bn.move_std(_f8(x), window=w, min_count=w, ddof=1, axis=0)


def ts_var(x, w):
    return bn.move_var(_f8(x), window=w, min_count=w, ddof=1, axis=0)


def ts_median(x, w):
    return bn.move_median(_f8(x), window=w, min_count=w, axis=0)


def ts_zscore(x, w):
    s = ts_std(x, w)
    return (_f8(x) - ts_mean(x, w)) / np.where(s < EPS, np.nan, s)


def _central_moments(x, w):
    """Central moments from raw power sums.

    Centring on the rolling mean instead would make the summand NaN through the
    warm-up, so move_sum would need a further w-1 bars and the operator would
    start at 2w-1 rather than w-1 -- a silently different warm-up from every
    other operator here.
    """
    # Central moments are shift-invariant, so subtracting a per-column constant
    # is exact -- and it is what keeps the power sums usable. On raw prices near
    # 1e5, x**4 reaches 1e20 and m4 is a difference of nearly equal giants;
    # centring first drops the operands to the scale of the spread. nanmean
    # rather than nanmedian: any constant is equally exact here, and the median
    # sorts 147M values for no gain.
    d = _f8(x)
    with warnings.catch_warnings():
        # Columns that are entirely NaN -- symbols not yet listed -- would warn
        # once per call; their output is NaN either way.
        warnings.simplefilter("ignore", RuntimeWarning)
        d = d - np.nanmean(d, axis=0, keepdims=True)
    m = _full(d, w) / w
    s2, s3, s4 = _full(d ** 2, w), _full(d ** 3, w), _full(d ** 4, w)
    m2 = s2 / w - m ** 2
    m3 = s3 / w - 3 * m * s2 / w + 2 * m ** 3
    m4 = s4 / w - 4 * m * s3 / w + 6 * m ** 2 * s2 / w - 3 * m ** 4
    return m2, m3, m4


def ts_skew(x, w):
    """Sample skewness, bias-corrected to match pandas Series.skew()."""
    m2, m3, _ = _central_moments(x, w)
    m2 = np.where(m2 < EPS, np.nan, m2)
    g1 = m3 / m2 ** 1.5
    return g1 * np.sqrt(w * (w - 1.0)) / (w - 2.0)


def ts_kurt(x, w):
    """Excess sample kurtosis, bias-corrected to match pandas Series.kurt()."""
    m2, _, m4 = _central_moments(x, w)
    m2 = np.where(m2 < EPS, np.nan, m2)
    g2 = m4 / m2 ** 2 - 3.0
    return (w - 1.0) / ((w - 2.0) * (w - 3.0)) * ((w + 1.0) * g2 + 6.0)


# --- extrema --------------------------------------------------------------

def ts_min(x, w):
    return bn.move_min(_f8(x), window=w, min_count=w, axis=0)


def ts_max(x, w):
    return bn.move_max(_f8(x), window=w, min_count=w, axis=0)


def ts_range(x, w):
    return ts_max(x, w) - ts_min(x, w)


def ts_argmin(x, w):
    """Bars since the window minimum; 0 means the minimum is the current bar."""
    return bn.move_argmin(_f8(x), window=w, min_count=w, axis=0)


def ts_argmax(x, w):
    return bn.move_argmax(_f8(x), window=w, min_count=w, axis=0)


def ts_rank(x, w):
    """Percentile rank of the current value in its trailing window, in (0, 1].

    bottleneck's move_rank is a symmetric [-1, 1] score over average ranks, so
    it needs converting rather than merely rescaling: a plain (r+1)/2 maps the
    window minimum to 0 instead of 1/w and mishandles ties.
    """
    r = bn.move_rank(_f8(x), window=w, min_count=w, axis=0)
    avg_rank = r * (w - 1.0) / 2.0 + (w + 1.0) / 2.0
    return avg_rank / w


def ts_count_above_mean(x, w):
    d = _f8(x)
    return _full((d > ts_mean(d, w)).astype("float64"), w)


# --- pairwise -------------------------------------------------------------

def ts_cov(x, y, w):
    a, b = _f8(x), _f8(y)
    return (_full(a * b, w) - _full(a, w) * _full(b, w) / w) / (w - 1)


def ts_corr(x, y, w):
    sx, sy = ts_std(x, w), ts_std(y, w)
    denom = sx * sy
    return ts_cov(x, y, w) / np.where(denom < EPS, np.nan, denom)


def ts_beta(x, y, w):
    """Slope of x regressed on y."""
    v = ts_var(y, w)
    return ts_cov(x, y, w) / np.where(v < EPS, np.nan, v)


# --- shape ----------------------------------------------------------------

def ts_slope(x, w):
    """Least-squares slope against bar index -- trend per bar."""
    t = np.arange(w, dtype="float64")
    var_t = t.var(ddof=1)
    idx = np.arange(len(x), dtype="float64")[:, None]
    return ts_cov(x, np.broadcast_to(idx, np.shape(x)), w) / var_t


def decay_linear(x, w):
    """Weighted mean with linearly decaying weights, heaviest on the newest bar.

    One FIR pass rather than w shifted copies: the naive form allocated a full
    (T, N) array per tap, which dominated the whole operator suite at 5m.
    NaNs are zero-filled for the convolution and masked back afterwards, since
    convolve1d has no notion of a minimum count.
    """
    d = _f8(x)
    weights = np.arange(1, w + 1, dtype="float64")
    weights /= weights.sum()
    finite = np.isfinite(d)
    filled = np.where(finite, d, 0.0)
    # convolve1d correlates with reversed weights, so flip to keep the heaviest
    # weight on the newest bar; origin shifts the window to trail rather than centre.
    acc = convolve1d(filled, weights[::-1], axis=0, mode="constant", cval=0.0,
                     origin=-(w // 2) if w % 2 else -(w // 2) + 0)
    valid = bn.move_sum(finite.astype("float64"), window=w, min_count=w, axis=0)
    return np.where(valid >= w, acc, np.nan)


def ts_ema(x, halflife):
    """Exponential mean; unlike the rolling operators this has no hard window."""
    d = _f8(x)
    alpha = 1.0 - np.exp(np.log(0.5) / halflife)
    out = np.empty_like(d)
    out[0] = d[0]
    for i in range(1, len(d)):
        prev = np.where(np.isnan(out[i - 1]), d[i], out[i - 1])
        out[i] = np.where(np.isnan(d[i]), prev, alpha * d[i] + (1 - alpha) * prev)
    return out


def ts_autocorr(x, w, lag=1):
    return ts_corr(x, ts_delay(x, lag), w)


# --- elementwise helpers --------------------------------------------------

def safe_div(a, b):
    b = _f8(b)
    return _f8(a) / np.where(np.abs(b) < EPS, np.nan, b)


def ts_rsi(x, w):
    """Wilder's RSI on the *changes* of x, scaled to (0, 100).

    Takes a level series and differences it internally, so ts_rsi(close, 14) is
    the familiar indicator. Uses a simple rolling mean of gains and losses
    rather than Wilder's recursive smoothing: the recursion has no fixed window
    and would give the operator a different warm-up rule from every other one
    here, which matters when specs are compared across windows.
    """
    d = _f8(x)
    delta = np.full_like(d, np.nan)
    delta[1:] = d[1:] - d[:-1]
    gain = np.where(np.isfinite(delta), np.maximum(delta, 0.0), np.nan)
    loss = np.where(np.isfinite(delta), np.maximum(-delta, 0.0), np.nan)
    avg_gain = bn.move_mean(gain, window=w, min_count=w, axis=0)
    avg_loss = bn.move_mean(loss, window=w, min_count=w, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    # All-gain windows give an infinite RS; the limit is 100, and all-flat
    # windows have no defined RSI at all.
    out = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, out)
    return np.where((avg_loss == 0) & (avg_gain == 0), np.nan, out)


def sign_log(x):
    """Signed log magnitude -- compresses heavy tails without losing direction."""
    d = _f8(x)
    return np.sign(d) * np.log1p(np.abs(d))


UNARY = {
    "ts_mean": ts_mean, "ts_std": ts_std, "ts_var": ts_var, "ts_sum": ts_sum,
    "ts_median": ts_median, "ts_zscore": ts_zscore, "ts_skew": ts_skew,
    "ts_kurt": ts_kurt, "ts_min": ts_min, "ts_max": ts_max, "ts_range": ts_range,
    "ts_argmin": ts_argmin, "ts_argmax": ts_argmax, "ts_rank": ts_rank,
    "ts_count_above_mean": ts_count_above_mean, "ts_slope": ts_slope,
    "decay_linear": decay_linear, "ts_delta": ts_delta,
    "ts_pct_change": ts_pct_change, "ts_delay": ts_delay,
    "ts_autocorr": ts_autocorr,
}

BINARY = {"ts_corr": ts_corr, "ts_cov": ts_cov, "ts_beta": ts_beta}
