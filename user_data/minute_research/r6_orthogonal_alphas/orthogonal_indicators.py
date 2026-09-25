"""Vectorized Numba JIT indicators for orthogonal alphas (r6_orthogonal_alphas)."""
import numba as nb
import numpy as np


@nb.njit(cache=True)
def nb_rolling_sum(arr: np.ndarray, period: int) -> np.ndarray:
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        s = 0.0
        cnt = 0
        for i in range(T):
            v = arr[i, j]
            if not np.isnan(v):
                s += v
                cnt += 1
            if i >= period:
                old = arr[i - period, j]
                if not np.isnan(old):
                    s -= old
                    cnt -= 1
            if i >= period - 1 and cnt > 0:
                out[i, j] = s
    return out


@nb.njit(cache=True)
def nb_rolling_max(arr: np.ndarray, period: int) -> np.ndarray:
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        for i in range(period - 1, T):
            m = -1e18
            for k in range(i - period + 1, i + 1):
                v = arr[k, j]
                if v > m:
                    m = v
            out[i, j] = m
    return out


@nb.njit(cache=True)
def nb_rolling_min(arr: np.ndarray, period: int) -> np.ndarray:
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        for i in range(period - 1, T):
            m = 1e18
            for k in range(i - period + 1, i + 1):
                v = arr[k, j]
                if v < m:
                    m = v
            out[i, j] = m
    return out


@nb.njit(cache=True)
def nb_sma(arr: np.ndarray, period: int) -> np.ndarray:
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        s = 0.0
        valid_cnt = 0
        for i in range(T):
            val = arr[i, j]
            if not np.isnan(val):
                s += val
                valid_cnt += 1
            if i >= period:
                old_val = arr[i - period, j]
                if not np.isnan(old_val):
                    s -= old_val
                    valid_cnt -= 1
            if i >= period - 1 and valid_cnt == period:
                out[i, j] = s / period
    return out


@nb.njit(cache=True)
def nb_ema(arr: np.ndarray, period: int) -> np.ndarray:
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    alpha = 2.0 / (period + 1.0)
    for j in range(N):
        start_i = -1
        for i in range(T):
            if not np.isnan(arr[i, j]):
                start_i = i
                break
        if start_i == -1 or start_i + period > T:
            continue
        s = 0.0
        for i in range(start_i, start_i + period):
            s += arr[i, j]
        prev = s / period
        out[start_i + period - 1, j] = prev
        for i in range(start_i + period, T):
            val = arr[i, j]
            if np.isnan(val):
                out[i, j] = prev
            else:
                prev = alpha * val + (1.0 - alpha) * prev
                out[i, j] = prev
    return out


@nb.njit(cache=True)
def nb_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    T, N = close.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    alpha = 1.0 / period
    for j in range(N):
        start_i = -1
        for i in range(T):
            if not np.isnan(close[i, j]):
                start_i = i
                break
        if start_i == -1 or start_i + period >= T:
            continue
        sum_gain = 0.0
        sum_loss = 0.0
        for i in range(start_i + 1, start_i + period + 1):
            diff = close[i, j] - close[i - 1, j]
            if diff > 0:
                sum_gain += diff
            else:
                sum_loss -= diff
        avg_gain = sum_gain / period
        avg_loss = sum_loss / period
        rs = avg_gain / (avg_loss + 1e-12)
        out[start_i + period, j] = 100.0 - (100.0 / (1.0 + rs))
        for i in range(start_i + period + 1, T):
            diff = close[i, j] - close[i - 1, j]
            gain = diff if diff > 0 else 0.0
            loss = -diff if diff < 0 else 0.0
            avg_gain = alpha * gain + (1.0 - alpha) * avg_gain
            avg_loss = alpha * loss + (1.0 - alpha) * avg_loss
            rs = avg_gain / (avg_loss + 1e-12)
            out[i, j] = 100.0 - (100.0 / (1.0 + rs))
    return out


@nb.njit(cache=True)
def nb_semi_variance(returns: np.ndarray, period: int = 24):
    """Calculates Upside Semi-Variance (USV) and Downside Semi-Variance (DSV)."""
    T, N = returns.shape
    usv = np.full((T, N), np.nan, dtype=np.float64)
    dsv = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        for i in range(period - 1, T):
            u_sum = 0.0
            d_sum = 0.0
            valid_cnt = 0
            for k in range(i - period + 1, i + 1):
                r = returns[k, j]
                if not np.isnan(r):
                    valid_cnt += 1
                    if r > 0:
                        u_sum += r * r
                    elif r < 0:
                        d_sum += r * r
            if valid_cnt >= period // 2:
                usv[i, j] = u_sum / valid_cnt
                dsv[i, j] = d_sum / valid_cnt
    return usv, dsv
