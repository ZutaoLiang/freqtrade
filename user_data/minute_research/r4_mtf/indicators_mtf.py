"""Ultra-fast vectorized & Numba JIT indicators for multi-timeframe research (r4_mtf)."""
import numba as nb
import numpy as np


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
def nb_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    T, N = high.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    alpha = 1.0 / period
    for j in range(N):
        prev_atr = 0.0
        start_i = -1
        for i in range(T):
            if not np.isnan(high[i, j]) and not np.isnan(low[i, j]) and not np.isnan(close[i, j]):
                start_i = i
                break
        if start_i == -1 or start_i + period >= T:
            continue
        s = 0.0
        for i in range(start_i, start_i + period):
            if i == start_i:
                tr = high[i, j] - low[i, j]
            else:
                c_prev = close[i - 1, j]
                tr = max(high[i, j] - low[i, j], abs(high[i, j] - c_prev), abs(low[i, j] - c_prev))
            s += tr
        prev_atr = s / period
        out[start_i + period - 1, j] = prev_atr
        for i in range(start_i + period, T):
            c_prev = close[i - 1, j]
            tr = max(high[i, j] - low[i, j], abs(high[i, j] - c_prev), abs(low[i, j] - c_prev))
            prev_atr = alpha * tr + (1.0 - alpha) * prev_atr
            out[i, j] = prev_atr
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
def nb_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10, mult: float = 3.0):
    T, N = high.shape
    atr = nb_atr(high, low, close, period)
    trend = np.full((T, N), np.nan, dtype=np.float64)
    st = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        in_trend = 1.0
        prev_upper = 0.0
        prev_lower = 0.0
        prev_st = 0.0
        init = False
        for i in range(period, T):
            hl2 = 0.5 * (high[i, j] + low[i, j])
            a = atr[i, j]
            if np.isnan(a):
                continue
            basic_upper = hl2 + mult * a
            basic_lower = hl2 - mult * a
            c_prev = close[i - 1, j]
            final_upper = basic_upper if (basic_upper < prev_upper or c_prev > prev_upper) else prev_upper
            final_lower = basic_lower if (basic_lower > prev_lower or c_prev < prev_lower) else prev_lower
            if not init:
                in_trend = 1.0 if close[i, j] > final_upper else -1.0
                prev_st = final_lower if in_trend == 1.0 else final_upper
                init = True
            else:
                if in_trend == 1.0:
                    if close[i, j] < final_lower:
                        in_trend = -1.0
                        prev_st = final_upper
                    else:
                        prev_st = final_lower
                else:
                    if close[i, j] > final_upper:
                        in_trend = 1.0
                        prev_st = final_lower
                    else:
                        prev_st = final_upper
            prev_upper = final_upper
            prev_lower = final_lower
            trend[i, j] = in_trend
            st[i, j] = prev_st
    return trend, st


@nb.njit(cache=True)
def nb_bollinger(close: np.ndarray, period: int = 20, mult: float = 2.0):
    mid = nb_sma(close, period)
    T, N = close.shape
    upper = np.full((T, N), np.nan, dtype=np.float64)
    lower = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        for i in range(period - 1, T):
            m = mid[i, j]
            if np.isnan(m):
                continue
            var = 0.0
            for k in range(i - period + 1, i + 1):
                diff = close[k, j] - m
                var += diff * diff
            std = np.sqrt(var / period)
            upper[i, j] = m + mult * std
            lower[i, j] = m - mult * std
    return mid, upper, lower


@nb.njit(cache=True)
def nb_keltner(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20, mult: float = 1.5):
    mid = nb_ema(close, period)
    atr = nb_atr(high, low, close, period)
    upper = mid + mult * atr
    lower = mid - mult * atr
    return mid, upper, lower


@nb.njit(cache=True)
def nb_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = nb_ema(close, fast)
    ema_slow = nb_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = nb_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


@nb.njit(cache=True)
def nb_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    hh = nb_rolling_max(high, period)
    ll = nb_rolling_min(low, period)
    wr = -100.0 * (hh - close) / (hh - ll + 1e-12)
    return wr
