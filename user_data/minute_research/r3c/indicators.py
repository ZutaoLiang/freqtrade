"""Ultra-fast vectorized & Numba JIT indicators for multi-timeframe research."""
import numba as nb
import numpy as np


@nb.njit(cache=True)
def nb_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Vectorized EMA across (T, N) 2D array."""
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
    """Rolling SMA across (T, N)."""
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
    """True Range and ATR (T, N)."""
    T, N = high.shape
    tr = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        for i in range(T):
            h = high[i, j]
            l = low[i, j]
            if np.isnan(h) or np.isnan(l):
                continue
            if i == 0 or np.isnan(close[i - 1, j]):
                tr[i, j] = h - l
            else:
                c_prev = close[i - 1, j]
                tr[i, j] = max(h - l, max(abs(h - c_prev), abs(l - c_prev)))
    return nb_ema(tr, period)


@nb.njit(cache=True)
def nb_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI (T, N)."""
    T, N = close.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        start_i = -1
        for i in range(T):
            if not np.isnan(close[i, j]):
                start_i = i
                break
        if start_i == -1 or start_i + period >= T:
            continue
        gains = 0.0
        losses = 0.0
        for i in range(start_i + 1, start_i + period + 1):
            diff = close[i, j] - close[i - 1, j]
            if diff > 0:
                gains += diff
            else:
                losses -= diff
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            out[start_i + period, j] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[start_i + period, j] = 100.0 - (100.0 / (1.0 + rs))

        for i in range(start_i + period + 1, T):
            diff = close[i, j] - close[i - 1, j]
            gain = diff if diff > 0 else 0.0
            loss = -diff if diff < 0 else 0.0
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            if avg_loss == 0:
                out[i, j] = 100.0
            else:
                rs = avg_gain / avg_loss
                out[i, j] = 100.0 - (100.0 / (1.0 + rs))
    return out


@nb.njit(cache=True)
def nb_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10, mult: float = 3.0):
    """SuperTrend indicator (T, N). Returns trend (1 for bullish, -1 for bearish), and band."""
    T, N = high.shape
    atr = nb_atr(high, low, close, period)
    trend = np.zeros((T, N), dtype=np.int8)
    band = np.full((T, N), np.nan, dtype=np.float64)
    
    for j in range(N):
        current_trend = 1
        upper_band = 0.0
        lower_band = 0.0
        inited = False
        
        for i in range(period, T):
            if np.isnan(atr[i, j]) or np.isnan(close[i, j]):
                continue
            hl2 = (high[i, j] + low[i, j]) * 0.5
            matr = mult * atr[i, j]
            basic_upper = hl2 + matr
            basic_lower = hl2 - matr
            
            if not inited:
                upper_band = basic_upper
                lower_band = basic_lower
                inited = True
                trend[i, j] = current_trend
                band[i, j] = lower_band if current_trend == 1 else upper_band
                continue
                
            prev_upper = upper_band
            prev_lower = lower_band
            prev_close = close[i - 1, j]
            
            if basic_lower > prev_lower or prev_close < prev_lower:
                lower_band = basic_lower
            else:
                lower_band = prev_lower
                
            if basic_upper < prev_upper or prev_close > prev_upper:
                upper_band = basic_upper
            else:
                upper_band = prev_upper
                
            if current_trend == 1:
                if close[i, j] < lower_band:
                    current_trend = -1
            else:
                if close[i, j] > upper_band:
                    current_trend = 1
                    
            trend[i, j] = current_trend
            band[i, j] = lower_band if current_trend == 1 else upper_band
            
    return trend, band


@nb.njit(cache=True)
def nb_donchian(high: np.ndarray, low: np.ndarray, period: int):
    """Donchian channel upper and lower bands (T, N)."""
    T, N = high.shape
    upper = np.full((T, N), np.nan, dtype=np.float64)
    lower = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        for i in range(period, T):
            m_h = -1e9
            m_l = 1e9
            valid = True
            for k in range(i - period, i):
                h_val = high[k, j]
                l_val = low[k, j]
                if np.isnan(h_val) or np.isnan(l_val):
                    valid = False
                    break
                if h_val > m_h:
                    m_h = h_val
                if l_val < m_l:
                    m_l = l_val
            if valid:
                upper[i, j] = m_h
                lower[i, j] = m_l
    return upper, lower


@nb.njit(cache=True)
def nb_bollinger(close: np.ndarray, period: int = 20, mult: float = 2.0):
    """Bollinger Bands (T, N). Returns mid, upper, lower, bandwidth, pct_b."""
    sma = nb_sma(close, period)
    T, N = close.shape
    upper = np.full((T, N), np.nan, dtype=np.float64)
    lower = np.full((T, N), np.nan, dtype=np.float64)
    bbw = np.full((T, N), np.nan, dtype=np.float64)
    pct_b = np.full((T, N), np.nan, dtype=np.float64)
    
    for j in range(N):
        for i in range(period - 1, T):
            m = sma[i, j]
            if np.isnan(m):
                continue
            v = 0.0
            for k in range(i - period + 1, i + 1):
                diff = close[k, j] - m
                v += diff * diff
            std = np.sqrt(v / period)
            up = m + mult * std
            dn = m - mult * std
            upper[i, j] = up
            lower[i, j] = dn
            if m > 0:
                bbw[i, j] = (2.0 * mult * std) / m
            if up > dn:
                pct_b[i, j] = (close[i, j] - dn) / (up - dn)
    return sma, upper, lower, bbw, pct_b


@nb.njit(cache=True)
def nb_keltner(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20, mult: float = 1.5):
    """Keltner Channels (T, N). Mid is EMA(close, period), bands are EMA +/- mult * ATR."""
    mid = nb_ema(close, period)
    atr = nb_atr(high, low, close, period)
    upper = mid + mult * atr
    lower = mid - mult * atr
    return mid, upper, lower


@nb.njit(cache=True)
def nb_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    """ADX indicator (T, N)."""
    T, N = high.shape
    adx = np.full((T, N), np.nan, dtype=np.float64)
    plus_di = np.full((T, N), np.nan, dtype=np.float64)
    minus_di = np.full((T, N), np.nan, dtype=np.float64)
    
    atr = nb_atr(high, low, close, period)
    
    # Compute +DM, -DM
    plus_dm = np.zeros((T, N), dtype=np.float64)
    minus_dm = np.zeros((T, N), dtype=np.float64)
    for j in range(N):
        for i in range(1, T):
            up_move = high[i, j] - high[i - 1, j]
            dn_move = low[i - 1, j] - low[i, j]
            if up_move > dn_move and up_move > 0:
                plus_dm[i, j] = up_move
            if dn_move > up_move and dn_move > 0:
                minus_dm[i, j] = dn_move
                
    smooth_plus = nb_ema(plus_dm, period)
    smooth_minus = nb_ema(minus_dm, period)
    
    dx = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        for i in range(period, T):
            a = atr[i, j]
            if a > 0:
                p_di = 100.0 * smooth_plus[i, j] / a
                m_di = 100.0 * smooth_minus[i, j] / a
                plus_di[i, j] = p_di
                minus_di[i, j] = m_di
                di_sum = p_di + m_di
                if di_sum > 0:
                    dx[i, j] = 100.0 * abs(p_di - m_di) / di_sum
                    
    adx = nb_ema(dx, period)
    return adx, plus_di, minus_di


@nb.njit(cache=True)
def nb_rolling_zscore(arr: np.ndarray, period: int = 24):
    """Rolling z-score across (T, N)."""
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    sma = nb_sma(arr, period)
    for j in range(N):
        for i in range(period - 1, T):
            m = sma[i, j]
            if np.isnan(m):
                continue
            v = 0.0
            valid = True
            for k in range(i - period + 1, i + 1):
                val = arr[k, j]
                if np.isnan(val):
                    valid = False
                    break
                diff = val - m
                v += diff * diff
            if valid:
                std = np.sqrt(v / period)
                if std > 1e-8:
                    out[i, j] = (arr[i, j] - m) / std
                else:
                    out[i, j] = 0.0
    return out


@nb.njit(cache=True)
def nb_cmf(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 20):
    """Chaikin Money Flow (CMF) (T, N)."""
    T, N = high.shape
    mfv = np.zeros((T, N), dtype=np.float64)
    for j in range(N):
        for i in range(T):
            hl = high[i, j] - low[i, j]
            if hl > 0 and volume[i, j] > 0:
                clv = ((close[i, j] - low[i, j]) - (high[i, j] - close[i, j])) / hl
                mfv[i, j] = clv * volume[i, j]
    sum_mfv = nb_sma(mfv, period) * period
    sum_vol = nb_sma(volume, period) * period
    cmf = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        for i in range(period - 1, T):
            if sum_vol[i, j] > 0:
                cmf[i, j] = sum_mfv[i, j] / sum_vol[i, j]
    return cmf


@nb.njit(cache=True)
def nb_choppiness(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    """Choppiness Index (CHOP) (T, N)."""
    T, N = high.shape
    chop = np.full((T, N), np.nan, dtype=np.float64)
    tr = np.zeros((T, N), dtype=np.float64)
    for j in range(N):
        for i in range(T):
            h = high[i, j]
            l = low[i, j]
            if i == 0:
                tr[i, j] = h - l
            else:
                c_prev = close[i - 1, j]
                tr[i, j] = max(h - l, max(abs(h - c_prev), abs(l - c_prev)))
                
    for j in range(N):
        for i in range(period - 1, T):
            sum_tr = 0.0
            max_h = -1e9
            min_l = 1e9
            valid = True
            for k in range(i - period + 1, i + 1):
                h = high[k, j]
                l = low[k, j]
                if np.isnan(h) or np.isnan(l):
                    valid = False
                    break
                sum_tr += tr[k, j]
                if h > max_h:
                    max_h = h
                if l < min_l:
                    min_l = l
            if valid and max_h > min_l and sum_tr > 0:
                denom = max_h - min_l
                chop[i, j] = 100.0 * np.log10(sum_tr / denom) / np.log10(period)
    return chop


@nb.njit(cache=True)
def nb_rolling_std(arr: np.ndarray, period: int = 24) -> np.ndarray:
    """Ultra-fast rolling standard deviation across (T, N)."""
    T, N = arr.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        s = 0.0
        sq = 0.0
        for i in range(T):
            val = arr[i, j]
            if not np.isnan(val):
                s += val
                sq += val * val
            if i >= period:
                old = arr[i - period, j]
                if not np.isnan(old):
                    s -= old
                    sq -= old * old
            if i >= period - 1:
                var = (sq / period) - (s / period) ** 2
                out[i, j] = np.sqrt(max(0.0, var))
    return out
