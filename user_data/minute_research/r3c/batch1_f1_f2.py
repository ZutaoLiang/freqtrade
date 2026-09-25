"""Batch 1: Family 1 (R51-R70: MTF Trend & Momentum) & Family 2 (R71-R90: MTF Mean Reversion)."""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append("/root/freqtrade/user_data/minute_research/r3c")
from engine200 import PanelData, evaluate_strategy
import indicators as ind


def run_batch1():
    print("Loading 1h, 4h, 1d panels...")
    t0 = time.time()
    p1h = PanelData("1h")
    p4h = PanelData("4h")
    p1d = PanelData("1d")
    print(f"Panels loaded in {time.time()-t0:.2f}s.")
    
    # Pre-extract arrays for 1h
    c1h = np.array(p1h.close)
    o1h = np.array(p1h.open)
    h1h = np.array(p1h.high)
    l1h = np.array(p1h.low)
    v1h = np.array(p1h.volume)
    
    # 4h arrays
    c4h = np.array(p4h.close)
    h4h = np.array(p4h.high)
    l4h = np.array(p4h.low)
    
    # 1d arrays
    c1d = np.array(p1d.close)
    
    print("Pre-computing multi-timeframe indicators...")
    # 1d indicators mapped to 1h causally
    sma100_1d = ind.nb_sma(c1d, 100)
    sma200_1d = ind.nb_sma(c1d, 200)
    m_sma100_1d = p1h.map_htf_to_ltf(sma100_1d, "1d")
    m_sma200_1d = p1h.map_htf_to_ltf(sma200_1d, "1d")
    m_c1d = p1h.map_htf_to_ltf(c1d, "1d")
    
    # 4h indicators mapped to 1h causally
    ema20_4h = ind.nb_ema(c4h, 20)
    ema50_4h = ind.nb_ema(c4h, 50)
    ema100_4h = ind.nb_ema(c4h, 100)
    m_ema20_4h = p1h.map_htf_to_ltf(ema20_4h, "4h")
    m_ema50_4h = p1h.map_htf_to_ltf(ema50_4h, "4h")
    m_ema100_4h = p1h.map_htf_to_ltf(ema100_4h, "4h")
    m_c4h = p1h.map_htf_to_ltf(c4h, "4h")
    
    st_trend_4h, st_band_4h = ind.nb_supertrend(h4h, l4h, c4h, 10, 3.0)
    m_st_trend_4h = p1h.map_htf_to_ltf(st_trend_4h.astype(np.float64), "4h")
    
    adx_4h, pdi_4h, mdi_4h = ind.nb_adx(h4h, l4h, c4h, 14)
    m_adx_4h = p1h.map_htf_to_ltf(adx_4h, "4h")
    
    don_u_4h, don_l_4h = ind.nb_donchian(h4h, l4h, 20)
    m_don_u_4h = p1h.map_htf_to_ltf(don_u_4h, "4h")
    m_don_l_4h = p1h.map_htf_to_ltf(don_l_4h, "4h")
    
    # 1h core indicators
    ema12_1h = ind.nb_ema(c1h, 12)
    ema20_1h = ind.nb_ema(c1h, 20)
    ema26_1h = ind.nb_ema(c1h, 26)
    ema50_1h = ind.nb_ema(c1h, 50)
    atr14_1h = ind.nb_atr(h1h, l1h, c1h, 14)
    rsi14_1h = ind.nb_rsi(c1h, 14)
    don_u_24, don_l_24 = ind.nb_donchian(h1h, l1h, 24)
    sma_bb, up_bb, low_bb, bbw_bb, pct_b_1h = ind.nb_bollinger(c1h, 20, 2.0)
    kelt_mid, kelt_u, kelt_l = ind.nb_keltner(h1h, l1h, c1h, 20, 1.5)
    cmf_1h = ind.nb_cmf(h1h, l1h, c1h, v1h, 20)
    chop_1h = ind.nb_choppiness(h1h, l1h, c1h, 14)
    
    print("Indicators ready. Commencing R51-R90 backtests on TRAIN...")
    results = []
    
    # Helper to test and log
    def test_round(r_id, name, sig_l, sig_s, hold, sl=0.0, tp=0.0):
        seg = os.environ.get("RUN_SEG", "TRAIN")
        res = evaluate_strategy(p1h, sig_l, sig_s, hold_bars=hold, seg=seg, sl_pct=sl, tp_pct=tp)
        status = "REJECT"
        if res.get("mean_bp", 0) > 0 and res.get("day_t", 0) >= 1.5 and res.get("per_day", 0) >= 0.2:
            status = f"PASS_{seg}"
        elif res.get("mean_bp", 0) > 0:
            status = "PROFITABLE"
        row = {
            "round": r_id, "name": name, "n": res["n"], "mean_bp": res["mean_bp"],
            "med_bp": res["med_bp"], "win": res["win"], "pf": res["pf"],
            "day_t": res["day_t"], "per_day": res["per_day"], "status": status
        }
        results.append(row)
        print(f"[{r_id:03d}] {name[:35]:35s} | n={res['n']:5d} | mean={res['mean_bp']:6.1f}bp | pf={res['pf']:5.2f} | day_t={res['day_t']:5.2f} |/d={res['per_day']:4.1f} | {status}")
        return row

    T, N = c1h.shape
    empty_sig = np.zeros((T, N), dtype=bool)
    
    # --- FAMILY 1: MTF Trend & Momentum (R51 - R70) ---
    # R51: MTF Moving Average Ribbon Consensus
    sig_l = (m_c1d > m_sma200_1d) & (m_ema50_4h > m_ema100_4h) & (c1h > ema20_1h) & (ema20_1h > ema50_1h)
    sig_s = (m_c1d < m_sma200_1d) & (m_ema50_4h < m_ema100_4h) & (c1h < ema20_1h) & (ema20_1h < ema50_1h)
    test_round(51, "MTF MA Ribbon Consensus", sig_l, sig_s, hold=24)
    
    # R52: MTF MA Ribbon Pullback Entry
    # In 1d uptrend & 4h uptrend, 1h dips below ema20 then reclaims
    c_prev = np.roll(c1h, 1, axis=0)
    reclaim_l = (c_prev < ema20_1h) & (c1h > ema20_1h)
    reclaim_s = (c_prev > ema20_1h) & (c1h < ema20_1h)
    sig_l = (m_c1d > m_sma200_1d) & (m_ema50_4h > m_ema100_4h) & reclaim_l
    sig_s = (m_c1d < m_sma200_1d) & (m_ema50_4h < m_ema100_4h) & reclaim_s
    test_round(52, "MTF MA Ribbon Pullback Reclaim", sig_l, sig_s, hold=24)
    
    # R53: MTF SuperTrend 4h Regime + 1h Momentum Follow
    st_cross_up = (c_prev < ema20_1h) & (c1h > ema20_1h)
    st_cross_dn = (c_prev > ema20_1h) & (c1h < ema20_1h)
    sig_l = (m_st_trend_4h == 1) & st_cross_up
    sig_s = (m_st_trend_4h == -1) & st_cross_dn
    test_round(53, "MTF SuperTrend 4h + 1h EMA Cross", sig_l, sig_s, hold=18)
    
    # R54: MTF SuperTrend 4h Regime + 1h Dip Buying (Long only)
    sig_l = (m_st_trend_4h == 1) & (rsi14_1h < 35) & (c1h > ema50_1h)
    test_round(54, "MTF SuperTrend 4h + 1h RSI Dip", sig_l, empty_sig, hold=12)
    
    # R55: MTF Donchian Channel 4h Breakout + 1d Macro Filter
    sig_l = (m_c1d > m_sma100_1d) & (c1h > m_don_u_4h)
    sig_s = (m_c1d < m_sma100_1d) & (c1h < m_don_l_4h)
    test_round(55, "MTF Donchian 4h Break + 1d Filter", sig_l, sig_s, hold=48)
    
    # R56: MTF Donchian 1h Breakout with Volatility Expansion
    atr_ma = ind.nb_sma(atr14_1h, 24)
    sig_l = (c1h > don_u_24) & (atr14_1h > atr_ma * 1.2)
    sig_s = (c1h < don_l_24) & (atr14_1h > atr_ma * 1.2)
    test_round(56, "MTF Donchian 1h Break + ATR Exp", sig_l, sig_s, hold=24)
    
    # R57: MTF Dual Thrust 1h Range Expansion
    dt_up = o1h + 0.5 * atr14_1h
    dt_dn = o1h - 0.5 * atr14_1h
    sig_l = (m_ema50_4h > m_ema100_4h) & (c1h > dt_up)
    sig_s = (m_ema50_4h < m_ema100_4h) & (c1h < dt_dn)
    test_round(57, "MTF Dual Thrust Range Exp", sig_l, sig_s, hold=12)
    
    # R58: MTF ADX Trend Strength + 1h EMA Cross
    sig_l = (m_adx_4h > 25) & (c_prev < ema12_1h) & (c1h > ema12_1h) & (ema12_1h > ema26_1h)
    sig_s = (m_adx_4h > 25) & (c_prev > ema12_1h) & (c1h < ema12_1h) & (ema12_1h < ema26_1h)
    test_round(58, "MTF 4h ADX Trend + 1h EMA Cross", sig_l, sig_s, hold=24)
    
    # R59: MTF Moving Average Slope Consensus
    slope_4h = (m_ema50_4h - np.roll(m_ema50_4h, 4, axis=0)) / m_ema50_4h
    slope_1h = (ema20_1h - np.roll(ema20_1h, 1, axis=0)) / ema20_1h
    sig_l = (slope_4h > 0.005) & (slope_1h > 0.002)
    sig_s = (slope_4h < -0.005) & (slope_1h < -0.002)
    test_round(59, "MTF MA Slope Consensus", sig_l, sig_s, hold=24)
    
    # R60: MTF Aroon Trend Onset (Using 48h High/Low proximity)
    aroon_up = (c1h == don_u_24)
    aroon_dn = (c1h == don_l_24)
    sig_l = (m_ema50_4h > m_ema100_4h) & aroon_up
    sig_s = (m_ema50_4h < m_ema100_4h) & aroon_dn
    test_round(60, "MTF Aroon Trend Onset", sig_l, sig_s, hold=24)
    
    # R61: MTF Guppy Multiple Moving Average Compression-Expansion
    g_fast = (ema12_1h > ema20_1h) & (ema20_1h > ema26_1h)
    g_slow = (ema26_1h > ema50_1h)
    sig_l = g_fast & g_slow & (m_c4h > m_ema50_4h)
    sig_s = (~g_fast) & (~g_slow) & (m_c4h < m_ema50_4h)
    test_round(61, "MTF Guppy GMMA Expansion", sig_l, sig_s, hold=36)
    
    # R62: MTF Kaufman Efficiency Ratio Trend Pulse
    net_chg = np.abs(c1h - np.roll(c1h, 24, axis=0))
    vol_sum = ind.nb_sma(np.abs(c1h - c_prev), 24) * 24.0
    er = np.where(vol_sum > 0, net_chg / vol_sum, 0.0)
    sig_l = (er > 0.6) & (c1h > c_prev) & (m_ema50_4h > m_ema100_4h)
    sig_s = (er > 0.6) & (c1h < c_prev) & (m_ema50_4h < m_ema100_4h)
    test_round(62, "MTF Kaufman ER Trend Pulse", sig_l, sig_s, hold=24)
    
    # R63: MTF MACD Histogram Burst
    macd_line = ema12_1h - ema26_1h
    macd_signal = ind.nb_ema(macd_line, 9)
    macd_hist = macd_line - macd_signal
    hist_prev = np.roll(macd_hist, 1, axis=0)
    sig_l = (m_st_trend_4h == 1) & (hist_prev < 0) & (macd_hist > 0)
    sig_s = (m_st_trend_4h == -1) & (hist_prev > 0) & (macd_hist < 0)
    test_round(63, "MTF MACD Hist Burst + 4h ST", sig_l, sig_s, hold=18)
    
    # R64: MTF 1h Keltner Upper Breakout with 4h Trend
    sig_l = (m_ema50_4h > m_ema100_4h) & (c1h > kelt_u)
    sig_s = (m_ema50_4h < m_ema100_4h) & (c1h < kelt_l)
    test_round(64, "MTF Keltner Breakout + 4h Trend", sig_l, sig_s, hold=24)
    
    # R65: MTF Keltner Upper Breakout + ATR Stop
    test_round(65, "MTF Keltner Break + SL/TP", sig_l, sig_s, hold=36, sl=0.03, tp=0.06)
    
    # R66: MTF Vortex Trend Onset (Using True Range and directional movements)
    sig_l = (c1h > kelt_mid) & (rsi14_1h > 55) & (m_c4h > m_ema50_4h)
    sig_s = (c1h < kelt_mid) & (rsi14_1h < 45) & (m_c4h < m_ema50_4h)
    test_round(66, "MTF Vortex Proxy Consensus", sig_l, sig_s, hold=24)
    
    # R67: MTF 3 Consecutive Trend Bars
    three_up = (c1h > o1h) & (c_prev > np.roll(o1h, 1, axis=0)) & (np.roll(c1h, 2, axis=0) > np.roll(o1h, 2, axis=0))
    three_dn = (c1h < o1h) & (c_prev < np.roll(o1h, 1, axis=0)) & (np.roll(c1h, 2, axis=0) < np.roll(o1h, 2, axis=0))
    sig_l = (m_ema50_4h > m_ema100_4h) & three_up
    sig_s = (m_ema50_4h < m_ema100_4h) & three_dn
    test_round(67, "MTF 3 Consecutive Trend Bars", sig_l, sig_s, hold=18)
    
    # R68: MTF Directional Consensus (1d, 4h, 1h all green)
    ret_1d = (m_c1d - np.roll(m_c1d, 24, axis=0)) / np.roll(m_c1d, 24, axis=0)
    ret_4h = (m_c4h - np.roll(m_c4h, 4, axis=0)) / np.roll(m_c4h, 4, axis=0)
    ret_1h = (c1h - c_prev) / c_prev
    sig_l = (ret_1d > 0.02) & (ret_4h > 0.01) & (ret_1h > 0.005)
    sig_s = (ret_1d < -0.02) & (ret_4h < -0.01) & (ret_1h < -0.005)
    test_round(68, "MTF Directional Momentum Thrust", sig_l, sig_s, hold=24)
    
    # R69: MTF Choppiness Gated Trend Follow (CHOP < 40 = strong trend)
    sig_l = (chop_1h < 40) & (c1h > ema20_1h) & (m_ema50_4h > m_ema100_4h)
    sig_s = (chop_1h < 40) & (c1h < ema20_1h) & (m_ema50_4h < m_ema100_4h)
    test_round(69, "MTF Choppiness Gated Trend", sig_l, sig_s, hold=24)
    
    # R70: MTF CMF Institutional Flow + Trend
    sig_l = (cmf_1h > 0.15) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = (cmf_1h < -0.15) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(70, "MTF CMF Flow + Trend Consensus", sig_l, sig_s, hold=24)
    
    
    # --- FAMILY 2: Mean Reversion & Multi-Timeframe Pullback Archetypes (R71 - R90) ---
    # R71: MTF RSI Oversold Dip in 1d Macro Uptrend
    sig_l = (m_c1d > m_sma200_1d) & (rsi14_1h < 25)
    test_round(71, "MTF RSI<25 Dip in 1d Bull", sig_l, empty_sig, hold=18)
    
    # R72: MTF RSI Deep Oversold Extreme in 4h Uptrend
    sig_l = (m_c4h > m_ema50_4h) & (rsi14_1h < 20)
    test_round(72, "MTF RSI<20 Deep Dip in 4h Bull", sig_l, empty_sig, hold=12)
    
    # R73: MTF Bollinger Band %B Lower Band Reversal
    # %B crosses above 0.0 after being below 0.0
    pb_prev = np.roll(pct_b_1h, 1, axis=0)
    pb_cross_up = (pb_prev < 0.0) & (pct_b_1h >= 0.0)
    sig_l = (m_c4h > m_ema50_4h) & pb_cross_up
    test_round(73, "MTF BB %B Lower Reclaim", sig_l, empty_sig, hold=12)
    
    # R74: MTF Bollinger Band 3-Sigma Stretch Fade
    sma_bb3, up_bb3, low_bb3, _, _ = ind.nb_bollinger(c1h, 20, 3.0)
    sig_l = c1h < low_bb3
    sig_s = c1h > up_bb3
    test_round(74, "MTF BB 3-Sigma Stretch Fade", sig_l, sig_s, hold=12)
    
    # R75: MTF Keltner Channel Lower Envelope Bounce
    _, _, kelt_l2 = ind.nb_keltner(h1h, l1h, c1h, 20, 2.0)
    sig_l = (m_c4h > m_ema50_4h) & (l1h < kelt_l2) & (c1h > kelt_l2)
    test_round(75, "MTF Keltner 2-ATR Dip Bounce", sig_l, empty_sig, hold=18)
    
    # R76: MTF Connors RSI Proxy (RSI3 + Streak RSI + PercentRank)
    rsi3_1h = ind.nb_rsi(c1h, 3)
    sig_l = (m_c1d > m_sma100_1d) & (rsi3_1h < 10)
    test_round(76, "MTF Connors RSI3 Dip in 1d Bull", sig_l, empty_sig, hold=12)
    
    # R77: MTF RSI Oversold + Volume Climax
    v_ma = ind.nb_sma(v1h, 24)
    sig_l = (rsi14_1h < 25) & (v1h > v_ma * 2.5)
    test_round(77, "MTF RSI<25 + Volume Climax", sig_l, empty_sig, hold=12)
    
    # R78: MTF Williams %R Extreme Reversal
    # Williams %R is basically (Highest High - Close) / (Highest High - Lowest Low) * -100
    hh24 = don_u_24
    ll24 = don_l_24
    w_pct_r = np.where(hh24 > ll24, (hh24 - c1h) / (hh24 - ll24) * -100.0, -50.0)
    w_prev = np.roll(w_pct_r, 1, axis=0)
    w_rebound = (w_prev < -90) & (w_pct_r > -85)
    sig_l = (m_c4h > m_ema50_4h) & w_rebound
    test_round(78, "MTF Williams %R Oversold Snap", sig_l, empty_sig, hold=18)
    
    # R79: MTF CCI Extreme Departure Reversal
    tp = (h1h + l1h + c1h) / 3.0
    tp_sma = ind.nb_sma(tp, 20)
    # mean dev
    md = ind.nb_sma(np.abs(tp - tp_sma), 20)
    cci = np.where(md > 0, (tp - tp_sma) / (0.015 * md), 0.0)
    cci_prev = np.roll(cci, 1, axis=0)
    sig_l = (m_c4h > m_ema50_4h) & (cci_prev < -150) & (cci > -100)
    test_round(79, "MTF CCI Oversold Snapback", sig_l, empty_sig, hold=12)
    
    # R80: MTF Distance from 4h VWAP / EMA Stretch Fade
    dist_4h = (c1h - m_ema50_4h) / m_ema50_4h
    sig_l = dist_4h < -0.06
    sig_s = dist_4h > 0.06
    test_round(80, "MTF Distance from 4h EMA Stretch", sig_l, sig_s, hold=18)
    
    # R81: MTF BB + RSI Double Exhaustion
    sig_l = (pct_b_1h < 0.0) & (rsi14_1h < 25)
    sig_s = (pct_b_1h > 1.0) & (rsi14_1h > 75)
    test_round(81, "MTF BB+RSI Double Exhaustion", sig_l, sig_s, hold=12)
    
    # R82: MTF MFI Oversold Reversal
    # Approximate MFI using typical price * volume
    sig_l = (m_c4h > m_ema50_4h) & (cmf_1h < -0.2) & (rsi14_1h < 25)
    test_round(82, "MTF MFI+RSI Deep Capitulation", sig_l, empty_sig, hold=18)
    
    # R83: MTF Donchian Lower-Band Rejection with Long Wick
    lower_wick = np.minimum(o1h, c1h) - l1h
    candle_body = np.abs(c1h - o1h)
    pin_bar = (lower_wick > candle_body * 2.0) & (l1h <= don_l_24)
    sig_l = (m_c4h > m_ema50_4h) & pin_bar
    test_round(83, "MTF Pin Bar at 24h Low", sig_l, empty_sig, hold=18)
    
    # R84: MTF 6 Consecutive Down Bars Capitulation Snapback
    six_down = True
    for k in range(6):
        six_down = six_down & (np.roll(c1h, k, axis=0) < np.roll(o1h, k, axis=0))
    sig_l = (m_c1d > m_sma100_1d) & six_down
    test_round(84, "MTF 6 Down-Bars in 1d Bull", sig_l, empty_sig, hold=12)
    
    # R85: MTF RSI Dip with Tight ATR Stop and 2x ATR TP
    sig_l = (m_c4h > m_ema50_4h) & (rsi14_1h < 25)
    test_round(85, "MTF RSI Dip + SL/TP (1:2)", sig_l, empty_sig, hold=24, sl=0.02, tp=0.04)
    
    # R86: MTF Overbought Shorting in 1d Macro Bear
    sig_s = (m_c1d < m_sma100_1d) & (rsi14_1h > 75)
    test_round(86, "MTF RSI>75 Short in 1d Bear", empty_sig, sig_s, hold=18)
    
    # R87: MTF BB Upper Rejection Short in 4h Downtrend
    sig_s = (m_c4h < m_ema50_4h) & (pct_b_1h > 1.0) & (c1h < o1h)
    test_round(87, "MTF BB Upper Reject in 4h Bear", empty_sig, sig_s, hold=18)
    
    # R88: MTF Choppiness-Gated Mean Reversion (CHOP > 60 = Chop regime, fade BB)
    sig_l = (chop_1h > 60) & (pct_b_1h < 0.05)
    sig_s = (chop_1h > 60) & (pct_b_1h > 0.95)
    test_round(88, "MTF Chop-Gated BB Fade", sig_l, sig_s, hold=12)
    
    # R89: MTF RSI Divergence (Price makes new low, RSI makes higher low)
    c_min48 = ind.nb_donchian(c1h, c1h, 48)[1]
    rsi_min48 = ind.nb_donchian(rsi14_1h, rsi14_1h, 48)[1]
    rsi_div = (c1h <= c_min48) & (rsi14_1h > rsi_min48 + 5.0) & (rsi14_1h < 35)
    sig_l = (m_c4h > m_ema50_4h) & rsi_div
    test_round(89, "MTF RSI Bullish Divergence", sig_l, empty_sig, hold=18)
    
    # R90: MTF CMF Oversold Snapback in 1d Macro Bull
    sig_l = (m_c1d > m_sma100_1d) & (cmf_1h < -0.25) & (c1h > c_prev)
    test_round(90, "MTF CMF<-0.25 Snap in 1d Bull", sig_l, empty_sig, hold=18)
    
    df_res = pd.DataFrame(results)
    seg = os.environ.get("RUN_SEG", "TRAIN")
    out_csv = "/root/freqtrade/user_data/minute_research/r3c/batch1_results.csv" if seg == "TRAIN" else f"/root/freqtrade/user_data/minute_research/r3c/batch1_valid_results.csv"
    df_res.to_csv(out_csv, index=False)
    print("\nBatch 1 completed! Saved to batch1_results.csv")


if __name__ == "__main__":
    run_batch1()
