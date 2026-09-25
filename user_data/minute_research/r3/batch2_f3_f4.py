"""Batch 2: Family 3 (R91-R110: Volume, Liquidity & Order Flow) & Family 4 (R111-R130: Volatility & Regime)."""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append("/root/freqtrade/user_data/minute_research/r3")
from engine200 import PanelData, evaluate_strategy
import indicators as ind


def run_batch2():
    print("Loading 1h, 4h, 1d panels for Batch 2...")
    t0 = time.time()
    p1h = PanelData("1h")
    p4h = PanelData("4h")
    p1d = PanelData("1d")
    print(f"Panels loaded in {time.time()-t0:.2f}s.")
    
    c1h = np.array(p1h.close)
    o1h = np.array(p1h.open)
    h1h = np.array(p1h.high)
    l1h = np.array(p1h.low)
    v1h = np.array(p1h.volume)
    qv1h = np.array(p1h.quote_volume) if p1h.quote_volume is not None else v1h * c1h
    tbv1h = np.array(p1h.taker_buy_volume) if p1h.taker_buy_volume is not None else v1h * 0.5
    cnt1h = np.array(p1h.count) if p1h.count is not None else np.ones_like(v1h)
    tt_ls1h = np.array(p1h.toptrader_ls) if p1h.toptrader_ls is not None else np.ones_like(v1h)
    
    c4h = np.array(p4h.close)
    h4h = np.array(p4h.high)
    l4h = np.array(p4h.low)
    
    print("Pre-computing multi-timeframe regime and indicators...")
    ema20_1h = ind.nb_ema(c1h, 20)
    ema50_1h = ind.nb_ema(c1h, 50)
    atr14_1h = ind.nb_atr(h1h, l1h, c1h, 14)
    atr100_1h = ind.nb_atr(h1h, l1h, c1h, 100)
    rsi14_1h = ind.nb_rsi(c1h, 14)
    don_u_24, don_l_24 = ind.nb_donchian(h1h, l1h, 24)
    sma_bb, up_bb, low_bb, bbw_1h, pct_b_1h = ind.nb_bollinger(c1h, 20, 2.0)
    kelt_mid, kelt_u, kelt_l = ind.nb_keltner(h1h, l1h, c1h, 20, 1.5)
    chop_1h = ind.nb_choppiness(h1h, l1h, c1h, 14)
    adx_1h, pdi_1h, mdi_1h = ind.nb_adx(h1h, l1h, c1h, 14)
    
    # 4h indicators mapped causally
    ema50_4h = ind.nb_ema(c4h, 50)
    ema100_4h = ind.nb_ema(c4h, 100)
    m_ema50_4h = p1h.map_htf_to_ltf(ema50_4h, "4h")
    m_ema100_4h = p1h.map_htf_to_ltf(ema100_4h, "4h")
    m_c4h = p1h.map_htf_to_ltf(c4h, "4h")
    
    st_trend_4h, _ = ind.nb_supertrend(h4h, l4h, c4h, 10, 3.0)
    m_st_trend_4h = p1h.map_htf_to_ltf(st_trend_4h.astype(np.float64), "4h")
    
    # Volume metrics
    v_ma24 = ind.nb_sma(v1h, 24)
    qv_ma24 = ind.nb_sma(qv1h, 24)
    cnt_ma24 = ind.nb_sma(cnt1h, 24)
    
    # Taker ratio
    taker_ratio = np.where(v1h > 0, tbv1h / v1h, 0.5)
    taker_ratio_ma4 = ind.nb_sma(taker_ratio, 4)
    
    # Avg trade size proxy
    avg_trade_size = np.where(cnt1h > 0, v1h / cnt1h, 0.0)
    avg_trade_size_ma24 = ind.nb_sma(avg_trade_size, 24)
    
    # Top trader LS metrics
    tt_ls_ma24 = ind.nb_sma(tt_ls1h, 24)
    tt_ls_delta24 = tt_ls1h - np.roll(tt_ls1h, 24, axis=0)
    
    results = []
    T, N = c1h.shape
    empty_sig = np.zeros((T, N), dtype=bool)
    c_prev = np.roll(c1h, 1, axis=0)

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

    print("Executing Batch 2 (R91-R130) on TRAIN...")
    
    # --- FAMILY 3: Volume, Liquidity & Order Flow Dynamics (R91 - R110) ---
    # R91: Top-Trader LS Divergence Accumulation (Top trader LS > 1.3 while price down > 3% over 24h)
    ret24 = (c1h - np.roll(c1h, 24, axis=0)) / np.roll(c1h, 24, axis=0)
    sig_l = (tt_ls1h > 1.3) & (ret24 < -0.03) & (m_ema50_4h > m_ema100_4h)
    test_round(91, "Top-Trader LS Div Accumulation", sig_l, empty_sig, hold=24)
    
    # R92: Top-Trader vs Crowd Sentiment Spread
    # When top trader LS is well above its 24h average and taker buying is positive
    sig_l = (tt_ls1h > tt_ls_ma24 * 1.15) & (taker_ratio > 0.55) & (m_c4h > m_ema50_4h)
    sig_s = (tt_ls1h < tt_ls_ma24 * 0.85) & (taker_ratio < 0.45) & (m_c4h < m_ema50_4h)
    test_round(92, "Top-Trader LS Surge + Taker Flow", sig_l, sig_s, hold=24)
    
    # R93: Taker Buy Volume Share Surge (> 65% taker buy over 4h)
    sig_l = (taker_ratio_ma4 > 0.65) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = (taker_ratio_ma4 < 0.35) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(93, "Taker Buy 4h Surge >65%", sig_l, sig_s, hold=18)
    
    # R94: Taker Buy Climax Reversal Fade (Taker buy > 75% but candle is red or has upper wick)
    upper_wick = h1h - np.maximum(o1h, c1h)
    candle_body = np.abs(c1h - o1h)
    sig_s = (taker_ratio > 0.75) & (upper_wick > candle_body * 1.5) & (v1h > v_ma24 * 2.0)
    test_round(94, "Taker Climax Exhaustion Fade", empty_sig, sig_s, hold=12)
    
    # R95: Quote Volume Breakout with 4h Trend (QV > 3x 24h avg + price breakout)
    sig_l = (qv1h > qv_ma24 * 3.0) & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = (qv1h > qv_ma24 * 3.0) & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(95, "Quote Vol Breakout + 4h Trend", sig_l, sig_s, hold=24)
    
    # R96: Retail Frenzy Fade (Count > 3x avg but trade size < 0.5x avg)
    sig_s = (cnt1h > cnt_ma24 * 3.0) & (avg_trade_size < avg_trade_size_ma24 * 0.5) & (ret24 > 0.05)
    test_round(96, "Retail Count Frenzy Fade", empty_sig, sig_s, hold=18)
    
    # R97: Institutional Block Flow Follow (Trade size > 2.5x avg in direction of move)
    sig_l = (avg_trade_size > avg_trade_size_ma24 * 2.5) & (c1h > o1h) & (m_c4h > m_ema50_4h)
    sig_s = (avg_trade_size > avg_trade_size_ma24 * 2.5) & (c1h < o1h) & (m_c4h < m_ema50_4h)
    test_round(97, "Inst Block Flow Follow", sig_l, sig_s, hold=24)
    
    # R98: Taker Flow Imbalance Persistence (3 consecutive bars with taker buy > 60%)
    tb3 = (taker_ratio > 0.60) & (np.roll(taker_ratio, 1, axis=0) > 0.60) & (np.roll(taker_ratio, 2, axis=0) > 0.60)
    ts3 = (taker_ratio < 0.40) & (np.roll(taker_ratio, 1, axis=0) < 0.40) & (np.roll(taker_ratio, 2, axis=0) < 0.40)
    sig_l = tb3 & (m_c4h > m_ema50_4h)
    sig_s = ts3 & (m_c4h < m_ema50_4h)
    test_round(98, "Taker Flow 3-Bar Persistence", sig_l, sig_s, hold=18)
    
    # R99: Relative Volume (RVOL) 5x Explosion
    sig_l = (v1h > v_ma24 * 5.0) & (c1h > o1h) & (c1h > ema20_1h)
    sig_s = (v1h > v_ma24 * 5.0) & (c1h < o1h) & (c1h < ema20_1h)
    test_round(99, "RVOL 5x Explosion Breakout", sig_l, sig_s, hold=18)
    
    # R100: Top-Trader LS Momentum (24h change in top-trader LS > 0.3)
    sig_l = (tt_ls_delta24 > 0.3) & (c1h > ema20_1h)
    sig_s = (tt_ls_delta24 < -0.3) & (c1h < ema20_1h)
    test_round(100, "Top-Trader LS 24h Delta Mom", sig_l, sig_s, hold=24)
    
    # R101: Top-Trader Extreme Short Squeeze (Top trader LS < 0.75 + price breaks 24h high)
    sig_l = (tt_ls1h < 0.75) & (c1h > don_u_24) & (taker_ratio > 0.55)
    test_round(101, "Top-Trader Short Squeeze Hunter", sig_l, empty_sig, hold=24)
    
    # R102: Volume Climax + Pin Bar Absorption (Volume > 3x MA + long lower wick)
    lower_wick = np.minimum(o1h, c1h) - l1h
    candle_body = np.maximum(np.abs(c1h - o1h), 1e-8)
    sig_l = (v1h > v_ma24 * 3.0) & (lower_wick > candle_body * 2.0) & (rsi14_1h < 30)
    test_round(102, "Volume Climax Absorption Pin", sig_l, empty_sig, hold=18)
    
    # R103: Low-Volume Drift Fade (Price breaks 24h high on low volume < 0.6x avg)
    sig_s = (c1h > don_u_24) & (v1h < v_ma24 * 0.6) & (m_c4h < m_ema50_4h)
    test_round(103, "Low-Vol Breakout Fade Short", empty_sig, sig_s, hold=18)
    
    # R104: Volume-Weighted Price Action Reversal
    # Heavy selling volume followed by sharp reclaim
    heavy_sell = (np.roll(v1h, 1, axis=0) > v_ma24 * 2.5) & (c_prev < np.roll(o1h, 1, axis=0))
    reclaim = (c1h > np.roll(o1h, 1, axis=0)) & (taker_ratio > 0.55)
    sig_l = heavy_sell & reclaim & (m_c4h > m_ema50_4h)
    test_round(104, "Heavy Sell Volume Trap Reclaim", sig_l, empty_sig, hold=24)
    
    # R105: Taker Flow Reversal in Oversold (RSI < 25 + taker buy ratio > 60%)
    sig_l = (rsi14_1h < 25) & (taker_ratio > 0.60) & (c1h > o1h)
    test_round(105, "Taker Flow Reversal in Oversold", sig_l, empty_sig, hold=18)
    
    # R106: Taker Buy Volume Surge with ATR expansion
    sig_l = (taker_ratio > 0.65) & (atr14_1h > ind.nb_sma(atr14_1h, 24) * 1.3) & (m_c4h > m_ema50_4h)
    sig_s = (taker_ratio < 0.35) & (atr14_1h > ind.nb_sma(atr14_1h, 24) * 1.3) & (m_c4h < m_ema50_4h)
    test_round(106, "Taker Surge + ATR Expansion", sig_l, sig_s, hold=18)
    
    # R107: Top-Trader Long Accumulation in 4h SuperTrend Bull
    sig_l = (m_st_trend_4h == 1) & (tt_ls1h > 1.25) & (c1h > ema20_1h) & (c_prev <= ema20_1h)
    test_round(107, "Top-Trader + 4h SuperTrend Bull", sig_l, empty_sig, hold=24)
    
    # R108: Top-Trader Short Accumulation in 4h SuperTrend Bear
    sig_s = (m_st_trend_4h == -1) & (tt_ls1h < 0.80) & (c1h < ema20_1h) & (c_prev >= ema20_1h)
    test_round(108, "Top-Trader + 4h SuperTrend Bear", empty_sig, sig_s, hold=24)
    
    # R109: Volume Surge with Moving Average Breakout
    sig_l = (v1h > v_ma24 * 2.5) & (c_prev < ema50_1h) & (c1h > ema50_1h) & (m_c4h > m_ema50_4h)
    sig_s = (v1h > v_ma24 * 2.5) & (c_prev > ema50_1h) & (c1h < ema50_1h) & (m_c4h < m_ema50_4h)
    test_round(109, "Volume Surge 50-EMA Breakout", sig_l, sig_s, hold=24)
    
    # R110: Volume Weighted Momentum Thrust (Return * RVOL)
    vol_thrust = (c1h / c_prev - 1.0) * (v1h / np.maximum(v_ma24, 1e-8))
    sig_l = (vol_thrust > 0.08) & (m_c4h > m_ema50_4h)
    sig_s = (vol_thrust < -0.08) & (m_c4h < m_ema50_4h)
    test_round(110, "Volume Weighted Momentum Thrust", sig_l, sig_s, hold=18)

    
    # --- FAMILY 4: Structural Market Regime & Volatility Compression (R111 - R130) ---
    # R111: Bollinger Band Width (BBW) Compression into Expansion Breakout
    bbw_min168 = ind.nb_donchian(bbw_1h, bbw_1h, 168)[1]
    bbw_compressed = bbw_1h <= bbw_min168 * 1.15
    sig_l = np.roll(bbw_compressed, 1, axis=0) & (c1h > up_bb) & (m_c4h > m_ema50_4h)
    sig_s = np.roll(bbw_compressed, 1, axis=0) & (c1h < low_bb) & (m_c4h < m_ema50_4h)
    test_round(111, "BBW 7d Compression Breakout", sig_l, sig_s, hold=24)
    
    # R112: ATR Ratio (ATR14 / ATR100) Compression Breakout
    atr_ratio = np.where(atr100_1h > 0, atr14_1h / atr100_1h, 1.0)
    sig_l = (np.roll(atr_ratio, 1, axis=0) < 0.7) & (atr_ratio > 0.85) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = (np.roll(atr_ratio, 1, axis=0) < 0.7) & (atr_ratio > 0.85) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(112, "ATR Ratio Compression Break", sig_l, sig_s, hold=24)
    
    # R113: Historical Volatility (HV24) Pinch Breakout
    # HV is std of log returns
    log_ret = np.log(np.maximum(c1h, 1e-8) / np.maximum(c_prev, 1e-8))
    hv24 = ind.nb_rolling_std(log_ret, 24)
    hv_min = ind.nb_donchian(hv24, hv24, 72)[1]
    sig_l = (np.roll(hv24, 1, axis=0) <= hv_min * 1.1) & (hv24 > hv_min * 1.3) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = (np.roll(hv24, 1, axis=0) <= hv_min * 1.1) & (hv24 > hv_min * 1.3) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(113, "HV24 Pinch into Expansion", sig_l, sig_s, hold=24)
    
    # R114: Choppiness Index (CHOP < 38) Trend Expansion Follow
    sig_l = (chop_1h < 38) & (c_prev < ema20_1h) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = (chop_1h < 38) & (c_prev > ema20_1h) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(114, "CHOP<38 Trend Momentum Cross", sig_l, sig_s, hold=24)
    
    # R115: Choppiness Index (CHOP > 62) Range Boundary Fade
    sig_l = (chop_1h > 62) & (c1h < low_bb) & (c1h > c_prev)
    sig_s = (chop_1h > 62) & (c1h > up_bb) & (c1h < c_prev)
    test_round(115, "CHOP>62 Range Boundary Fade", sig_l, sig_s, hold=12)
    
    # R116: ADX Compression (<15) Followed by Hook Up Above 20
    adx_prev = np.roll(adx_1h, 1, axis=0)
    adx_hook = (adx_prev < 18) & (adx_1h >= 20)
    sig_l = adx_hook & (pdi_1h > mdi_1h) & (m_c4h > m_ema50_4h)
    sig_s = adx_hook & (pdi_1h < mdi_1h) & (m_c4h < m_ema50_4h)
    test_round(116, "ADX Compression Hook Breakout", sig_l, sig_s, hold=24)
    
    # R117: Volatility Squeeze (1h BB inside Keltner) with 4h SuperTrend
    bb_inside_keltner = (up_bb < kelt_u) & (low_bb > kelt_l)
    squeeze_release = np.roll(bb_inside_keltner, 1, axis=0) & (~bb_inside_keltner)
    sig_l = squeeze_release & (c1h > kelt_u) & (m_st_trend_4h == 1)
    sig_s = squeeze_release & (c1h < kelt_l) & (m_st_trend_4h == -1)
    test_round(117, "TTM Squeeze Release + 4h ST", sig_l, sig_s, hold=18)
    
    # R118: Normalized ATR Extreme Spike Capitulation Snapback
    natr = np.where(c1h > 0, atr14_1h / c1h, 0.0)
    natr_ma = ind.nb_sma(natr, 72)
    sig_l = (natr > natr_ma * 2.5) & (rsi14_1h < 25) & (c1h > o1h)
    test_round(118, "NATR 2.5x Spike Capitulation Snap", sig_l, empty_sig, hold=12)
    
    # R119: 4h Volatility Compression + 1h Donchian Breakout
    # 4h BBW
    _, _, _, bbw_4h, _ = ind.nb_bollinger(c4h, 20, 2.0)
    m_bbw_4h = p1h.map_htf_to_ltf(bbw_4h, "4h")
    m_bbw_4h_min = p1h.map_htf_to_ltf(ind.nb_donchian(bbw_4h, bbw_4h, 42)[1], "4h")
    bbw_4h_quiet = m_bbw_4h <= m_bbw_4h_min * 1.2
    sig_l = bbw_4h_quiet & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = bbw_4h_quiet & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(119, "4h Vol Compression + 1h Donchian", sig_l, sig_s, hold=36)
    
    # R120: High-Volatility Regime Breakout with Trailing SL/TP
    sig_l = (hv24 > ind.nb_sma(hv24, 72) * 1.2) & (c1h > kelt_u) & (m_c4h > m_ema50_4h)
    sig_s = (hv24 > ind.nb_sma(hv24, 72) * 1.2) & (c1h < kelt_l) & (m_c4h < m_ema50_4h)
    test_round(120, "High-Vol Breakout + SL/TP (2:4)", sig_l, sig_s, hold=24, sl=0.025, tp=0.05)
    
    # R121: Low-Volatility Regime Drift Follow (Slow trend continuation)
    sig_l = (hv24 < ind.nb_sma(hv24, 72) * 0.8) & (c1h > ema20_1h) & (ema20_1h > ema50_1h) & (m_c4h > m_ema50_4h)
    sig_s = (hv24 < ind.nb_sma(hv24, 72) * 0.8) & (c1h < ema20_1h) & (ema20_1h < ema50_1h) & (m_c4h < m_ema50_4h)
    test_round(121, "Low-Vol Drift Trend Follow", sig_l, sig_s, hold=36)
    
    # R122: Dynamic Volatility Breakout with 2-ATR Trailing TP
    sig_l = (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(122, "Donchian Break + Dynamic TP/SL", sig_l, sig_s, hold=36, sl=0.03, tp=0.06)
    
    # R123: Volatility Ratio (High-Low / Open-Close) Expansion
    candle_range = h1h - l1h
    candle_body_raw = np.abs(c1h - o1h)
    body_ratio = np.where(candle_range > 0, candle_body_raw / candle_range, 0.0)
    # Marubozu-style solid breakout bar
    solid_bar_up = (body_ratio > 0.80) & (c1h > o1h) & (candle_range > atr14_1h * 1.5)
    solid_bar_dn = (body_ratio > 0.80) & (c1h < o1h) & (candle_range > atr14_1h * 1.5)
    sig_l = solid_bar_up & (m_c4h > m_ema50_4h)
    sig_s = solid_bar_dn & (m_c4h < m_ema50_4h)
    test_round(123, "Solid Marubozu Vol Expansion", sig_l, sig_s, hold=18)
    
    # R124: Garman-Klass Volatility Proxy Spike Fade
    # Extreme bar range > 3x ATR followed by reversal inside range
    extreme_bar_dn = (np.roll(candle_range, 1, axis=0) > np.roll(atr14_1h, 1, axis=0) * 3.0) & (c_prev < np.roll(o1h, 1, axis=0))
    rebound = (c1h > o1h) & (l1h > np.roll(l1h, 1, axis=0))
    sig_l = extreme_bar_dn & rebound & (m_c4h > m_ema50_4h)
    test_round(124, "GK Vol Spike Exhaustion Rebound", sig_l, empty_sig, hold=12)
    
    # R125: Intraday Range Expansion Breakout (NR7 proxy: lowest range in 7 bars followed by expansion)
    range_min7 = ind.nb_donchian(candle_range, candle_range, 7)[1]
    nr7_prev = np.roll(candle_range, 1, axis=0) <= np.roll(range_min7, 1, axis=0) * 1.05
    sig_l = nr7_prev & (candle_range > atr14_1h * 1.2) & (c1h > o1h) & (m_c4h > m_ema50_4h)
    sig_s = nr7_prev & (candle_range > atr14_1h * 1.2) & (c1h < o1h) & (m_c4h < m_ema50_4h)
    test_round(125, "NR7 Range Expansion Breakout", sig_l, sig_s, hold=18)
    
    # R126: Volatility Pinch with Pre-Volume Build
    vol_building = v1h > v_ma24 * 1.5
    sig_l = bbw_compressed & vol_building & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = bbw_compressed & vol_building & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(126, "Vol Pinch + Pre-Vol Building", sig_l, sig_s, hold=24)
    
    # R127: MTF Double Volatility Squeeze (1h and 4h both in squeeze)
    # 4h BB inside Keltner
    kelt_4h_mid, kelt_4h_u, kelt_4h_l = ind.nb_keltner(h4h, l4h, c4h, 20, 1.5)
    bb_4h_sma, bb_4h_u, bb_4h_l, _, _ = ind.nb_bollinger(c4h, 20, 2.0)
    sq_4h = (bb_4h_u < kelt_4h_u) & (bb_4h_l > kelt_4h_l)
    m_sq_4h = p1h.map_htf_to_ltf(sq_4h.astype(np.float64), "4h")
    dual_sq = (m_sq_4h == 1.0) & bb_inside_keltner
    sig_l = np.roll(dual_sq, 1, axis=0) & (c1h > kelt_u)
    sig_s = np.roll(dual_sq, 1, axis=0) & (c1h < kelt_l)
    test_round(127, "Dual 4h/1h MTF Squeeze Break", sig_l, sig_s, hold=24)
    
    # R128: ATR Expansion with Volume Confirmation
    atr_surge = (atr14_1h > ind.nb_sma(atr14_1h, 24) * 1.5)
    vol_surge = (v1h > v_ma24 * 2.0)
    sig_l = atr_surge & vol_surge & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = atr_surge & vol_surge & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(128, "ATR + Volume Surge Momentum", sig_l, sig_s, hold=24)
    
    # R129: Volatility-Adjusted Momentum Breakout (Return / ATR > 2.0)
    ret_in_atr = np.where(atr14_1h > 0, (c1h - c_prev) / atr14_1h, 0.0)
    sig_l = (ret_in_atr > 1.8) & (m_c4h > m_ema50_4h)
    sig_s = (ret_in_atr < -1.8) & (m_c4h < m_ema50_4h)
    test_round(129, "Vol-Adjusted Return Spike Follow", sig_l, sig_s, hold=18)
    
    # R130: Regime-Switching Volatility Filtered MA Cross
    # Only take EMA cross when CHOP < 45 and ADX > 22
    trending_regime = (chop_1h < 45) & (adx_1h > 22)
    ema_cross_up = (c_prev < ema20_1h) & (c1h > ema20_1h)
    ema_cross_dn = (c_prev > ema20_1h) & (c1h < ema20_1h)
    sig_l = trending_regime & ema_cross_up & (m_c4h > m_ema50_4h)
    sig_s = trending_regime & ema_cross_dn & (m_c4h < m_ema50_4h)
    test_round(130, "Regime-Filtered 20-EMA Cross", sig_l, sig_s, hold=24)
    
    df_res = pd.DataFrame(results)
    seg = os.environ.get("RUN_SEG", "TRAIN")
    out_csv = "/root/freqtrade/user_data/minute_research/r3/batch2_results.csv" if seg == "TRAIN" else f"/root/freqtrade/user_data/minute_research/r3/batch2_valid_results.csv"
    df_res.to_csv(out_csv, index=False)
    print("\nBatch 2 completed! Saved to batch2_results.csv")


if __name__ == "__main__":
    run_batch2()
