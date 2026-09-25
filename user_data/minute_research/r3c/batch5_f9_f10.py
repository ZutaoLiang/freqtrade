"""Batch 5: Family 9 (R211-R230: Macro BTC Transmission) & Family 10 (R231-R250: Multi-Factor Ensembles)."""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append("/root/freqtrade/user_data/minute_research/r3c")
from engine200 import PanelData, evaluate_strategy
import indicators as ind


def run_batch5():
    print("Loading 1h, 4h, 1d panels for Batch 5...")
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
    fr1h = np.array(p1h.funding_rate) if p1h.funding_rate is not None else np.zeros_like(c1h)
    oi1h = np.array(p1h.oi) if p1h.oi is not None else np.zeros_like(c1h)
    tt_ls1h = np.array(p1h.toptrader_ls) if p1h.toptrader_ls is not None else np.ones_like(c1h)
    
    c4h = np.array(p4h.close)
    h4h = np.array(p4h.high)
    l4h = np.array(p4h.low)
    
    c1d = np.array(p1d.close)
    
    print("Pre-computing multi-timeframe indicators for Batch 5...")
    ema20_1h = ind.nb_ema(c1h, 20)
    ema50_1h = ind.nb_ema(c1h, 50)
    atr14_1h = ind.nb_atr(h1h, l1h, c1h, 14)
    rsi14_1h = ind.nb_rsi(c1h, 14)
    don_u_24, don_l_24 = ind.nb_donchian(h1h, l1h, 24)
    cmf_1h = ind.nb_cmf(h1h, l1h, c1h, v1h, 20)
    chop_1h = ind.nb_choppiness(h1h, l1h, c1h, 14)
    adx_1h, _, _ = ind.nb_adx(h1h, l1h, c1h, 14)
    kelt_mid, kelt_u, kelt_l = ind.nb_keltner(h1h, l1h, c1h, 20, 1.5)
    
    # 4h indicators mapped causally
    ema50_4h = ind.nb_ema(c4h, 50)
    ema100_4h = ind.nb_ema(c4h, 100)
    m_ema50_4h = p1h.map_htf_to_ltf(ema50_4h, "4h")
    m_ema100_4h = p1h.map_htf_to_ltf(ema100_4h, "4h")
    m_c4h = p1h.map_htf_to_ltf(c4h, "4h")
    
    st_trend_4h, _ = ind.nb_supertrend(h4h, l4h, c4h, 10, 3.0)
    m_st_trend_4h = p1h.map_htf_to_ltf(st_trend_4h.astype(np.float64), "4h")
    
    # 1d indicators mapped causally
    sma100_1d = ind.nb_sma(c1d, 100)
    m_sma100_1d = p1h.map_htf_to_ltf(sma100_1d, "1d")
    m_c1d = p1h.map_htf_to_ltf(c1d, "1d")
    
    # BTC indicators (BTC is index 0 in symbols or find BTCUSDT)
    btc_idx = p1h.symbols.index("BTCUSDT") if "BTCUSDT" in p1h.symbols else p1h.u162_indices[0]
    btc_c1h = c1h[:, btc_idx:btc_idx+1]
    btc_m_c4h = m_c4h[:, btc_idx:btc_idx+1]
    btc_m_ema50_4h = m_ema50_4h[:, btc_idx:btc_idx+1]
    btc_ret24 = (btc_c1h - np.roll(btc_c1h, 24, axis=0)) / np.roll(btc_c1h, 24, axis=0)
    btc_rsi14 = rsi14_1h[:, btc_idx:btc_idx+1]
    btc_bull_4h = btc_m_c4h > btc_m_ema50_4h
    btc_fr = fr1h[:, btc_idx:btc_idx+1]
    
    # Market Breadth (% of U162 alts with close > 4h 50 EMA)
    u_c4h = m_c4h[:, p1h.u162_indices]
    u_ema50_4h = m_ema50_4h[:, p1h.u162_indices]
    breadth_above_50 = np.mean(u_c4h > u_ema50_4h, axis=1, keepdims=True) # shape (T, 1)
    
    # Taker ratio
    taker_ratio = np.where(v1h > 0, tbv1h / v1h, 0.5)
    
    # OI change
    oi_prev24 = np.roll(oi1h, 24, axis=0)
    oi_pct_chg24 = np.where(oi_prev24 > 0, (oi1h - oi_prev24) / oi_prev24, 0.0)
    
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
            "round": r_id, "name": name, "n": res.get("n", 0), "mean_bp": res.get("mean_bp", 0.0),
            "med_bp": res.get("med_bp", 0.0), "win": res.get("win", 0.0), "pf": res.get("pf", 0.0),
            "day_t": res.get("day_t", 0.0), "per_day": res.get("per_day", 0.0), "status": status
        }
        results.append(row)
        print(f"[{r_id:03d}] {name[:35]:35s} | n={row['n']:5d} | mean={row['mean_bp']:6.1f}bp | pf={row['pf']:5.2f} | day_t={row['day_t']:5.2f} |/d={row['per_day']:4.1f} | {status}")
        return row

    print("Executing Batch 5 (R211-R250) on TRAIN...")
    
    # --- FAMILY 9: Macro BTC Lead-Lag & Regime Transmission (R211 - R230) ---
    # R211: BTC 4h Trend Gated Alt Breakout
    sig_l = btc_bull_4h & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = (~btc_bull_4h) & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(211, "BTC 4h Bull Gated Alt Breakout", sig_l, sig_s, hold=24)
    
    # R212: BTC Volatility Compression Alt Breakout (BTC ATR / Price < 2%)
    btc_atr = atr14_1h[:, btc_idx:btc_idx+1]
    btc_natr = np.where(btc_c1h > 0, btc_atr / btc_c1h, 0.0)
    btc_calm = btc_natr < 0.015
    sig_l = btc_calm & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = btc_calm & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(212, "BTC Low-Vol Calm Alt Breakout", sig_l, sig_s, hold=24)
    
    # R213: BTC Momentum High-Beta Follow (When BTC 24h return > 3%, buy top momentum alts)
    ret24 = (c1h - np.roll(c1h, 24, axis=0)) / np.roll(c1h, 24, axis=0)
    sig_l = (btc_ret24 > 0.03) & (ret24 > 0.05) & (c1h > ema20_1h)
    test_round(213, "BTC 24h Surge Alt Momentum Follow", sig_l, empty_sig, hold=24)
    
    # R214: BTC Overextension Alt Fade (BTC 24h return > 6% fade overbought alts)
    sig_s = (btc_ret24 > 0.06) & (rsi14_1h > 75)
    test_round(214, "BTC 6% Overextension Alt Fade", empty_sig, sig_s, hold=18)
    
    # R215: ETH/BTC Ratio Trend Gating
    eth_idx = p1h.symbols.index("ETHUSDT") if "ETHUSDT" in p1h.symbols else p1h.u162_indices[1]
    eth_c1h = c1h[:, eth_idx:eth_idx+1]
    eth_btc = np.where(btc_c1h > 0, eth_c1h / btc_c1h, 0.0)
    eth_btc_ma = ind.nb_sma(eth_btc, 72)
    altseason = eth_btc > eth_btc_ma
    sig_l = altseason & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    test_round(215, "ETH/BTC Altseason Breakout Gating", sig_l, empty_sig, hold=24)
    
    # R216: BTC Climax Capitulation Spillover Bounce (BTC RSI < 25 -> Buy liquid alts)
    sig_l = (btc_rsi14 < 25) & (rsi14_1h < 25) & (c1h > o1h)
    test_round(216, "BTC Capitulation Spillover Snap", sig_l, empty_sig, hold=12)
    
    # R217: Alt Relative Strength during BTC Flat/Chop
    btc_chop = np.abs(btc_ret24) < 0.01
    alt_breakout = (c1h > don_u_24) & (ret24 > 0.04)
    sig_l = btc_chop & alt_breakout & (m_c4h > m_ema50_4h)
    test_round(217, "Alt Relative Strength in BTC Chop", sig_l, empty_sig, hold=24)
    
    # R218: Alt Relative Weakness during BTC Drops (BTC down > 2%, short alts breaking 48h lows)
    don_l_48 = ind.nb_donchian(h1h, l1h, 48)[1]
    sig_s = (btc_ret24 < -0.02) & (c1h < don_l_48) & (m_c4h < m_ema50_4h)
    test_round(218, "Alt Relative Weakness Short", empty_sig, sig_s, hold=24)
    
    # R219: BTC Funding Rate Regime Transmission (When BTC FR < 0, long alts)
    sig_l = (btc_fr < 0.0) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = (btc_fr > 0.0004) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(219, "BTC FR Regime Transmission", sig_l, sig_s, hold=24)
    
    # R220: Alt Market Breadth Gating (>65% alts in 4h uptrend)
    sig_l = (breadth_above_50 > 0.65) & (c_prev < ema20_1h) & (c1h > ema20_1h)
    sig_s = (breadth_above_50 < 0.35) & (c_prev > ema20_1h) & (c1h < ema20_1h)
    test_round(220, "Market Breadth Gated EMA Cross", sig_l, sig_s, hold=24)
    
    # R221: Market Breadth Thrust (Breadth crosses from < 30% to > 50% in 48h)
    breadth_prev48 = np.roll(breadth_above_50, 48, axis=0)
    breadth_thrust = (breadth_prev48 < 0.30) & (breadth_above_50 > 0.50)
    sig_l = breadth_thrust & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    test_round(221, "Alt Market Breadth Thrust Follow", sig_l, empty_sig, hold=36)
    
    # R222: BTC Taker Flow Surge Transmission
    btc_tbv = tbv1h[:, btc_idx:btc_idx+1]
    btc_v = v1h[:, btc_idx:btc_idx+1]
    btc_taker_ratio = np.where(btc_v > 0, btc_tbv / btc_v, 0.5)
    sig_l = (btc_taker_ratio > 0.62) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = (btc_taker_ratio < 0.38) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(222, "BTC Taker Flow Spillover Follow", sig_l, sig_s, hold=18)
    
    # R223: BTC 4h SuperTrend Regime Filter on Alt Momentum
    btc_st = m_st_trend_4h[:, btc_idx:btc_idx+1]
    sig_l = (btc_st == 1) & (c1h > don_u_24) & (m_st_trend_4h == 1)
    sig_s = (btc_st == -1) & (c1h < don_l_24) & (m_st_trend_4h == -1)
    test_round(223, "BTC+Alt Dual SuperTrend Break", sig_l, sig_s, hold=24)
    
    # R224: BTC 200 EMA Macro Regime Filter
    btc_sma200_1d = m_sma100_1d[:, btc_idx:btc_idx+1]
    sig_l = (btc_c1h > btc_sma200_1d) & (c1h > ema20_1h) & (ema20_1h > ema50_1h)
    sig_s = (btc_c1h < btc_sma200_1d) & (c1h < ema20_1h) & (ema20_1h < ema50_1h)
    test_round(224, "BTC Macro 100d Filter MA Cross", sig_l, sig_s, hold=24)
    
    # R225: Alt Beta-Adjusted Residual Breakout (Alt return - BTC return > 4%)
    residual = ret24 - btc_ret24
    sig_l = (residual > 0.04) & (m_c4h > m_ema50_4h)
    test_round(225, "Beta-Residual Breakout Long", sig_l, empty_sig, hold=24)
    
    # R226: BTC Drawdown Rebound Leader (BTC down > 4% then first green bar)
    btc_dump = np.roll(btc_ret24, 4, axis=0) < -0.04
    btc_rebound = (btc_c1h > np.roll(btc_c1h, 1, axis=0))
    sig_l = btc_dump & btc_rebound & (c1h > o1h) & (rsi14_1h < 35)
    test_round(226, "BTC Dump Rebound Leader", sig_l, empty_sig, hold=18)
    
    # R227: Alt Dispersion Expansion Strategy (Std of alt returns > 90th percentile)
    u_ret24 = ret24[:, p1h.u162_indices]
    alt_disp = np.nanstd(u_ret24, axis=1, keepdims=True)
    alt_disp_ma = ind.nb_sma(alt_disp, 168)
    high_disp = alt_disp > alt_disp_ma * 1.5
    sig_l = high_disp & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = high_disp & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(227, "High Alt Dispersion Breakout", sig_l, sig_s, hold=24)
    
    # R228: BTC Weekend Liquidity Momentum
    hours = pd.to_datetime(p1h.dates).dayofweek.values
    is_weekend = (hours >= 5) # Sat & Sun
    is_wk_2d = np.repeat(is_weekend[:, None], N, axis=1)
    sig_l = is_wk_2d & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = is_wk_2d & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(228, "Weekend Alt Momentum Breakout", sig_l, sig_s, hold=24)
    
    # R229: BTC Open Interest Expansion Gate
    btc_oi = oi1h[:, btc_idx:btc_idx+1]
    btc_oi_pct = np.where(np.roll(btc_oi, 24, axis=0) > 0, (btc_oi - np.roll(btc_oi, 24, axis=0)) / np.roll(btc_oi, 24, axis=0), 0.0)
    sig_l = (btc_oi_pct > 0.08) & (btc_ret24 > 0.02) & (c1h > don_u_24)
    test_round(229, "BTC OI Inflow Alt Breakout", sig_l, empty_sig, hold=24)
    
    # R230: BTC Macro + Alt Breakout with Trailing SL/TP
    test_round(230, "BTC Bull + Alt Break + SL/TP", sig_l, empty_sig, hold=36, sl=0.03, tp=0.06)

    
    # --- FAMILY 10: Multi-Factor Ensembles & Finalist Tuning (R231 - R250) ---
    # R231: Hybrid 1: 4h Trend + 1h Taker Surge + ATR Expansion (R106 Refinement)
    atr_exp = atr14_1h > ind.nb_sma(atr14_1h, 24) * 1.3
    sig_l = (taker_ratio > 0.65) & atr_exp & (m_c4h > m_ema50_4h) & (btc_bull_4h)
    sig_s = (taker_ratio < 0.35) & atr_exp & (m_c4h < m_ema50_4h) & (~btc_bull_4h)
    test_round(231, "Hybrid: Taker Surge+ATR+BTC Bull", sig_l, sig_s, hold=18)
    
    # R232: Hybrid 2: 4h Trend + Top-Trader Long Accumulation + Volume Breakout
    sig_l = (m_c4h > m_ema50_4h) & (tt_ls1h > 1.3) & (v1h > ind.nb_sma(v1h, 24) * 2.0) & (c1h > don_u_24)
    test_round(232, "Hybrid: TT Long+Vol+4h Break", sig_l, empty_sig, hold=24)
    
    # R233: Hybrid 3: Macro Bull + Funding Discount + Donchian Breakout
    sig_l = (m_c1d > m_sma100_1d) & (fr1h < 0.0001) & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    test_round(233, "Hybrid: 1d Bull+FR Discount+Break", sig_l, empty_sig, hold=24)
    
    # R234: Hybrid 4: Liquidation Cascade Flush + Top-Trader Rebound
    oi_pct4 = np.where(np.roll(oi1h, 4, axis=0) > 0, (oi1h - np.roll(oi1h, 4, axis=0)) / np.roll(oi1h, 4, axis=0), 0.0)
    sig_l = (oi_pct4 < -0.15) & (tt_ls1h > 1.2) & (rsi14_1h < 25) & (c1h > o1h)
    test_round(234, "Hybrid: OI Flush+TT Rebound", sig_l, empty_sig, hold=18)
    
    # R235: Hybrid 5: FVG Retest + 4h SuperTrend Bull + Low Funding
    fvg_bull = np.roll(l1h, 1, axis=0) > np.roll(h1h, 3, axis=0)
    fvg_top = np.roll(l1h, 1, axis=0)
    fvg_bot = np.roll(h1h, 3, axis=0)
    retest_bull = fvg_bull & (l1h <= fvg_top) & (c1h >= fvg_bot) & (c1h > o1h)
    sig_l = retest_bull & (m_st_trend_4h == 1) & (fr1h < 0.0002)
    test_round(235, "Hybrid: FVG+4h ST+Low FR", sig_l, empty_sig, hold=24)
    
    # R236: Hybrid 6: Cross-Sectional Leader + 1h Dip Reclaim
    sig_l = (ret24 > 0.06) & (c_prev < ema20_1h) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    test_round(236, "Hybrid: XS Leader 20-EMA Dip Reclaim", sig_l, empty_sig, hold=24)
    
    # R237: Hybrid 7: Negative Funding + High OI + Breakout Follow (Short Squeeze Hunter)
    sig_l = (fr1h < -0.0003) & (oi_pct_chg24 > 0.08) & (c1h > don_u_24)
    test_round(237, "Hybrid: Short Squeeze Triple Trigger", sig_l, empty_sig, hold=24)
    
    # R238: Hybrid 8: Mean Reversion with Trend Regime Gating & Dynamic ATR Exit
    sig_l = (m_c1d > m_sma100_1d) & (rsi14_1h < 25) & (c1h > o1h)
    test_round(238, "Hybrid: Trend Dip + Dynamic SL/TP", sig_l, empty_sig, hold=24, sl=0.025, tp=0.05)
    
    # R239: Hybrid 9: Breakout with Volume & OI Dual-Confirmation
    vol_surge = v1h > ind.nb_sma(v1h, 24) * 2.0
    oi_surge = oi_pct_chg24 > 0.10
    sig_l = vol_surge & oi_surge & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = vol_surge & oi_surge & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(239, "Hybrid: Vol + OI Dual Confirmation", sig_l, sig_s, hold=24)
    
    # R240: Hybrid 10: Chaikin Money Flow + SuperTrend Consensus
    sig_l = (cmf_1h > 0.12) & (m_st_trend_4h == 1) & (c1h > ema20_1h) & (c_prev <= ema20_1h)
    sig_s = (cmf_1h < -0.12) & (m_st_trend_4h == -1) & (c1h < ema20_1h) & (c_prev >= ema20_1h)
    test_round(240, "Hybrid: CMF Flow + 4h ST Cross", sig_l, sig_s, hold=24)
    
    # R241: Hybrid 11: Multi-Timeframe Keltner Breakout + ADX > 25
    sig_l = (c1h > kelt_u) & (adx_1h > 25) & (m_c4h > m_ema50_4h)
    sig_s = (c1h < kelt_l) & (adx_1h > 25) & (m_c4h < m_ema50_4h)
    test_round(241, "Hybrid: Keltner Break + ADX>25", sig_l, sig_s, hold=24)
    
    # R242: Hybrid 12: Smart Money Long / Retail Short Divergence + Momentum
    sig_l = (tt_ls1h > 1.25) & (taker_ratio > 0.58) & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    test_round(242, "Hybrid: Smart Money Long Divergence", sig_l, empty_sig, hold=24)
    
    # R243: Hybrid 13: High-Beta Alt Momentum + BTC Trend Protection
    natr = np.where(c1h > 0, atr14_1h / c1h, 0.0)
    high_beta = natr > ind.nb_sma(natr, 72) * 1.1
    sig_l = high_beta & (ret24 > 0.04) & btc_bull_4h & (m_c4h > m_ema50_4h)
    test_round(243, "Hybrid: High-Beta Alt + BTC Protection", sig_l, empty_sig, hold=24)
    
    # R244: Hybrid 14: Dynamic Volatility Trailing Stop on Breakout
    test_round(244, "Hybrid: Donchian + ATR SL/TP (2:4)", sig_l, empty_sig, hold=36, sl=0.02, tp=0.04)
    
    # R245: Hybrid 15: Confluence of 3 Factors (EMA + ST + CMF)
    f1 = c1h > ema20_1h
    f2 = m_st_trend_4h == 1
    f3 = cmf_1h > 0.05
    f4 = btc_bull_4h
    confluence = (f1.astype(int) + f2.astype(int) + f3.astype(int) + f4.astype(int)) >= 3
    conf_cross = confluence & (~np.roll(confluence, 1, axis=0))
    test_round(245, "Hybrid: 3-of-4 Confluence Consensus", conf_cross, empty_sig, hold=24)
    
    # R246: Hybrid 16: Multi-Factor Score Model
    score = (
        (c1h > ema20_1h).astype(float) * 1.0 +
        (m_c4h > m_ema50_4h).astype(float) * 1.5 +
        (taker_ratio > 0.55).astype(float) * 1.0 +
        (tt_ls1h > 1.15).astype(float) * 1.0 +
        (fr1h < 0.0001).astype(float) * 0.5
    )
    score_cross = (score >= 4.0) & (np.roll(score, 1, axis=0) < 4.0)
    test_round(246, "Hybrid: Multi-Factor Composite >=4.0", score_cross, empty_sig, hold=24)
    
    # R247: Hybrid 17: Multi-Factor Composite with BTC Gate
    sig_l = score_cross & btc_bull_4h
    test_round(247, "Hybrid: Multi-Factor + BTC Bull Gate", sig_l, empty_sig, hold=24)
    
    # R248: Hybrid 18: Regime-Switching Ensemble (Trend in Bull, Mean Reversion in Chop)
    trend_arm = (chop_1h < 45) & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    _, _, low_bb, _, pct_b = ind.nb_bollinger(c1h, 20, 2.0)
    mr_arm = (chop_1h > 60) & (pct_b < 0.05)
    sig_l = trend_arm | mr_arm
    test_round(248, "Hybrid: Regime-Switching Dual Arm", sig_l, empty_sig, hold=18)
    
    # R249: Hybrid 19: High-Conviction Taker Breakout with Trailing SL/TP
    sig_l = (taker_ratio > 0.65) & (c1h > don_u_24) & (m_c4h > m_ema50_4h) & btc_bull_4h
    test_round(249, "Hybrid: Conviction Breakout + SL/TP", sig_l, empty_sig, hold=24, sl=0.025, tp=0.05)
    
    # R250: Hybrid 20: Master Multi-Timeframe System (Macro 1d + Trend 4h + Timing 1h)
    sig_l = (m_c1d > m_sma100_1d) & (m_c4h > m_ema50_4h) & (c_prev <= ema20_1h) & (c1h > ema20_1h) & (cmf_1h > 0.0) & btc_bull_4h
    sig_s = (m_c1d < m_sma100_1d) & (m_c4h < m_ema50_4h) & (c_prev >= ema20_1h) & (c1h < ema20_1h) & (cmf_1h < 0.0) & (~btc_bull_4h)
    test_round(250, "Master MTF 1d/4h/1h Consensus", sig_l, sig_s, hold=24)
    
    df_res = pd.DataFrame(results)
    seg = os.environ.get("RUN_SEG", "TRAIN")
    out_csv = "/root/freqtrade/user_data/minute_research/r3c/batch5_results.csv" if seg == "TRAIN" else f"/root/freqtrade/user_data/minute_research/r3c/batch5_valid_results.csv"
    df_res.to_csv(out_csv, index=False)
    print("\nBatch 5 completed! Saved to batch5_results.csv")


if __name__ == "__main__":
    run_batch5()
