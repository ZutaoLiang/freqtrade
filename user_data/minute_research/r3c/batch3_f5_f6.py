"""Batch 3: Family 5 (R131-R150: Funding Rate & Derivatives) & Family 6 (R151-R170: Open Interest Cascades)."""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append("/root/freqtrade/user_data/minute_research/r3c")
from engine200 import PanelData, evaluate_strategy
import indicators as ind


def run_batch3():
    print("Loading 1h, 4h panels for Batch 3...")
    t0 = time.time()
    p1h = PanelData("1h")
    p4h = PanelData("4h")
    print(f"Panels loaded in {time.time()-t0:.2f}s.")
    
    c1h = np.array(p1h.close)
    o1h = np.array(p1h.open)
    h1h = np.array(p1h.high)
    l1h = np.array(p1h.low)
    v1h = np.array(p1h.volume)
    fr1h = np.array(p1h.funding_rate) if p1h.funding_rate is not None else np.zeros_like(c1h)
    oi1h = np.array(p1h.oi) if p1h.oi is not None else np.zeros_like(c1h)
    tt_ls1h = np.array(p1h.toptrader_ls) if p1h.toptrader_ls is not None else np.ones_like(c1h)
    
    c4h = np.array(p4h.close)
    h4h = np.array(p4h.high)
    l4h = np.array(p4h.low)
    
    print("Pre-computing multi-timeframe indicators for Batch 3...")
    ema20_1h = ind.nb_ema(c1h, 20)
    ema50_1h = ind.nb_ema(c1h, 50)
    atr14_1h = ind.nb_atr(h1h, l1h, c1h, 14)
    rsi14_1h = ind.nb_rsi(c1h, 14)
    don_u_24, don_l_24 = ind.nb_donchian(h1h, l1h, 24)
    don_u_48, don_l_48 = ind.nb_donchian(h1h, l1h, 48)
    
    # 4h indicators mapped causally
    ema50_4h = ind.nb_ema(c4h, 50)
    ema100_4h = ind.nb_ema(c4h, 100)
    m_ema50_4h = p1h.map_htf_to_ltf(ema50_4h, "4h")
    m_ema100_4h = p1h.map_htf_to_ltf(ema100_4h, "4h")
    m_c4h = p1h.map_htf_to_ltf(c4h, "4h")
    
    st_trend_4h, _ = ind.nb_supertrend(h4h, l4h, c4h, 10, 3.0)
    m_st_trend_4h = p1h.map_htf_to_ltf(st_trend_4h.astype(np.float64), "4h")
    
    # Funding rate indicators
    fr_cum72 = ind.nb_sma(fr1h, 72) * 72.0
            
    fr_zscore = ind.nb_rolling_zscore(fr1h, 336) # 14-day rolling zscore
    fr_ma24 = ind.nb_sma(fr1h, 24)
    
    # OI indicators
    oi_prev24 = np.roll(oi1h, 24, axis=0)
    oi_pct_chg24 = np.where(oi_prev24 > 0, (oi1h - oi_prev24) / oi_prev24, 0.0)
    
    oi_prev4 = np.roll(oi1h, 4, axis=0)
    oi_pct_chg4 = np.where(oi_prev4 > 0, (oi1h - oi_prev4) / oi_prev4, 0.0)
    
    oi_ma168 = ind.nb_sma(oi1h, 168)
    
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

    print("Executing Batch 3 (R131-R170) on TRAIN...")
    
    # --- FAMILY 5: Funding Rate & Derivatives Microstructure (R131 - R150) ---
    # R131: Cumulative 3-day Funding Rate Dislocation Long (FR cum < -0.3%)
    sig_l = (fr_cum72 < -0.003) & (c1h > ema20_1h)
    test_round(131, "FR Cum72h < -0.3% Long Rebound", sig_l, empty_sig, hold=24)
    
    # R132: Cumulative 3-day Funding Rate Dislocation Short (FR cum > +0.3%)
    sig_s = (fr_cum72 > 0.003) & (c1h < ema20_1h)
    test_round(132, "FR Cum72h > +0.3% Short Exhaust", empty_sig, sig_s, hold=24)
    
    # R133: Funding Rate Momentum (FR delta across consecutive 8h > +0.02%)
    fr_delta8 = fr1h - np.roll(fr1h, 8, axis=0)
    sig_l = (fr_delta8 > 0.0002) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = (fr_delta8 < -0.0002) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(133, "FR 8h Delta Momentum Follow", sig_l, sig_s, hold=24)
    
    # R134: Funding Rate vs Price Divergence (Price up, FR down/negative)
    ret24 = (c1h - np.roll(c1h, 24, axis=0)) / np.roll(c1h, 24, axis=0)
    sig_l = (ret24 > 0.03) & (fr1h < 0.0) & (m_c4h > m_ema50_4h)
    test_round(134, "Spot-Led Rally (Price Up, FR<=0)", sig_l, empty_sig, hold=24)
    
    # R135: Extreme Negative Funding Rate Short Squeeze Hunter (FR < -0.05% per 8h + breakout)
    sig_l = (fr1h < -0.0005) & (c1h > don_u_24)
    test_round(135, "FR<-0.05% Short Squeeze Break", sig_l, empty_sig, hold=24)
    
    # R136: Extreme Positive Funding Rate Long Squeeze Hunter (FR > +0.08% per 8h + breakdown)
    sig_s = (fr1h > 0.0008) & (c1h < don_l_24)
    test_round(136, "FR>+0.08% Long Flush Breakdown", empty_sig, sig_s, hold=24)
    
    # R137: Funding Rate Moving Average Cross (FR > FR MA24 + Price Trend)
    sig_l = (fr1h > fr_ma24) & (np.roll(fr1h, 1, axis=0) <= np.roll(fr_ma24, 1, axis=0)) & (m_c4h > m_ema50_4h)
    sig_s = (fr1h < fr_ma24) & (np.roll(fr1h, 1, axis=0) >= np.roll(fr_ma24, 1, axis=0)) & (m_c4h < m_ema50_4h)
    test_round(137, "FR MA Cross + 4h Trend", sig_l, sig_s, hold=24)
    
    # R138: Multi-settlement Funding Exhaustion Short with Concentration Limit
    # 3 consecutive positive funding bars with FR > 0.04%
    fr_high3 = (fr1h > 0.0004) & (np.roll(fr1h, 8, axis=0) > 0.0004) & (np.roll(fr1h, 16, axis=0) > 0.0004)
    sig_s = fr_high3 & (c1h < o1h) & (m_c4h < m_ema50_4h)
    test_round(138, "Multi-Settlement FR Short (Filtered)", empty_sig, sig_s, hold=24)
    
    # R139: Funding Rate Z-score Dislocation Mean Reversion (Z > 2.5 short, Z < -2.5 long)
    sig_l = (fr_zscore < -2.5) & (rsi14_1h < 35)
    sig_s = (fr_zscore > 2.5) & (rsi14_1h > 65)
    test_round(139, "FR 14d Z-Score Mean Reversion", sig_l, sig_s, hold=24)
    
    # R140: High FR Alt Basket Short with 4h BTC Bear
    sig_s = (fr1h > 0.0005) & (m_c4h < m_ema50_4h) & (c1h < ema20_1h)
    test_round(140, "High-FR Alt Short in Macro Bear", empty_sig, sig_s, hold=24)
    
    # R141: Zero-Funding Transition Momentum (FR crosses from negative to positive with price expansion)
    fr_trans_up = (np.roll(fr1h, 1, axis=0) < 0) & (fr1h > 0)
    sig_l = fr_trans_up & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    test_round(141, "FR Zero-Cross Momentum Burst", sig_l, empty_sig, hold=24)
    
    # R142: Extreme Funding Rate Decay Reversal (FR spike > 0.1% drops back down)
    fr_spike_decay = (np.roll(fr1h, 8, axis=0) > 0.001) & (fr1h < 0.0003)
    sig_s = fr_spike_decay & (c1h < ema20_1h)
    test_round(142, "FR Extreme Spike Decay Fade", empty_sig, sig_s, hold=24)
    
    # R143: Funding Rate Regime-Gated Trend (Only buy when FR < 0.01%, only short when FR > -0.005%)
    sig_l = (fr1h < 0.0001) & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = (fr1h > -0.00005) & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(143, "FR-Discount Trend Following", sig_l, sig_s, hold=24)
    
    # R144: Funding Rate Normalized by Volatility (FR / ATR)
    fr_norm = np.where(atr14_1h > 0, fr1h / (atr14_1h / c1h), 0.0)
    sig_l = (fr_norm < -0.02) & (c1h > ema20_1h)
    sig_s = (fr_norm > 0.02) & (c1h < ema20_1h)
    test_round(144, "FR Volatility-Normalized Stretch", sig_l, sig_s, hold=24)
    
    # R145: Funding Rate Capitulation Bottom Rebound (FR < -0.05% + RSI < 25)
    sig_l = (fr1h < -0.0005) & (rsi14_1h < 25) & (c1h > o1h)
    test_round(145, "FR Capitulation Rebound Long", sig_l, empty_sig, hold=18)
    
    # R146: FR Dislocation with Tight ATR Stop/TP
    test_round(146, "FR Dislocation + ATR SL/TP", sig_l, empty_sig, hold=24, sl=0.025, tp=0.05)
    
    # R147: Multi-Period Funding Exhaustion (5 consecutive positive funding rates)
    fr_pos5 = True
    for k in range(5):
        fr_pos5 = fr_pos5 & (np.roll(fr1h, k * 8, axis=0) > 0.0003)
    sig_s = fr_pos5 & (c1h < ema20_1h)
    test_round(147, "FR 5-Period Exhaustion Short", empty_sig, sig_s, hold=36)
    
    # R148: Funding Rate Carry with Volatility Filter
    sig_l = (fr1h < -0.0003) & (c1h > m_ema50_4h)
    sig_s = (fr1h > 0.0005) & (c1h < m_ema50_4h)
    test_round(148, "FR Trend-Aligned Carry Bias", sig_l, sig_s, hold=36)
    
    # R149: FR Divergence: New 48h Price Low but FR Rising (Bullish Absorption)
    fr_rising = fr1h > np.roll(fr1h, 24, axis=0)
    sig_l = (c1h <= don_l_48) & fr_rising & (rsi14_1h < 30)
    test_round(149, "FR Bullish Divergence at Lows", sig_l, empty_sig, hold=24)
    
    # R150: FR Spike Mean Reversion + 4h SuperTrend
    sig_l = (fr1h < -0.0004) & (m_st_trend_4h == 1)
    sig_s = (fr1h > 0.0006) & (m_st_trend_4h == -1)
    test_round(150, "FR Extreme + 4h SuperTrend", sig_l, sig_s, hold=24)

    
    # --- FAMILY 6: Open Interest & Positioning Cascades (R151 - R170) ---
    # R151: OI Surge + Price Breakout (OI expands > 15% in 24h + breaks 24h high)
    sig_l = (oi_pct_chg24 > 0.15) & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = (oi_pct_chg24 > 0.15) & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(151, "OI Surge + Price Breakout Follow", sig_l, sig_s, hold=24)
    
    # R152: OI Surge + Price Stalled (OI expands > 20% while price moves < 1%)
    oi_absorbed = (oi_pct_chg24 > 0.20) & (np.abs(ret24) < 0.015)
    sig_l = np.roll(oi_absorbed, 1, axis=0) & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = np.roll(oi_absorbed, 1, axis=0) & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(152, "OI Absorption Coiled Spring Break", sig_l, sig_s, hold=24)
    
    # R153: OI Collapse + Price Waterfall (Liquidation cascade continuation: OI drops > 15% in 4h)
    sig_s = (oi_pct_chg4 < -0.15) & (c1h < o1h) & (m_c4h < m_ema50_4h)
    test_round(153, "OI Collapse Cascade Continuation", empty_sig, sig_s, hold=18)
    
    # R154: OI Collapse + Price Climax Snapback (OI drops > 20% in 4h, RSI < 20)
    sig_l = (oi_pct_chg4 < -0.20) & (rsi14_1h < 22) & (c1h > o1h)
    test_round(154, "OI Flush Climax Snapback Long", sig_l, empty_sig, hold=12)
    
    # R155: Multi-Timeframe OI Trend Alignment (OI > 7d SMA + Price > 4h 50 EMA)
    sig_l = (oi1h > oi_ma168 * 1.15) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = (oi1h > oi_ma168 * 1.15) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(155, "OI 7d Expansion + Trend Consensus", sig_l, sig_s, hold=24)
    
    # R156: OI Divergence Bearish (Price makes 48h high, but OI drops)
    sig_s = (c1h >= don_u_48) & (oi_pct_chg24 < -0.05) & (m_c4h < m_ema50_4h)
    test_round(156, "OI Bearish Divergence at Highs", empty_sig, sig_s, hold=24)
    
    # R157: OI Divergence Bullish (Price makes 48h low, but OI expands > 10%)
    sig_l = (c1h <= don_l_48) & (oi_pct_chg24 > 0.10) & (rsi14_1h < 30)
    test_round(157, "OI Bullish Accumulation at Lows", sig_l, empty_sig, hold=24)
    
    # R158: OI Build-up with Negative Funding (Short Squeeze Hunter)
    sig_l = (oi_pct_chg24 > 0.10) & (fr1h < -0.0003) & (c1h > ema20_1h)
    test_round(158, "OI Build + Negative FR Squeeze", sig_l, empty_sig, hold=24)
    
    # R159: OI Build-up with Positive Funding (Long Squeeze Hunter)
    sig_s = (oi_pct_chg24 > 0.10) & (fr1h > 0.0004) & (c1h < ema20_1h)
    test_round(159, "OI Build + Positive FR Long Squeeze", empty_sig, sig_s, hold=24)
    
    # R160: Post-Liquidation Re-accumulation Reclaim (OI was down > 15%, now flat, price reclaims 20 EMA)
    oi_down_prior = np.roll(oi_pct_chg24, 12, axis=0) < -0.15
    oi_stabilizing = np.abs(oi_pct_chg4) < 0.03
    sig_l = oi_down_prior & oi_stabilizing & (c1h > ema20_1h) & (c_prev <= ema20_1h)
    test_round(160, "Post-Flush Stabilization Reclaim", sig_l, empty_sig, hold=24)
    
    # R161: Top-Trader LS + OI Joint Expansion (Top trader long ratio > 1.25 and OI rising > 10%)
    sig_l = (tt_ls1h > 1.25) & (oi_pct_chg24 > 0.10) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = (tt_ls1h < 0.80) & (oi_pct_chg24 > 0.10) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(161, "Top-Trader + OI Joint Expansion", sig_l, sig_s, hold=24)
    
    # R162: OI Velocity Spike (1h OI change > 3x average 1h OI change)
    oi_delta1 = np.abs(oi1h - np.roll(oi1h, 1, axis=0))
    oi_delta_ma24 = ind.nb_sma(oi_delta1, 24)
    sig_l = (oi_delta1 > oi_delta_ma24 * 3.0) & (c1h > o1h) & (m_c4h > m_ema50_4h)
    sig_s = (oi_delta1 > oi_delta_ma24 * 3.0) & (c1h < o1h) & (m_c4h < m_ema50_4h)
    test_round(162, "OI Velocity Spike Follow", sig_l, sig_s, hold=18)
    
    # R163: OI Surge + Donchian Breakout + SL/TP (2.5% SL, 5% TP)
    test_round(163, "OI Breakout + Tight SL/TP", sig_l, sig_s, hold=24, sl=0.025, tp=0.05)
    
    # R164: OI Exhaustion at 7d Extreme High
    oi_max168 = ind.nb_donchian(oi1h, oi1h, 168)[0]
    oi_at_ath = oi1h >= oi_max168 * 0.98
    sig_s = oi_at_ath & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(164, "OI 7d ATH Rejection Short", empty_sig, sig_s, hold=24)
    
    # R165: Fast OI Flush (< -10% in 1h) Capitulation Reversal
    oi_pct1 = np.where(np.roll(oi1h, 1, axis=0) > 0, (oi1h - np.roll(oi1h, 1, axis=0)) / np.roll(oi1h, 1, axis=0), 0.0)
    sig_l = (oi_pct1 < -0.10) & (rsi14_1h < 25) & (c1h > o1h)
    test_round(165, "Fast 1h OI Flush Snapback", sig_l, empty_sig, hold=12)
    
    # R166: OI Compression in Low Volatility (Coiled Spring)
    _, _, _, bbw_1h, _ = ind.nb_bollinger(c1h, 20, 2.0)
    coiled = (bbw_1h < ind.nb_sma(bbw_1h, 72) * 0.7) & (oi_pct_chg24 > 0.08)
    sig_l = np.roll(coiled, 1, axis=0) & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = np.roll(coiled, 1, axis=0) & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(166, "Coiled Spring Low-Vol OI Break", sig_l, sig_s, hold=24)
    
    # R167: OI-Volume Ratio Spike (Positioning without matching trading volume)
    oi_v_ratio = np.where(v1h * c1h > 0, oi1h / (v1h * c1h), 0.0)
    sig_l = (oi_v_ratio > ind.nb_sma(oi_v_ratio, 72) * 2.0) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    sig_s = (oi_v_ratio > ind.nb_sma(oi_v_ratio, 72) * 2.0) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    test_round(167, "OI/Turnover Ratio Spike", sig_l, sig_s, hold=24)
    
    # R168: Persistent OI Growth (5 consecutive 4h periods with rising OI)
    oi_up5 = True
    for k in range(5):
        oi_up5 = oi_up5 & (np.roll(oi1h, k * 4, axis=0) > np.roll(oi1h, (k + 1) * 4, axis=0))
    sig_l = oi_up5 & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    test_round(168, "Persistent 20h OI Inflow Trend", sig_l, empty_sig, hold=24)
    
    # R169: Post-Flush Bottom Rebound with ATR Trailing Exit
    sig_l = (oi_pct_chg4 < -0.15) & (rsi14_1h < 25)
    test_round(169, "Post-Flush Bottom + Dynamic SL/TP", sig_l, empty_sig, hold=24, sl=0.03, tp=0.06)
    
    # R170: OI Momentum + 4h SuperTrend Consensus
    sig_l = (oi_pct_chg24 > 0.05) & (m_st_trend_4h == 1) & (c_prev < ema20_1h) & (c1h > ema20_1h)
    sig_s = (oi_pct_chg24 > 0.05) & (m_st_trend_4h == -1) & (c_prev > ema20_1h) & (c1h < ema20_1h)
    test_round(170, "OI Mom + 4h SuperTrend Cross", sig_l, sig_s, hold=24)
    
    df_res = pd.DataFrame(results)
    seg = os.environ.get("RUN_SEG", "TRAIN")
    out_csv = "/root/freqtrade/user_data/minute_research/r3c/batch3_results.csv" if seg == "TRAIN" else f"/root/freqtrade/user_data/minute_research/r3c/batch3_valid_results.csv"
    df_res.to_csv(out_csv, index=False)
    print("\nBatch 3 completed! Saved to batch3_results.csv")


if __name__ == "__main__":
    run_batch3()
