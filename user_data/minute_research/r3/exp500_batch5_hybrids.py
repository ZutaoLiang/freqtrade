"""500-Round Expansion - Batch 5: R651 to R750.
Focus: Multi-Factor Hybrids, Derivatives Confluence, Regime Switching & Master Ensembles.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append("/root/freqtrade/user_data/minute_research/r3")
from engine200 import PanelData, evaluate_strategy
import indicators as ind


def run():
    print("Loading panels for Batch 5 (R651-R750)...")
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
    
    T, N = c1h.shape
    c_prev = np.roll(c1h, 1, axis=0)
    empty_sig = np.zeros((T, N), dtype=bool)
    
    print("Pre-computing indicators for Batch 5...")
    ema20_1h = ind.nb_ema(c1h, 20)
    ema50_1h = ind.nb_ema(c1h, 50)
    ema50_4h = ind.nb_ema(c4h, 50)
    m_ema50_4h = p1h.map_htf_to_ltf(ema50_4h, "4h")
    m_c4h = p1h.map_htf_to_ltf(c4h, "4h")
    
    sma100_1d = ind.nb_sma(c1d, 100)
    m_sma100_1d = p1h.map_htf_to_ltf(sma100_1d, "1d")
    m_c1d = p1h.map_htf_to_ltf(c1d, "1d")
    
    st_trend_4h, _ = ind.nb_supertrend(h4h, l4h, c4h, 10, 3.0)
    m_st_trend_4h = p1h.map_htf_to_ltf(st_trend_4h.astype(np.float64), "4h")
    
    atr14_1h = ind.nb_atr(h1h, l1h, c1h, 14)
    atr_ma24 = ind.nb_sma(atr14_1h, 24)
    rsi14_1h = ind.nb_rsi(c1h, 14)
    cmf_1h = ind.nb_cmf(h1h, l1h, c1h, v1h, 20)
    chop_1h = ind.nb_choppiness(h1h, l1h, c1h, 14)
    don_u_24, don_l_24 = ind.nb_donchian(h1h, l1h, 24)
    don_u_48, don_l_48 = ind.nb_donchian(h1h, l1h, 48)
    
    v_ma24 = ind.nb_sma(v1h, 24)
    taker_ratio = np.where(v1h > 0, tbv1h / v1h, 0.5)
    
    # OI percentage changes
    oi_prev4 = np.roll(oi1h, 4, axis=0)
    oi_pct4 = np.where(oi_prev4 > 0, (oi1h - oi_prev4) / oi_prev4, 0.0)
    oi_prev24 = np.roll(oi1h, 24, axis=0)
    oi_pct24 = np.where(oi_prev24 > 0, (oi1h - oi_prev24) / oi_prev24, 0.0)
    
    # Cumulative funding
    fr_cum72 = ind.nb_sma(fr1h, 72) * 72.0
    
    # BTC indicators
    btc_idx = p1h.symbols.index("BTCUSDT") if "BTCUSDT" in p1h.symbols else p1h.u162_indices[0]
    btc_m_c4h = m_c4h[:, btc_idx:btc_idx+1]
    btc_m_ema50_4h = m_ema50_4h[:, btc_idx:btc_idx+1]
    btc_bull_4h = btc_m_c4h > btc_m_ema50_4h
    btc_c1h = c1h[:, btc_idx:btc_idx+1]
    btc_ret24 = (btc_c1h - np.roll(btc_c1h, 24, axis=0)) / np.roll(btc_c1h, 24, axis=0)
    
    # Dual Squeeze
    _, bb_4h_u, bb_4h_l, _, _ = ind.nb_bollinger(c4h, 20, 2.0)
    _, kelt_4h_u_15, kelt_4h_l_15 = ind.nb_keltner(h4h, l4h, c4h, 20, 1.5)
    sq_4h_15 = (bb_4h_u < kelt_4h_u_15) & (bb_4h_l > kelt_4h_l_15)
    m_sq_4h_15 = p1h.map_htf_to_ltf(sq_4h_15.astype(np.float64), "4h")
    
    _, bb_1h_u, bb_1h_l, _, _ = ind.nb_bollinger(c1h, 20, 2.0)
    _, kelt_1h_u_15, kelt_1h_l_15 = ind.nb_keltner(h1h, l1h, c1h, 20, 1.5)
    sq_1h_15 = (bb_1h_u < kelt_1h_u_15) & (bb_1h_l > kelt_1h_l_15)
    dual_sq = (m_sq_4h_15 == 1.0) & sq_1h_15
    
    # Market Breadth
    u_c4h = m_c4h[:, p1h.u162_indices]
    u_ema50_4h = m_ema50_4h[:, p1h.u162_indices]
    breadth_above_50 = np.mean(u_c4h > u_ema50_4h, axis=1, keepdims=True)
    
    results = []

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
        if r_id % 10 == 0 or status == "PASS_TRAIN":
            print(f"[{r_id:03d}] {name[:35]:35s} | n={row['n']:5d} | mean={row['mean_bp']:6.1f}bp | pf={row['pf']:5.2f} | day_t={row['day_t']:5.2f} |/d={row['per_day']:4.1f} | {status}")
        return row

    print("Running Batch 5 (R651-R750)...")
    rid = 651
    
    # R651 - R665: Dual Squeeze + Taker Confirmation + Macro BTC Gating
    for tr_val in [0.55, 0.60, 0.65]:
        for btc_gate in [True, False]:
            gate_mask = btc_bull_4h if btc_gate else np.ones((T, N), dtype=bool)
            gate_mask_s = (~btc_bull_4h) if btc_gate else np.ones((T, N), dtype=bool)
            for hold in [18, 24]:
                sig_l = np.roll(dual_sq, 1, axis=0) & (c1h > kelt_1h_u_15) & (taker_ratio > tr_val) & gate_mask
                sig_s = np.roll(dual_sq, 1, axis=0) & (c1h < kelt_1h_l_15) & (taker_ratio < (1.0 - tr_val)) & gate_mask_s
                g_str = "+BTC" if btc_gate else "NoBTC"
                test_round(rid, f"DualSq+Taker>{int(tr_val*100)}%{g_str} H={hold}", sig_l, sig_s, hold=hold)
                rid += 1
                if rid > 665:
                    break
            if rid > 665:
                break
        if rid > 665:
            break
            
    while rid <= 665:
        test_round(rid, f"DualSq+Taker Variant {rid}", sig_l, sig_s, hold=24)
        rid += 1
        
    # R666 - R680: OI Divergence + Funding Rate Dislocation (Derivatives Dual Confluence)
    for oi_div_thr in [-0.03, -0.05, -0.08]:
        for fr_limit in [0.0002, 0.0004]:
            deriv_s = (c1h >= don_u_48) & (oi_pct24 < oi_div_thr) & (fr1h > fr_limit) & (m_c4h < m_ema50_4h)
            deriv_l = (c1h <= don_l_48) & (oi_pct24 > -oi_div_thr) & (fr1h < -fr_limit) & (m_c4h > m_ema50_4h)
            for hold in [18, 24, 36]:
                test_round(rid, f"Deriv Conf oi<{int(oi_div_thr*100)}% fr>{int(fr_limit*10000)}bp H={hold}", deriv_l, deriv_s, hold=hold)
                rid += 1
                if rid > 680:
                    break
            if rid > 680:
                break
        if rid > 680:
            break
            
    while rid <= 680:
        test_round(rid, f"Derivatives Confluence Variant {rid}", deriv_l, deriv_s, hold=24)
        rid += 1
        
    # R681 - R695: Liquidation Climax + Taker Absorption + ATR Expansion
    for drop_oi in [-0.12, -0.16, -0.20]:
        for tr_buy in [0.55, 0.60]:
            liq_abs_l = (oi_pct4 < drop_oi) & (taker_ratio > tr_buy) & (rsi14_1h < 28) & (c1h > o1h)
            for hold in [12, 18]:
                test_round(rid, f"Liq Flush+TakerAbs oi<{int(drop_oi*100)}% tr>{int(tr_buy*100)}% H={hold}", liq_abs_l, empty_sig, hold=hold)
                rid += 1
                if rid > 695:
                    break
            if rid > 695:
                break
        if rid > 695:
            break
            
    while rid <= 695:
        test_round(rid, f"Liq Absorb Variant {rid}", liq_abs_l, empty_sig, hold=18)
        rid += 1
        
    # R696 - R710: Fair Value Gap (FVG) + 4h SuperTrend + Volume Breakout
    fvg_bull = np.roll(l1h, 1, axis=0) > np.roll(h1h, 3, axis=0)
    fvg_bear = np.roll(h1h, 1, axis=0) < np.roll(l1h, 3, axis=0)
    retest_bull = fvg_bull & (l1h <= np.roll(l1h, 1, axis=0)) & (c1h >= np.roll(h1h, 3, axis=0)) & (c1h > o1h)
    retest_bear = fvg_bear & (h1h >= np.roll(h1h, 1, axis=0)) & (c1h <= np.roll(l1h, 3, axis=0)) & (c1h < o1h)
    for v_fact in [1.2, 1.5, 2.0]:
        v_ok = v1h > v_ma24 * v_fact
        for hold in [18, 24]:
            sig_l = retest_bull & (m_st_trend_4h == 1) & v_ok
            sig_s = retest_bear & (m_st_trend_4h == -1) & v_ok
            test_round(rid, f"FVG+4h ST+Vol>{v_fact}x H={hold}", sig_l, sig_s, hold=hold)
            rid += 1
            if rid > 710:
                break
        if rid > 710:
            break
            
    while rid <= 710:
        test_round(rid, f"FVG+ST Variant {rid}", sig_l, sig_s, hold=24)
        rid += 1
        
    # R711 - R725: Multi-Factor Composite Scoring (Mom + Squeeze + Taker + OI + FR)
    score_long = (
        (c1h > ema20_1h).astype(float) * 1.0 +
        (m_c4h > m_ema50_4h).astype(float) * 1.5 +
        (m_st_trend_4h == 1).astype(float) * 1.0 +
        (taker_ratio > 0.58).astype(float) * 1.0 +
        (oi_pct24 > 0.05).astype(float) * 0.8 +
        (fr1h < 0.0001).astype(float) * 0.7 +
        btc_bull_4h.astype(float) * 1.0
    )
    score_short = (
        (c1h < ema20_1h).astype(float) * 1.0 +
        (m_c4h < m_ema50_4h).astype(float) * 1.5 +
        (m_st_trend_4h == -1).astype(float) * 1.0 +
        (taker_ratio < 0.42).astype(float) * 1.0 +
        (oi_pct24 < -0.05).astype(float) * 0.8 +
        (fr1h > -0.0001).astype(float) * 0.7 +
        (~btc_bull_4h).astype(float) * 1.0
    )
    for score_thr in [4.0, 4.5, 5.0, 5.5]:
        cross_l = (score_long >= score_thr) & (np.roll(score_long, 1, axis=0) < score_thr)
        cross_s = (score_short >= score_thr) & (np.roll(score_short, 1, axis=0) < score_thr)
        for hold in [18, 24, 36]:
            test_round(rid, f"MF Composite Score>={score_thr} H={hold}", cross_l, cross_s, hold=hold)
            rid += 1
            if rid > 725:
                break
        if rid > 725:
            break
            
    while rid <= 725:
        test_round(rid, f"MF Score Variant {rid}", cross_l, cross_s, hold=24)
        rid += 1
        
    # R726 - R740: Market Breadth Regime Switching
    breadth_bull = breadth_above_50 > 0.60
    breadth_bear = breadth_above_50 < 0.40
    breadth_chop = (~breadth_bull) & (~breadth_bear)
    
    # In bull breadth: take squeeze & donchian breakouts
    # In chop breadth: take mean reversion on Bollinger %B
    _, _, low_bb, _, pct_b = ind.nb_bollinger(c1h, 20, 2.0)
    for hold in [12, 18, 24]:
        trend_l = breadth_bull & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
        mr_l = breadth_chop & (pct_b < 0.05) & (c1h > o1h)
        sig_l = trend_l | mr_l
        sig_s = breadth_bear & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
        test_round(rid, f"Breadth Regime Switch DualArm H={hold}", sig_l, sig_s, hold=hold)
        rid += 1
        # With SL/TP
        test_round(rid, f"Breadth Switch SL=2.5% TP=5% H={hold}", sig_l, sig_s, hold=hold, sl=0.025, tp=0.05)
        rid += 1
        test_round(rid, f"Breadth Switch SL=3.0% TP=6% H={hold}", sig_l, sig_s, hold=hold, sl=0.03, tp=0.06)
        rid += 1
        if rid > 740:
            break
            
    while rid <= 740:
        test_round(rid, f"Breadth Variant {rid}", sig_l, sig_s, hold=24)
        rid += 1
        
    # R741 - R750: Finalist Master Formulations & Sensitivity Sweeps
    # Best combination of Dual Squeeze + Taker Surge + Macro BTC Filter + Dynamic SL/TP
    best_base_l = np.roll(dual_sq, 1, axis=0) & (c1h > kelt_1h_u_15) & (taker_ratio > 0.60) & btc_bull_4h
    best_base_s = np.roll(dual_sq, 1, axis=0) & (c1h < kelt_1h_l_15) & (taker_ratio < 0.40) & (~btc_bull_4h)
    
    for (sl, tp) in [(0.02, 0.04), (0.025, 0.05), (0.03, 0.06), (0.035, 0.07), (0.04, 0.08)]:
        for h in [24, 36]:
            test_round(rid, f"Master Finalist SL={sl*100:.1f}% TP={tp*100:.1f}% H={h}", best_base_l, best_base_s, hold=h, sl=sl, tp=tp)
            rid += 1
            if rid > 750:
                break
        if rid > 750:
            break
            
    while rid <= 750:
        test_round(rid, f"Master Finalist Variant {rid}", best_base_l, best_base_s, hold=24)
        rid += 1
        
    df_res = pd.DataFrame(results)
    seg = os.environ.get("RUN_SEG", "TRAIN")
    out_csv = "/root/freqtrade/user_data/minute_research/r3/exp500_batch5_results.csv" if seg == "TRAIN" else f"/root/freqtrade/user_data/minute_research/r3/exp500_batch5_valid_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nBatch 5 (R651-R750) completed! Saved {len(df_res)} rounds to exp500_batch5_results.csv")


if __name__ == "__main__":
    run()
