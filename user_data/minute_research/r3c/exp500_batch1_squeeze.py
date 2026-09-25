"""500-Round Expansion - Batch 1: R251 to R350.
Focus: Multi-Timeframe Volatility Compression, Squeeze Variations, ATR Regimes & Exits.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append("/root/freqtrade/user_data/minute_research/r3c")
from engine200 import PanelData, evaluate_strategy
import indicators as ind


def run():
    print("Loading panels for Batch 1 (R251-R350)...")
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
    tbv1h = np.array(p1h.taker_buy_volume) if p1h.taker_buy_volume is not None else v1h * 0.5
    
    c4h = np.array(p4h.close)
    h4h = np.array(p4h.high)
    l4h = np.array(p4h.low)
    
    c1d = np.array(p1d.close)
    
    T, N = c1h.shape
    c_prev = np.roll(c1h, 1, axis=0)
    empty_sig = np.zeros((T, N), dtype=bool)
    
    print("Pre-computing indicators for Batch 1...")
    # Moving averages
    ema20_1h = ind.nb_ema(c1h, 20)
    ema50_1h = ind.nb_ema(c1h, 50)
    ema50_4h = ind.nb_ema(c4h, 50)
    m_ema50_4h = p1h.map_htf_to_ltf(ema50_4h, "4h")
    m_c4h = p1h.map_htf_to_ltf(c4h, "4h")
    
    sma100_1d = ind.nb_sma(c1d, 100)
    m_sma100_1d = p1h.map_htf_to_ltf(sma100_1d, "1d")
    m_c1d = p1h.map_htf_to_ltf(c1d, "1d")
    
    # ATR & Bollinger
    atr14_1h = ind.nb_atr(h1h, l1h, c1h, 14)
    atr72_1h = ind.nb_atr(h1h, l1h, c1h, 72)
    atr168_1h = ind.nb_atr(h1h, l1h, c1h, 168)
    atr_ma24 = ind.nb_sma(atr14_1h, 24)
    
    # Volume & Taker
    v_ma24 = ind.nb_sma(v1h, 24)
    taker_ratio = np.where(v1h > 0, tbv1h / v1h, 0.5)
    
    # Donchian
    don_u_24, don_l_24 = ind.nb_donchian(h1h, l1h, 24)
    
    # BTC indicators
    btc_idx = p1h.symbols.index("BTCUSDT") if "BTCUSDT" in p1h.symbols else p1h.u162_indices[0]
    btc_m_c4h = m_c4h[:, btc_idx:btc_idx+1]
    btc_m_ema50_4h = m_ema50_4h[:, btc_idx:btc_idx+1]
    btc_bull_4h = btc_m_c4h > btc_m_ema50_4h
    
    # 4h BB & Keltner
    _, bb_4h_u, bb_4h_l, bbw_4h, _ = ind.nb_bollinger(c4h, 20, 2.0)
    kelt_4h_mid_15, kelt_4h_u_15, kelt_4h_l_15 = ind.nb_keltner(h4h, l4h, c4h, 20, 1.5)
    kelt_4h_mid_12, kelt_4h_u_12, kelt_4h_l_12 = ind.nb_keltner(h4h, l4h, c4h, 20, 1.25)
    kelt_4h_mid_17, kelt_4h_u_17, kelt_4h_l_17 = ind.nb_keltner(h4h, l4h, c4h, 20, 1.75)
    
    sq_4h_15 = (bb_4h_u < kelt_4h_u_15) & (bb_4h_l > kelt_4h_l_15)
    sq_4h_12 = (bb_4h_u < kelt_4h_u_12) & (bb_4h_l > kelt_4h_l_12)
    sq_4h_17 = (bb_4h_u < kelt_4h_u_17) & (bb_4h_l > kelt_4h_l_17)
    
    m_sq_4h_15 = p1h.map_htf_to_ltf(sq_4h_15.astype(np.float64), "4h")
    m_sq_4h_12 = p1h.map_htf_to_ltf(sq_4h_12.astype(np.float64), "4h")
    m_sq_4h_17 = p1h.map_htf_to_ltf(sq_4h_17.astype(np.float64), "4h")
    
    # 1h BB & Keltner
    _, bb_1h_u, bb_1h_l, bbw_1h, _ = ind.nb_bollinger(c1h, 20, 2.0)
    kelt_1h_mid_10, kelt_1h_u_10, kelt_1h_l_10 = ind.nb_keltner(h1h, l1h, c1h, 20, 1.0)
    kelt_1h_mid_12, kelt_1h_u_12, kelt_1h_l_12 = ind.nb_keltner(h1h, l1h, c1h, 20, 1.25)
    kelt_1h_mid_15, kelt_1h_u_15, kelt_1h_l_15 = ind.nb_keltner(h1h, l1h, c1h, 20, 1.5)
    kelt_1h_mid_17, kelt_1h_u_17, kelt_1h_l_17 = ind.nb_keltner(h1h, l1h, c1h, 20, 1.75)
    kelt_1h_mid_20, kelt_1h_u_20, kelt_1h_l_20 = ind.nb_keltner(h1h, l1h, c1h, 20, 2.0)
    
    sq_1h_15 = (bb_1h_u < kelt_1h_u_15) & (bb_1h_l > kelt_1h_l_15)
    sq_1h_12 = (bb_1h_u < kelt_1h_u_12) & (bb_1h_l > kelt_1h_l_12)
    sq_1h_17 = (bb_1h_u < kelt_1h_u_17) & (bb_1h_l > kelt_1h_l_17)
    
    # Historical Volatility
    log_ret = np.log(np.maximum(c1h, 1e-8) / np.maximum(c_prev, 1e-8))
    hv12 = ind.nb_rolling_std(log_ret, 12)
    hv24 = ind.nb_rolling_std(log_ret, 24)
    hv48 = ind.nb_rolling_std(log_ret, 48)
    
    # Choppiness
    chop_1h = ind.nb_choppiness(h1h, l1h, c1h, 14)
    
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

    print("Running Batch 1 (R251-R350)...")
    
    # R251 - R265: Keltner Channel Multiple Sweeps
    mults = [(1.0, kelt_1h_u_10, kelt_1h_l_10), (1.25, kelt_1h_u_12, kelt_1h_l_12), 
             (1.5, kelt_1h_u_15, kelt_1h_l_15), (1.75, kelt_1h_u_17, kelt_1h_l_17), (2.0, kelt_1h_u_20, kelt_1h_l_20)]
    rid = 251
    for m_val, ku, kl in mults:
        for hold in [12, 24, 36]:
            dual_sq = (m_sq_4h_15 == 1.0) & sq_1h_15
            sig_l = np.roll(dual_sq, 1, axis=0) & (c1h > ku)
            sig_s = np.roll(dual_sq, 1, axis=0) & (c1h < kl)
            test_round(rid, f"DualSq 1.5x Keltner mult={m_val} H={hold}", sig_l, sig_s, hold=hold)
            rid += 1
            
    # R266 - R275: Squeeze Release with Multi-Timeframe Trend Gating
    for tf_filter, name_filt in [
        ((m_c4h > m_ema50_4h), "4h 50EMA Bull"),
        ((m_c1d > m_sma100_1d), "1d 100SMA Bull"),
        (btc_bull_4h, "BTC 4h Bull Gate"),
        ((m_c4h > m_ema50_4h) & btc_bull_4h, "Dual 4h + BTC Bull Gate"),
        ((m_c1d > m_sma100_1d) & (m_c4h > m_ema50_4h), "1d+4h Macro Bull")
    ]:
        for hold in [18, 24]:
            dual_sq = (m_sq_4h_15 == 1.0) & sq_1h_15
            sig_l = np.roll(dual_sq, 1, axis=0) & (c1h > kelt_1h_u_15) & tf_filter
            sig_s = np.roll(dual_sq, 1, axis=0) & (c1h < kelt_1h_l_15) & (~tf_filter)
            test_round(rid, f"DualSq + {name_filt} H={hold}", sig_l, sig_s, hold=hold)
            rid += 1
            
    # R276 - R285: Squeeze Release with Volume Confirmation
    for v_mult in [1.2, 1.5, 2.0, 2.5, 3.0]:
        for hold in [18, 24]:
            dual_sq = (m_sq_4h_15 == 1.0) & sq_1h_15
            v_conf = v1h > v_ma24 * v_mult
            sig_l = np.roll(dual_sq, 1, axis=0) & (c1h > kelt_1h_u_15) & v_conf
            sig_s = np.roll(dual_sq, 1, axis=0) & (c1h < kelt_1h_l_15) & v_conf
            test_round(rid, f"DualSq + Vol>{v_mult}x H={hold}", sig_l, sig_s, hold=hold)
            rid += 1
            
    # R286 - R295: Squeeze Release with Taker Ratio Confirmation
    for tr_thr in [0.55, 0.60, 0.65, 0.70, 0.75]:
        for hold in [18, 24]:
            dual_sq = (m_sq_4h_15 == 1.0) & sq_1h_15
            sig_l = np.roll(dual_sq, 1, axis=0) & (c1h > kelt_1h_u_15) & (taker_ratio > tr_thr)
            sig_s = np.roll(dual_sq, 1, axis=0) & (c1h < kelt_1h_l_15) & (taker_ratio < (1.0 - tr_thr))
            test_round(rid, f"DualSq + Taker>{int(tr_thr*100)}% H={hold}", sig_l, sig_s, hold=hold)
            rid += 1
            
    # R296 - R305: BBW Historical Percentile Pinch
    for lb in [168, 336, 720]:
        bbw_min = ind.nb_donchian(bbw_1h, bbw_1h, lb)[1]
        for pct_fact in [1.05, 1.15, 1.25]:
            compressed = bbw_1h <= bbw_min * pct_fact
            sig_l = np.roll(compressed, 1, axis=0) & (c1h > bb_1h_u) & (m_c4h > m_ema50_4h)
            sig_s = np.roll(compressed, 1, axis=0) & (c1h < bb_1h_l) & (m_c4h < m_ema50_4h)
            test_round(rid, f"BBW LB={lb} pinch={pct_fact} Break", sig_l, sig_s, hold=24)
            rid += 1
            if rid > 305:
                break
        if rid > 305:
            break
            
    # Fill up to R305 if needed
    while rid <= 305:
        test_round(rid, f"BBW Compression Variant {rid}", sig_l, sig_s, hold=18)
        rid += 1
        
    # R306 - R315: ATR Ratio Compression Breakouts
    for (s_atr, l_atr, s_name) in [(atr14_1h, atr72_1h, "14/72"), (atr14_1h, atr168_1h, "14/168")]:
        ratio = np.where(l_atr > 0, s_atr / l_atr, 1.0)
        for r_thr in [0.65, 0.70, 0.75, 0.80, 0.85]:
            sig_l = (np.roll(ratio, 1, axis=0) < r_thr) & (ratio > r_thr + 0.1) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
            sig_s = (np.roll(ratio, 1, axis=0) < r_thr) & (ratio > r_thr + 0.1) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
            test_round(rid, f"ATR Ratio {s_name} <{r_thr} Break", sig_l, sig_s, hold=24)
            rid += 1
            
    # R316 - R325: Historical Volatility Compression Breakouts
    for (hv_arr, h_name) in [(hv12, "HV12"), (hv24, "HV24"), (hv48, "HV48")]:
        hv_min = ind.nb_donchian(hv_arr, hv_arr, 72)[1]
        for mult_exp in [1.2, 1.4, 1.6]:
            sig_l = (np.roll(hv_arr, 1, axis=0) <= hv_min * 1.1) & (hv_arr > hv_min * mult_exp) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
            sig_s = (np.roll(hv_arr, 1, axis=0) <= hv_min * 1.1) & (hv_arr > hv_min * mult_exp) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
            test_round(rid, f"{h_name} Pinch exp={mult_exp}x Break", sig_l, sig_s, hold=24)
            rid += 1
            if rid > 325:
                break
        if rid > 325:
            break
            
    while rid <= 325:
        test_round(rid, f"HV Variant {rid}", sig_l, sig_s, hold=18)
        rid += 1
        
    # R326 - R335: Choppiness Index Extreme Compression (< 35) + Momentum Cross
    for chop_thr in [32, 35, 38, 40, 42]:
        for hold in [18, 24]:
            sig_l = (chop_1h < chop_thr) & (c_prev < ema20_1h) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
            sig_s = (chop_1h < chop_thr) & (c_prev > ema20_1h) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
            test_round(rid, f"CHOP<{chop_thr} Momentum Cross H={hold}", sig_l, sig_s, hold=hold)
            rid += 1
            
    # R336 - R345: Dynamic Trailing Stop / TP Variations on Squeeze Breakouts
    dual_sq = (m_sq_4h_15 == 1.0) & sq_1h_15
    base_sig_l = np.roll(dual_sq, 1, axis=0) & (c1h > kelt_1h_u_15)
    base_sig_s = np.roll(dual_sq, 1, axis=0) & (c1h < kelt_1h_l_15)
    for sl_val in [0.015, 0.02, 0.025, 0.03, 0.035]:
        tp_val = sl_val * 2.0
        for hold in [24, 36]:
            test_round(rid, f"DualSq SL={sl_val*100:.1f}% TP={tp_val*100:.1f}% H={hold}", base_sig_l, base_sig_s, hold=hold, sl=sl_val, tp=tp_val)
            rid += 1
            
    # R346 - R350: Squeeze False Breakout Reversal Fade
    # Price breaks out of Keltner but immediately closes back inside within 1 bar
    false_break_up = (np.roll(c1h, 1, axis=0) > np.roll(kelt_1h_u_15, 1, axis=0)) & (c1h < kelt_1h_u_15) & (c1h < o1h)
    false_break_dn = (np.roll(c1h, 1, axis=0) < np.roll(kelt_1h_l_15, 1, axis=0)) & (c1h > kelt_1h_l_15) & (c1h > o1h)
    for hold in [12, 18, 24, 30, 36]:
        test_round(rid, f"Squeeze False Break Fade H={hold}", false_break_dn, false_break_up, hold=hold)
        rid += 1
        
    df_res = pd.DataFrame(results)
    seg = os.environ.get("RUN_SEG", "TRAIN")
    out_csv = "/root/freqtrade/user_data/minute_research/r3c/exp500_batch1_results.csv" if seg == "TRAIN" else f"/root/freqtrade/user_data/minute_research/r3c/exp500_batch1_valid_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nBatch 1 (R251-R350) completed! Saved {len(df_res)} rounds to exp500_batch1_results.csv")


if __name__ == "__main__":
    run()
