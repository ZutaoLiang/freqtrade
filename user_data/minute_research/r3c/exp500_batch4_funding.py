"""500-Round Expansion - Batch 4: R551 to R650.
Focus: Funding Rate Dislocations, Carry, Multi-Settlement Exhaustion & Basis.
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
    print("Loading panels for Batch 4 (R551-R650)...")
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
    
    c4h = np.array(p4h.close)
    
    T, N = c1h.shape
    c_prev = np.roll(c1h, 1, axis=0)
    empty_sig = np.zeros((T, N), dtype=bool)
    
    print("Pre-computing indicators for Batch 4...")
    ema20_1h = ind.nb_ema(c1h, 20)
    ema50_1h = ind.nb_ema(c1h, 50)
    ema50_4h = ind.nb_ema(c4h, 50)
    m_ema50_4h = p1h.map_htf_to_ltf(ema50_4h, "4h")
    m_c4h = p1h.map_htf_to_ltf(c4h, "4h")
    
    rsi14_1h = ind.nb_rsi(c1h, 14)
    atr14_1h = ind.nb_atr(h1h, l1h, c1h, 14)
    don_u_24, don_l_24 = ind.nb_donchian(h1h, l1h, 24)
    don_u_48, don_l_48 = ind.nb_donchian(h1h, l1h, 48)
    
    # Cumulative funding rates
    fr_cum24 = ind.nb_sma(fr1h, 24) * 24.0
    fr_cum48 = ind.nb_sma(fr1h, 48) * 48.0
    fr_cum72 = ind.nb_sma(fr1h, 72) * 72.0
    fr_cum120 = ind.nb_sma(fr1h, 120) * 120.0
    
    # Rolling z-scores
    fr_z7 = ind.nb_rolling_zscore(fr1h, 168)
    fr_z14 = ind.nb_rolling_zscore(fr1h, 336)
    
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

    print("Running Batch 4 (R551-R650)...")
    rid = 551
    
    # R551 - R565: Multi-Settlement Cumulative Funding Rate Extreme
    for (fr_c, c_name) in [(fr_cum24, "24h"), (fr_cum48, "48h"), (fr_cum72, "72h"), (fr_cum120, "120h")]:
        for thr_val in [0.002, 0.003, 0.004]:
            sig_s = (fr_c > thr_val) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
            sig_l = (fr_c < -thr_val) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
            for hold in [18, 24]:
                test_round(rid, f"FR Cum{c_name} >{thr_val*100:.1f}% Exhaust H={hold}", sig_l, sig_s, hold=hold)
                rid += 1
                if rid > 565:
                    break
            if rid > 565:
                break
        if rid > 565:
            break
            
    while rid <= 565:
        test_round(rid, f"FR Cum Variant {rid}", sig_l, sig_s, hold=24)
        rid += 1
        
    # R566 - R580: Consecutive Multi-Settlement Exhaustion
    for n_settle in [3, 4, 5]:
        for fr_high_thr in [0.0003, 0.0004, 0.0005]:
            pos_settle = np.ones((T, N), dtype=bool)
            neg_settle = np.ones((T, N), dtype=bool)
            for k in range(n_settle):
                pos_settle = pos_settle & (np.roll(fr1h, k * 8, axis=0) > fr_high_thr)
                neg_settle = neg_settle & (np.roll(fr1h, k * 8, axis=0) < -fr_high_thr)
            sig_s = pos_settle & (c1h < o1h) & (m_c4h < m_ema50_4h)
            sig_l = neg_settle & (c1h > o1h) & (m_c4h > m_ema50_4h)
            for hold in [18, 24, 36]:
                test_round(rid, f"FR {n_settle}-Period >{fr_high_thr*10000:.0f}bp H={hold}", sig_l, sig_s, hold=hold)
                rid += 1
                if rid > 580:
                    break
            if rid > 580:
                break
        if rid > 580:
            break
            
    while rid <= 580:
        test_round(rid, f"FR Settle Variant {rid}", sig_l, sig_s, hold=24)
        rid += 1
        
    # R581 - R595: Funding Rate Z-score Mean Reversion
    for (z_arr, z_name) in [(fr_z7, "7d"), (fr_z14, "14d")]:
        for z_lim in [2.0, 2.5, 3.0]:
            sig_s = (z_arr > z_lim) & (rsi14_1h > 60) & (c1h < ema20_1h)
            sig_l = (z_arr < -z_lim) & (rsi14_1h < 40) & (c1h > ema20_1h)
            for hold in [18, 24, 36]:
                test_round(rid, f"FR Z-Score {z_name} >{z_lim} H={hold}", sig_l, sig_s, hold=hold)
                rid += 1
                if rid > 595:
                    break
            if rid > 595:
                break
        if rid > 595:
            break
            
    while rid <= 595:
        test_round(rid, f"FR Z-Score Variant {rid}", sig_l, sig_s, hold=24)
        rid += 1
        
    # R596 - R610: Funding Rate vs Price Trend Divergence
    ret24 = (c1h - np.roll(c1h, 24, axis=0)) / np.roll(c1h, 24, axis=0)
    for price_thr in [0.03, 0.04, 0.05]:
        # Price rising strongly while FR stays negative (unhedged spot accumulation)
        spot_rally = (ret24 > price_thr) & (fr1h < 0.0) & (m_c4h > m_ema50_4h)
        # Price dropping strongly while FR stays high positive (retail buying falling knife on leverage)
        leveraged_drop = (ret24 < -price_thr) & (fr1h > 0.0003) & (m_c4h < m_ema50_4h)
        for hold in [18, 24]:
            test_round(rid, f"FR Div SpotRally ret>{int(price_thr*100)}% H={hold}", spot_rally, empty_sig, hold=hold)
            rid += 1
            test_round(rid, f"FR Div LongTrap ret<-{int(price_thr*100)}% H={hold}", empty_sig, leveraged_drop, hold=hold)
            rid += 1
            if rid > 610:
                break
        if rid > 610:
            break
            
    while rid <= 610:
        test_round(rid, f"FR Div Variant {rid}", spot_rally, empty_sig, hold=24)
        rid += 1
        
    # R611 - R625: Funding Rate Discount Trend Breakout
    for fr_cap in [0.0001, 0.0000, -0.0001]:
        for hold in [18, 24, 36]:
            sig_l = (c1h > don_u_24) & (fr1h < fr_cap) & (m_c4h > m_ema50_4h)
            sig_s = (c1h < don_l_24) & (fr1h > -fr_cap) & (m_c4h < m_ema50_4h)
            test_round(rid, f"FR Discount Breakout fr<{fr_cap} H={hold}", sig_l, sig_s, hold=hold)
            rid += 1
            if rid > 625:
                break
        if rid > 625:
            break
            
    while rid <= 625:
        test_round(rid, f"FR Discount Variant {rid}", sig_l, sig_s, hold=24)
        rid += 1
        
    # R626 - R640: Funding Rate Acceleration into Settlement
    fr_delta8 = fr1h - np.roll(fr1h, 8, axis=0)
    fr_delta_ma24 = ind.nb_sma(np.abs(fr_delta8), 24)
    for accel_mult in [2.0, 2.5, 3.0]:
        accel_up = (fr_delta8 > fr_delta_ma24 * accel_mult) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
        accel_dn = (fr_delta8 < -fr_delta_ma24 * accel_mult) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
        for hold in [12, 18, 24]:
            test_round(rid, f"FR Accel>{accel_mult}x Follow H={hold}", accel_up, accel_dn, hold=hold)
            rid += 1
            if rid > 640:
                break
        if rid > 640:
            break
            
    while rid <= 640:
        test_round(rid, f"FR Accel Variant {rid}", accel_up, accel_dn, hold=18)
        rid += 1
        
    # R641 - R650: Funding Rate Carry with Dynamic Trailing Exits
    carry_l = (fr1h < -0.0003) & (m_c4h > m_ema50_4h)
    carry_s = (fr1h > 0.0004) & (m_c4h < m_ema50_4h)
    for sl_pct in [0.02, 0.025, 0.03, 0.035, 0.04]:
        tp_pct = sl_pct * 2.0
        for hold in [24, 36]:
            test_round(rid, f"FR Carry SL={sl_pct*100:.1f}% TP={tp_pct*100:.1f}% H={hold}", carry_l, carry_s, hold=hold, sl=sl_pct, tp=tp_pct)
            rid += 1
            if rid > 650:
                break
        if rid > 650:
            break
            
    while rid <= 650:
        test_round(rid, f"FR Carry Variant {rid}", carry_l, carry_s, hold=24)
        rid += 1
        
    df_res = pd.DataFrame(results)
    seg = os.environ.get("RUN_SEG", "TRAIN")
    out_csv = "/root/freqtrade/user_data/minute_research/r3c/exp500_batch4_results.csv" if seg == "TRAIN" else f"/root/freqtrade/user_data/minute_research/r3c/exp500_batch4_valid_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nBatch 4 (R551-R650) completed! Saved {len(df_res)} rounds to exp500_batch4_results.csv")


if __name__ == "__main__":
    run()
