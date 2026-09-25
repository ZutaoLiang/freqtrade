"""500-Round Expansion - Batch 3: R451 to R550.
Focus: Open Interest Dynamics, Positioning Squeezes & Liquidation Cascades.
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
    print("Loading panels for Batch 3 (R451-R550)...")
    t0 = time.time()
    p1h = PanelData("1h")
    p4h = PanelData("4h")
    print(f"Panels loaded in {time.time()-t0:.2f}s.")
    
    c1h = np.array(p1h.close)
    o1h = np.array(p1h.open)
    h1h = np.array(p1h.high)
    l1h = np.array(p1h.low)
    v1h = np.array(p1h.volume)
    qv1h = np.array(p1h.quote_volume) if p1h.quote_volume is not None else v1h * c1h
    fr1h = np.array(p1h.funding_rate) if p1h.funding_rate is not None else np.zeros_like(c1h)
    oi1h = np.array(p1h.oi) if p1h.oi is not None else np.zeros_like(c1h)
    tt_ls1h = np.array(p1h.toptrader_ls) if p1h.toptrader_ls is not None else np.ones_like(c1h)
    
    c4h = np.array(p4h.close)
    
    T, N = c1h.shape
    c_prev = np.roll(c1h, 1, axis=0)
    empty_sig = np.zeros((T, N), dtype=bool)
    
    print("Pre-computing indicators for Batch 3...")
    ema20_1h = ind.nb_ema(c1h, 20)
    ema50_1h = ind.nb_ema(c1h, 50)
    ema50_4h = ind.nb_ema(c4h, 50)
    m_ema50_4h = p1h.map_htf_to_ltf(ema50_4h, "4h")
    m_c4h = p1h.map_htf_to_ltf(c4h, "4h")
    
    rsi14_1h = ind.nb_rsi(c1h, 14)
    atr14_1h = ind.nb_atr(h1h, l1h, c1h, 14)
    don_u_24, don_l_24 = ind.nb_donchian(h1h, l1h, 24)
    don_u_48, don_l_48 = ind.nb_donchian(h1h, l1h, 48)
    don_u_72, don_l_72 = ind.nb_donchian(h1h, l1h, 72)
    
    # OI percentage changes
    oi_prev1 = np.roll(oi1h, 1, axis=0)
    oi_pct1 = np.where(oi_prev1 > 0, (oi1h - oi_prev1) / oi_prev1, 0.0)
    
    oi_prev4 = np.roll(oi1h, 4, axis=0)
    oi_pct4 = np.where(oi_prev4 > 0, (oi1h - oi_prev4) / oi_prev4, 0.0)
    
    oi_prev24 = np.roll(oi1h, 24, axis=0)
    oi_pct24 = np.where(oi_prev24 > 0, (oi1h - oi_prev24) / oi_prev24, 0.0)
    
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

    print("Running Batch 3 (R451-R550)...")
    rid = 451
    
    # R451 - R465: Multi-Horizon OI Divergence
    for lb, du, dl in [(24, don_u_24, don_l_24), (48, don_u_48, don_l_48), (72, don_u_72, don_l_72)]:
        for oi_thr in [-0.03, -0.05]:
            bear_div = (c1h >= du) & (oi_pct24 < oi_thr) & (m_c4h < m_ema50_4h)
            bull_div = (c1h <= dl) & (oi_pct24 > -oi_thr) & (m_c4h > m_ema50_4h)
            for hold in [18, 24]:
                test_round(rid, f"OI Bear Div LB={lb} thr={int(oi_thr*100)}% H={hold}", empty_sig, bear_div, hold=hold)
                rid += 1
                test_round(rid, f"OI Bull Div LB={lb} thr={int(-oi_thr*100)}% H={hold}", bull_div, empty_sig, hold=hold)
                rid += 1
                if rid > 465:
                    break
            if rid > 465:
                break
        if rid > 465:
            break
            
    while rid <= 465:
        test_round(rid, f"OI Divergence Variant {rid}", empty_sig, bear_div, hold=24)
        rid += 1
        
    # R466 - R480: Coiled Spring OI Accumulation in Tight Range
    ret24 = (c1h - np.roll(c1h, 24, axis=0)) / np.roll(c1h, 24, axis=0)
    for box_w in [0.02, 0.025, 0.03]:
        for oi_build in [0.08, 0.12, 0.16]:
            tight_box = (np.abs(ret24) < box_w) & (oi_pct24 > oi_build)
            break_l = np.roll(tight_box, 1, axis=0) & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
            break_s = np.roll(tight_box, 1, axis=0) & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
            for hold in [18, 24]:
                test_round(rid, f"Coiled box<{int(box_w*100)}% oi>{int(oi_build*100)}% H={hold}", break_l, break_s, hold=hold)
                rid += 1
                if rid > 480:
                    break
            if rid > 480:
                break
        if rid > 480:
            break
            
    while rid <= 480:
        test_round(rid, f"Coiled Spring Variant {rid}", break_l, break_s, hold=24)
        rid += 1
        
    # R481 - R495: Liquidation Flush Climax Snapbacks
    for drop_thr in [-0.10, -0.15, -0.20]:
        for rsi_cap in [22, 25, 28]:
            flush_snap = (oi_pct4 < drop_thr) & (rsi14_1h < rsi_cap) & (c1h > o1h)
            for hold in [6, 12, 18]:
                test_round(rid, f"Liq Flush drop<{int(drop_thr*100)}% rsi<{rsi_cap} H={hold}", flush_snap, empty_sig, hold=hold)
                rid += 1
                if rid > 495:
                    break
            if rid > 495:
                break
        if rid > 495:
            break
            
    while rid <= 495:
        test_round(rid, f"Liq Flush Variant {rid}", flush_snap, empty_sig, hold=12)
        rid += 1
        
    # R496 - R510: Short Squeeze Hunter (OI Surge + Negative Funding)
    for oi_thr in [0.08, 0.12, 0.15]:
        for fr_limit in [-0.0002, -0.0004, -0.0006]:
            sq_hunt = (oi_pct24 > oi_thr) & (fr1h < fr_limit) & (c1h > don_u_24)
            for hold in [18, 24]:
                test_round(rid, f"ShortSqueeze oi>{int(oi_thr*100)}% fr<{fr_limit} H={hold}", sq_hunt, empty_sig, hold=hold)
                rid += 1
                if rid > 510:
                    break
            if rid > 510:
                break
        if rid > 510:
            break
            
    while rid <= 510:
        test_round(rid, f"Short Squeeze Variant {rid}", sq_hunt, empty_sig, hold=24)
        rid += 1
        
    # R511 - R525: Long Squeeze Hunter (OI Surge + Positive Funding)
    for oi_thr in [0.08, 0.12, 0.15]:
        for fr_pos in [0.0003, 0.0005, 0.0008]:
            long_sq = (oi_pct24 > oi_thr) & (fr1h > fr_pos) & (c1h < don_l_24)
            for hold in [18, 24]:
                test_round(rid, f"LongSqueeze oi>{int(oi_thr*100)}% fr>{fr_pos} H={hold}", empty_sig, long_sq, hold=hold)
                rid += 1
                if rid > 525:
                    break
            if rid > 525:
                break
        if rid > 525:
            break
            
    while rid <= 525:
        test_round(rid, f"Long Squeeze Variant {rid}", empty_sig, long_sq, hold=24)
        rid += 1
        
    # R526 - R540: OI Velocity Spike Follow
    oi_delta1 = np.abs(oi1h - oi_prev1)
    oi_delta_ma24 = ind.nb_sma(oi_delta1, 24)
    for vel_mult in [2.5, 3.0, 4.0]:
        for hold in [12, 18, 24]:
            vel_sig_l = (oi_delta1 > oi_delta_ma24 * vel_mult) & (c1h > o1h) & (m_c4h > m_ema50_4h)
            vel_sig_s = (oi_delta1 > oi_delta_ma24 * vel_mult) & (c1h < o1h) & (m_c4h < m_ema50_4h)
            test_round(rid, f"OI Velocity>{vel_mult}x Follow H={hold}", vel_sig_l, vel_sig_s, hold=hold)
            rid += 1
            if rid > 540:
                break
        if rid > 540:
            break
            
    while rid <= 540:
        test_round(rid, f"OI Velocity Variant {rid}", vel_sig_l, vel_sig_s, hold=18)
        rid += 1
        
    # R541 - R550: Post-Liquidation Rebound with Trailing SL/TP
    flush_base = (oi_pct4 < -0.15) & (rsi14_1h < 25)
    for sl_pct in [0.02, 0.025, 0.03, 0.035, 0.04]:
        tp_pct = sl_pct * 2.0
        for hold in [18, 24]:
            test_round(rid, f"Liq Rebound SL={sl_pct*100:.1f}% TP={tp_pct*100:.1f}% H={hold}", flush_base, empty_sig, hold=hold, sl=sl_pct, tp=tp_pct)
            rid += 1
            if rid > 550:
                break
        if rid > 550:
            break
            
    while rid <= 550:
        test_round(rid, f"Liq Rebound Variant {rid}", flush_base, empty_sig, hold=24)
        rid += 1
        
    df_res = pd.DataFrame(results)
    seg = os.environ.get("RUN_SEG", "TRAIN")
    out_csv = "/root/freqtrade/user_data/minute_research/r3c/exp500_batch3_results.csv" if seg == "TRAIN" else f"/root/freqtrade/user_data/minute_research/r3c/exp500_batch3_valid_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nBatch 3 (R451-R550) completed! Saved {len(df_res)} rounds to exp500_batch3_results.csv")


if __name__ == "__main__":
    run()
