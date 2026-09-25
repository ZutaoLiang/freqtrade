"""500-Round Expansion - Batch 2: R351 to R450.
Focus: Volume, Taker Flow, Trade Count, Institutional Block Flow & Microstructure.
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
    print("Loading panels for Batch 2 (R351-R450)...")
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
    tbv1h = np.array(p1h.taker_buy_volume) if p1h.taker_buy_volume is not None else v1h * 0.5
    cnt1h = np.array(p1h.count) if p1h.count is not None else np.ones_like(v1h)
    tt_ls1h = np.array(p1h.toptrader_ls) if p1h.toptrader_ls is not None else np.ones_like(v1h)
    
    c4h = np.array(p4h.close)
    
    T, N = c1h.shape
    c_prev = np.roll(c1h, 1, axis=0)
    empty_sig = np.zeros((T, N), dtype=bool)
    
    print("Pre-computing indicators for Batch 2...")
    ema20_1h = ind.nb_ema(c1h, 20)
    ema50_1h = ind.nb_ema(c1h, 50)
    ema50_4h = ind.nb_ema(c4h, 50)
    m_ema50_4h = p1h.map_htf_to_ltf(ema50_4h, "4h")
    m_c4h = p1h.map_htf_to_ltf(c4h, "4h")
    
    atr14_1h = ind.nb_atr(h1h, l1h, c1h, 14)
    rsi14_1h = ind.nb_rsi(c1h, 14)
    don_u_24, don_l_24 = ind.nb_donchian(h1h, l1h, 24)
    
    v_ma24 = ind.nb_sma(v1h, 24)
    v_ma168 = ind.nb_sma(v1h, 168)
    qv_ma24 = ind.nb_sma(qv1h, 24)
    cnt_ma24 = ind.nb_sma(cnt1h, 24)
    
    taker_ratio = np.where(v1h > 0, tbv1h / v1h, 0.5)
    avg_trade_size = np.where(cnt1h > 0, v1h / cnt1h, 0.0)
    avg_size_ma24 = ind.nb_sma(avg_trade_size, 24)
    
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

    print("Running Batch 2 (R351-R450)...")
    rid = 351
    
    # R351 - R365: Multi-bar Taker Flow Persistence
    for bars_cnt in [2, 3, 4]:
        for tr_thr in [0.58, 0.62, 0.66]:
            tb_persist = np.ones((T, N), dtype=bool)
            ts_persist = np.ones((T, N), dtype=bool)
            for k in range(bars_cnt):
                tb_persist = tb_persist & (np.roll(taker_ratio, k, axis=0) > tr_thr)
                ts_persist = ts_persist & (np.roll(taker_ratio, k, axis=0) < (1.0 - tr_thr))
            for hold in [18, 24]:
                sig_l = tb_persist & (m_c4h > m_ema50_4h)
                sig_s = ts_persist & (m_c4h < m_ema50_4h)
                test_round(rid, f"Taker Persist {bars_cnt}b >{int(tr_thr*100)}% H={hold}", sig_l, sig_s, hold=hold)
                rid += 1
                if rid > 365:
                    break
            if rid > 365:
                break
        if rid > 365:
            break
            
    while rid <= 365:
        test_round(rid, f"Taker Persistence Variant {rid}", sig_l, sig_s, hold=18)
        rid += 1
        
    # R366 - R380: Taker Volume Surge into Resistance / Support
    for qv_mult in [2.0, 3.0, 4.0]:
        for hold in [12, 18, 24]:
            near_high = (c1h >= don_u_24 * 0.99) & (c1h <= don_u_24 * 1.01)
            near_low = (c1h <= don_l_24 * 1.01) & (c1h >= don_l_24 * 0.99)
            sig_l = near_high & (qv1h > qv_ma24 * qv_mult) & (taker_ratio > 0.60) & (m_c4h > m_ema50_4h)
            sig_s = near_low & (qv1h > qv_ma24 * qv_mult) & (taker_ratio < 0.40) & (m_c4h < m_ema50_4h)
            test_round(rid, f"Taker Surge QV>{qv_mult}x at Level H={hold}", sig_l, sig_s, hold=hold)
            rid += 1
            if rid > 380:
                break
        if rid > 380:
            break
            
    while rid <= 380:
        test_round(rid, f"Level Taker Surge Variant {rid}", sig_l, sig_s, hold=24)
        rid += 1
        
    # R381 - R395: Institutional Block Flow (High Volume + Low Count)
    for qv_mult in [2.0, 2.5, 3.0]:
        for cnt_fact in [0.7, 0.8, 0.9]:
            inst_flow = (qv1h > qv_ma24 * qv_mult) & (cnt1h < cnt_ma24 * cnt_fact)
            for hold in [18, 24]:
                sig_l = inst_flow & (c1h > o1h) & (m_c4h > m_ema50_4h)
                sig_s = inst_flow & (c1h < o1h) & (m_c4h < m_ema50_4h)
                test_round(rid, f"Inst Block QV>{qv_mult}x Cnt<{cnt_fact}x H={hold}", sig_l, sig_s, hold=hold)
                rid += 1
                if rid > 395:
                    break
            if rid > 395:
                break
        if rid > 395:
            break
            
    while rid <= 395:
        test_round(rid, f"Inst Block Flow Variant {rid}", sig_l, sig_s, hold=18)
        rid += 1
        
    # R396 - R410: Retail Fragmentation FOMO Climax Fade
    for cnt_mult in [2.5, 3.0, 4.0]:
        for size_fact in [0.4, 0.5, 0.6]:
            retail_fomo = (cnt1h > cnt_ma24 * cnt_mult) & (avg_trade_size < avg_size_ma24 * size_fact)
            ret24 = (c1h - np.roll(c1h, 24, axis=0)) / np.roll(c1h, 24, axis=0)
            sig_s = retail_fomo & (ret24 > 0.04) & (rsi14_1h > 65)
            sig_l = retail_fomo & (ret24 < -0.04) & (rsi14_1h < 35)
            for hold in [12, 18]:
                test_round(rid, f"Retail FOMO Fade Cnt>{cnt_mult}x H={hold}", sig_l, sig_s, hold=hold)
                rid += 1
                if rid > 410:
                    break
            if rid > 410:
                break
        if rid > 410:
            break
            
    while rid <= 410:
        test_round(rid, f"Retail Fade Variant {rid}", sig_l, sig_s, hold=12)
        rid += 1
        
    # R411 - R425: Taker Flow Absorption / Divergence
    ret4 = (c1h - np.roll(c1h, 4, axis=0)) / np.roll(c1h, 4, axis=0)
    for dip_thr in [-0.02, -0.03, -0.04]:
        for tr_thr in [0.60, 0.65, 0.70]:
            absorb_l = (ret4 < dip_thr) & (taker_ratio > tr_thr) & (c1h > o1h) & (m_c4h > m_ema50_4h)
            absorb_s = (ret4 > -dip_thr) & (taker_ratio < (1.0 - tr_thr)) & (c1h < o1h) & (m_c4h < m_ema50_4h)
            for hold in [12, 18]:
                test_round(rid, f"Taker Absorb dip<{int(dip_thr*100)}% tr>{int(tr_thr*100)}% H={hold}", absorb_l, absorb_s, hold=hold)
                rid += 1
                if rid > 425:
                    break
            if rid > 425:
                break
        if rid > 425:
            break
            
    while rid <= 425:
        test_round(rid, f"Taker Absorb Variant {rid}", sig_l, sig_s, hold=18)
        rid += 1
        
    # R426 - R440: RVOL Stealth Explosion from Low-Volume Base
    low_base = np.roll(v_ma24, 1, axis=0) < np.roll(v_ma168, 1, axis=0) * 0.75
    for rvol_mult in [3.0, 4.0, 5.0]:
        rvol_surge = low_base & (v1h > v_ma24 * rvol_mult)
        for hold in [18, 24, 36]:
            sig_l = rvol_surge & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
            sig_s = rvol_surge & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
            test_round(rid, f"Stealth RVOL>{rvol_mult}x Break H={hold}", sig_l, sig_s, hold=hold)
            rid += 1
            if rid > 440:
                break
        if rid > 440:
            break
            
    while rid <= 440:
        test_round(rid, f"Stealth RVOL Variant {rid}", sig_l, sig_s, hold=24)
        rid += 1
        
    # R441 - R450: Volume Climax Absorption with Dynamic Trailing Exits
    lower_wick = np.minimum(o1h, c1h) - l1h
    upper_wick = h1h - np.maximum(o1h, c1h)
    candle_body = np.maximum(np.abs(c1h - o1h), 1e-8)
    pin_l = (v1h > v_ma24 * 3.0) & (lower_wick > candle_body * 2.0) & (rsi14_1h < 30)
    pin_s = (v1h > v_ma24 * 3.0) & (upper_wick > candle_body * 2.0) & (rsi14_1h > 70)
    for sl_pct in [0.02, 0.025, 0.03, 0.035, 0.04]:
        tp_pct = sl_pct * 2.0
        for hold in [18, 24]:
            test_round(rid, f"Vol Climax Pin SL={sl_pct*100:.1f}% TP={tp_pct*100:.1f}% H={hold}", pin_l, pin_s, hold=hold, sl=sl_pct, tp=tp_pct)
            rid += 1
            if rid > 450:
                break
        if rid > 450:
            break
            
    while rid <= 450:
        test_round(rid, f"Vol Climax Variant {rid}", pin_l, pin_s, hold=24)
        rid += 1
        
    df_res = pd.DataFrame(results)
    seg = os.environ.get("RUN_SEG", "TRAIN")
    out_csv = "/root/freqtrade/user_data/minute_research/r3c/exp500_batch2_results.csv" if seg == "TRAIN" else f"/root/freqtrade/user_data/minute_research/r3c/exp500_batch2_valid_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nBatch 2 (R351-R450) completed! Saved {len(df_res)} rounds to exp500_batch2_results.csv")


if __name__ == "__main__":
    run()
