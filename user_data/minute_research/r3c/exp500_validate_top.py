"""Single-read validation on VALID segment (2025-10-01 to 2026-03-01) for top 500-round finalists.
Evaluates Criteria B, C, D, E, F, G under SKILL §4.
"""
import sys
import numpy as np
import pandas as pd

sys.path.append("/root/freqtrade/user_data/minute_research/r3c")
from engine200 import PanelData, evaluate_strategy
import indicators as ind


def audit_candidate(p1h, sig_l, sig_s, name, hold, sl=0.0, tp=0.0):
    print("\n" + "="*80)
    print(f"AUDITING: {name}")
    print("="*80)
    
    res_tr = evaluate_strategy(p1h, sig_l, sig_s, hold_bars=hold, seg="TRAIN", sl_pct=sl, tp_pct=tp)
    res_va = evaluate_strategy(p1h, sig_l, sig_s, hold_bars=hold, seg="VALID", sl_pct=sl, tp_pct=tp)
    
    print(f"TRAIN: n={res_tr['n']:5d} | mean={res_tr['mean_bp']:6.1f}bp | pf={res_tr['pf']:5.2f} | day_t={res_tr['day_t']:5.2f} |/d={res_tr['per_day']:4.1f} | max_coin={res_tr['max_coin_share']*100:.1f}% | max_day={res_tr['max_day_share']*100:.1f}%")
    print(f"VALID: n={res_va['n']:5d} | mean={res_va['mean_bp']:6.1f}bp | pf={res_va['pf']:5.2f} | day_t={res_va['day_t']:5.2f} |/d={res_va['per_day']:4.1f} | max_coin={res_va['max_coin_share']*100:.1f}% | max_day={res_va['max_day_share']*100:.1f}%")
    
    status_b = res_va['n'] >= 30 or res_va['per_day'] >= 0.2
    status_c = res_va['day_t'] >= 1.5
    status_d = res_va['pf'] >= 1.15
    status_e = res_va['max_coin_share'] <= 0.30 and res_va['max_day_share'] <= 0.30
    
    print(f"VALID §4 Check: B(freq)={'PASS' if status_b else 'FAIL'} | C(t>=1.5)={'PASS' if status_c else 'FAIL'} | D(pf>=1.15)={'PASS' if status_d else 'FAIL'} | E(conc<=30%)={'PASS' if status_e else 'FAIL'}")
    return res_tr, res_va


def run_validations():
    print("Loading panels for strict validation on VALID...")
    p1h = PanelData("1h")
    p4h = PanelData("4h")
    p1d = PanelData("1d")
    
    c1h = np.array(p1h.close)
    o1h = np.array(p1h.open)
    h1h = np.array(p1h.high)
    l1h = np.array(p1h.low)
    v1h = np.array(p1h.volume)
    tbv1h = np.array(p1h.taker_buy_volume) if p1h.taker_buy_volume is not None else v1h * 0.5
    fr1h = np.array(p1h.funding_rate) if p1h.funding_rate is not None else np.zeros_like(c1h)
    
    c4h = np.array(p4h.close)
    h4h = np.array(p4h.high)
    l4h = np.array(p4h.low)
    
    T, N = c1h.shape
    empty_sig = np.zeros((T, N), dtype=bool)
    
    # 4h indicators
    ema50_4h = ind.nb_ema(c4h, 50)
    m_ema50_4h = p1h.map_htf_to_ltf(ema50_4h, "4h")
    m_c4h = p1h.map_htf_to_ltf(c4h, "4h")
    
    # BTC indicators
    btc_idx = p1h.symbols.index("BTCUSDT") if "BTCUSDT" in p1h.symbols else p1h.u162_indices[0]
    btc_m_c4h = m_c4h[:, btc_idx:btc_idx+1]
    btc_m_ema50_4h = m_ema50_4h[:, btc_idx:btc_idx+1]
    btc_bull_4h = btc_m_c4h > btc_m_ema50_4h
    
    # 1h core
    ema20_1h = ind.nb_ema(c1h, 20)
    don_u_24, don_l_24 = ind.nb_donchian(h1h, l1h, 24)
    taker_ratio = np.where(v1h > 0, tbv1h / v1h, 0.5)
    
    # Dual Squeeze
    _, bb_4h_u, bb_4h_l, _, _ = ind.nb_bollinger(c4h, 20, 2.0)
    _, kelt_4h_u_15, kelt_4h_l_15 = ind.nb_keltner(h4h, l4h, c4h, 20, 1.5)
    sq_4h_15 = (bb_4h_u < kelt_4h_u_15) & (bb_4h_l > kelt_4h_l_15)
    m_sq_4h_15 = p1h.map_htf_to_ltf(sq_4h_15.astype(np.float64), "4h")
    
    _, bb_1h_u, bb_1h_l, _, _ = ind.nb_bollinger(c1h, 20, 2.0)
    _, kelt_1h_u_15, kelt_1h_l_15 = ind.nb_keltner(h1h, l1h, c1h, 20, 1.5)
    sq_1h_15 = (bb_1h_u < kelt_1h_u_15) & (bb_1h_l > kelt_1h_l_15)
    dual_sq = (m_sq_4h_15 == 1.0) & sq_1h_15
    
    # 1. R273: Dual Squeeze + 4h Trend + BTC 4h Bull Gate (H=24)
    sig_l_273 = np.roll(dual_sq, 1, axis=0) & (c1h > kelt_1h_u_15) & (m_c4h > m_ema50_4h) & btc_bull_4h
    sig_s_273 = np.roll(dual_sq, 1, axis=0) & (c1h < kelt_1h_l_15) & (m_c4h < m_ema50_4h) & (~btc_bull_4h)
    audit_candidate(p1h, sig_l_273, sig_s_273, "R273 (Dual Squeeze + 4h Trend + BTC Gate)", hold=24)
    
    # 2. R748: Master Finalist SL=3.5% TP=7.0% (H=36)
    sig_l_748 = np.roll(dual_sq, 1, axis=0) & (c1h > kelt_1h_u_15) & (taker_ratio > 0.60) & btc_bull_4h
    sig_s_748 = np.roll(dual_sq, 1, axis=0) & (c1h < kelt_1h_l_15) & (taker_ratio < 0.40) & (~btc_bull_4h)
    audit_candidate(p1h, sig_l_748, sig_s_748, "R748 (Master Finalist SL=3.5% TP=7.0%)", hold=36, sl=0.035, tp=0.07)
    
    # 3. R562: Cumulative 48h FR Exhaustion > 0.4% (H=24)
    fr_cum48 = ind.nb_sma(fr1h, 48) * 48.0
    sig_s_562 = (fr_cum48 > 0.004) & (c1h < ema20_1h) & (m_c4h < m_ema50_4h)
    sig_l_562 = (fr_cum48 < -0.004) & (c1h > ema20_1h) & (m_c4h > m_ema50_4h)
    audit_candidate(p1h, sig_l_562, sig_s_562, "R562 (Cumulative 48h FR Exhaustion >0.4%)", hold=24)
    
    # 4. R617: Funding Rate Discount Breakout fr < -0.0001 (H=18)
    sig_l_617 = (c1h > don_u_24) & (fr1h < -0.0001) & (m_c4h > m_ema50_4h)
    sig_s_617 = (c1h < don_l_24) & (fr1h > 0.0001) & (m_c4h < m_ema50_4h)
    audit_candidate(p1h, sig_l_617, sig_s_617, "R617 (FR Discount Breakout fr<-0.01%)", hold=18)


if __name__ == "__main__":
    run_validations()
