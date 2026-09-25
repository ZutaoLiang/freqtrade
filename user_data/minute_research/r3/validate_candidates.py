"""Strict single-read validation on VALID segment (2025-10-01 to 2026-03-01).
Evaluates R127 and R231 against SKILL §4 criteria A-G.
"""
import sys
import numpy as np
import pandas as pd

sys.path.append("/root/freqtrade/user_data/minute_research/r3")
from engine200 import PanelData, evaluate_strategy
import indicators as ind


def validate_all():
    print("Loading panels for validation on VALID...")
    p1h = PanelData("1h")
    p4h = PanelData("4h")
    p1d = PanelData("1d")
    
    c1h = np.array(p1h.close)
    o1h = np.array(p1h.open)
    h1h = np.array(p1h.high)
    l1h = np.array(p1h.low)
    v1h = np.array(p1h.volume)
    tbv1h = np.array(p1h.taker_buy_volume) if p1h.taker_buy_volume is not None else v1h * 0.5
    
    c4h = np.array(p4h.close)
    h4h = np.array(p4h.high)
    l4h = np.array(p4h.low)
    
    # 4h indicators mapped causally
    ema50_4h = ind.nb_ema(c4h, 50)
    m_ema50_4h = p1h.map_htf_to_ltf(ema50_4h, "4h")
    m_c4h = p1h.map_htf_to_ltf(c4h, "4h")
    
    # BTC indicators
    btc_idx = p1h.symbols.index("BTCUSDT") if "BTCUSDT" in p1h.symbols else p1h.u162_indices[0]
    btc_m_c4h = m_c4h[:, btc_idx:btc_idx+1]
    btc_m_ema50_4h = m_ema50_4h[:, btc_idx:btc_idx+1]
    btc_bull_4h = btc_m_c4h > btc_m_ema50_4h
    
    # 1h core
    atr14_1h = ind.nb_atr(h1h, l1h, c1h, 14)
    atr_ma24 = ind.nb_sma(atr14_1h, 24)
    taker_ratio = np.where(v1h > 0, tbv1h / v1h, 0.5)
    
    # Keltner & BB for R127
    kelt_4h_mid, kelt_4h_u, kelt_4h_l = ind.nb_keltner(h4h, l4h, c4h, 20, 1.5)
    _, bb_4h_u, bb_4h_l, _, _ = ind.nb_bollinger(c4h, 20, 2.0)
    sq_4h = (bb_4h_u < kelt_4h_u) & (bb_4h_l > kelt_4h_l)
    m_sq_4h = p1h.map_htf_to_ltf(sq_4h.astype(np.float64), "4h")
    
    kelt_1h_mid, kelt_1h_u, kelt_1h_l = ind.nb_keltner(h1h, l1h, c1h, 20, 1.5)
    _, bb_1h_u, bb_1h_l, _, _ = ind.nb_bollinger(c1h, 20, 2.0)
    bb_inside_keltner = (bb_1h_u < kelt_1h_u) & (bb_1h_l > kelt_1h_l)
    
    print("\n" + "="*80)
    print("AUDITING R127: Dual 4h/1h MTF Squeeze Breakout")
    print("="*80)
    dual_sq = (m_sq_4h == 1.0) & bb_inside_keltner
    r127_sig_l = np.roll(dual_sq, 1, axis=0) & (c1h > kelt_1h_u)
    r127_sig_s = np.roll(dual_sq, 1, axis=0) & (c1h < kelt_1h_l)
    
    res127_train = evaluate_strategy(p1h, r127_sig_l, r127_sig_s, hold_bars=24, seg="TRAIN")
    res127_valid = evaluate_strategy(p1h, r127_sig_l, r127_sig_s, hold_bars=24, seg="VALID")
    
    print(f"TRAIN: n={res127_train['n']}, mean={res127_train['mean_bp']}bp, pf={res127_train['pf']}, day_t={res127_train['day_t']}, max_coin={res127_train['max_coin_share']*100:.1f}%, max_day={res127_train['max_day_share']*100:.1f}%")
    print(f"VALID: n={res127_valid['n']}, mean={res127_valid['mean_bp']}bp, pf={res127_valid['pf']}, day_t={res127_valid['day_t']}, max_coin={res127_valid['max_coin_share']*100:.1f}%, max_day={res127_valid['max_day_share']*100:.1f}%")

    print("\n" + "="*80)
    print("AUDITING R231: Hybrid Taker Surge + ATR Expansion + BTC 4h Trend")
    print("="*80)
    atr_exp = atr14_1h > atr_ma24 * 1.3
    r231_sig_l = (taker_ratio > 0.65) & atr_exp & (m_c4h > m_ema50_4h) & btc_bull_4h
    r231_sig_s = (taker_ratio < 0.35) & atr_exp & (m_c4h < m_ema50_4h) & (~btc_bull_4h)
    
    res231_train = evaluate_strategy(p1h, r231_sig_l, r231_sig_s, hold_bars=18, seg="TRAIN")
    res231_valid = evaluate_strategy(p1h, r231_sig_l, r231_sig_s, hold_bars=18, seg="VALID")
    
    print(f"TRAIN: n={res231_train['n']}, mean={res231_train['mean_bp']}bp, pf={res231_train['pf']}, day_t={res231_train['day_t']}, max_coin={res231_train['max_coin_share']*100:.1f}%, max_day={res231_train['max_day_share']*100:.1f}%")
    print(f"VALID: n={res231_valid['n']}, mean={res231_valid['mean_bp']}bp, pf={res231_valid['pf']}, day_t={res231_valid['day_t']}, max_coin={res231_valid['max_coin_share']*100:.1f}%, max_day={res231_valid['max_day_share']*100:.1f}%")


if __name__ == "__main__":
    validate_all()
