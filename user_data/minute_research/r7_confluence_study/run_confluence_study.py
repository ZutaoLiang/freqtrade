"""Systematic Confluence & Filter Stacking Study (r7_confluence_study).

Core Question Investigated:
"If you combine strategies and make trade entry conditions stricter, does it bring better win rate and returns?"

Experimental Architecture:
- Evaluates 5 discrete levels of strictness across 2 fundamental alpha themes:
  * Theme A: Momentum / Volatility Breakout Confluence (Raw Breakout -> Squeeze -> Taker Flow -> Macro BTC Gate -> Volume Explosion)
  * Theme B: Liquidation / Mean-Reversion Confluence (Raw Drop -> OI Flushout -> RSI Panic -> Micro-Absorption -> Funding Stress)
  * Theme C: Hybrid Structural Regime Combination (Switching between Squeeze and OI Flushout)
- Evaluates all levels across Scheme-C TRAIN, VALID-C, and HOLDOUT.
- Measures:
  1. Trade Count (N) - liquidity & opportunity frequency
  2. Win Rate (Win%) - does strictness improve precision?
  3. Mean Net Return (mean_bp) - does quality per trade increase?
  4. Profit Factor (PF) - ratio of gross wins to gross losses
  5. Daily Clustered t-stat (t) - statistical reliability (penalized by low N)
  6. Total Cumulative Net PnL (tot_pnl_bp) - portfolio growth (N * mean_bp)
"""
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r7_confluence_study")
from mtf_engine import MTFPanels, eval_signals, calc_safe_ret
import indicators_mtf as ind

print("=" * 80)
print("Starting Confluence & Filter Stacking Systematic Study")
print("=" * 80)
t0 = time.time()

# 1. Load Panels
print("Loading Panels...")
p1h = MTFPanels("1h")
p4h = MTFPanels("4h")
p1d = MTFPanels("1d")
btc_idx = p1h.symbols.index("BTCUSDT")

# Base Arrays
c1h, o1h, h1h, l1h, v1h = (np.array(x) for x in (p1h.close, p1h.open, p1h.high, p1h.low, p1h.volume))
tbv1h = np.array(p1h.taker_buy) if p1h.taker_buy is not None else v1h * 0.5
taker_ratio_1h = np.where(v1h > 0, tbv1h / (v1h + 1e-12), 0.5)

c4h, h4h, l4h = (np.array(x) for x in (p4h.close, p4h.high, p4h.low))
c1d = np.array(p1d.close)

oi1h = np.array(p1h.oi) if p1h.oi is not None else np.zeros_like(c1h)
fr1h = np.array(p1h.funding_rate) if p1h.funding_rate is not None else np.zeros_like(c1h)

# Precompute Base Indicators
print("Precomputing indicators for layering...")
# 1. Channels
_, bb_u_1h, bb_l_1h = ind.nb_bollinger(c1h, 20, 2.0)
_, kelt_u_1h, kelt_l_1h = ind.nb_keltner(h1h, l1h, c1h, 20, 1.5)
sq_1h = (bb_u_1h < kelt_u_1h) & (bb_l_1h > kelt_l_1h)

_, bb_u_4h, bb_l_4h = ind.nb_bollinger(c4h, 20, 2.0)
_, kelt_u_4h, kelt_l_4h = ind.nb_keltner(h4h, l4h, c4h, 20, 1.5)
sq_4h = (bb_u_4h < kelt_u_4h) & (bb_l_4h > kelt_l_4h)
m_sq_4h = p1h.map_htf(sq_4h, "4h")

# Shifts for causal alignment
sq_1h_prior = np.roll(sq_1h, 1, axis=0)
sq_4h_prior = np.roll(m_sq_4h, 1, axis=0)
dual_sq_prior = sq_4h_prior & sq_1h_prior

# 2. Volume
vol_ma24 = ind.nb_sma(v1h, 24)
vol_surge_15 = v1h > (1.5 * vol_ma24)
vol_surge_25 = v1h > (2.5 * vol_ma24)

# 3. Macro BTC
btc_c1d = c1d[:, btc_idx:btc_idx+1]
btc_sma50_1d = ind.nb_sma(btc_c1d, 50)
btc_bull_1d = p1h.map_htf(btc_c1d > btc_sma50_1d, "1d")

btc_c4h = c4h[:, btc_idx:btc_idx+1]
btc_ema50_4h = ind.nb_ema(btc_c4h, 50)
btc_bull_4h = p1h.map_htf(btc_c4h > btc_ema50_4h, "4h")

btc_macro_bull = btc_bull_1d & btc_bull_4h
btc_macro_bear = (~btc_bull_1d) & (~btc_bull_4h)

# 4. Causal 4H RS
ret_4h_safe = calc_safe_ret(c4h)
rs_4h = ret_4h_safe - ret_4h_safe[:, btc_idx:btc_idx+1]
m_rs_4h = p1h.map_htf(rs_4h, "4h")
rs_lead_20 = np.roll(m_rs_4h > 0.02, 1, axis=0)
rs_lag_20 = np.roll(m_rs_4h < -0.02, 1, axis=0)

# 5. Liquidation & OI Indicators
oi_12h_ago = np.roll(oi1h, 12, axis=0)
oi_ret_12h = (oi1h - oi_12h_ago) / np.maximum(oi_12h_ago, 1e-12)
c_12h_ago = np.roll(c1h, 12, axis=0)
c_ret_12h = (c1h - c_12h_ago) / np.maximum(c_12h_ago, 1e-12)
rsi14_1h = ind.nb_rsi(c1h, 14)

print("Indicators ready. Testing Confluence Levels across splits...\n")

records = []

def run_test(theme, level, name, desc, sig_l, sig_s, hold=18, sl=0.07, tp=0.14):
    tr = eval_signals(p1h, sig_l, sig_s, hold_bars=hold, sl_pct=sl, tp_pct=tp, seg="TRAIN")
    va = eval_signals(p1h, sig_l, sig_s, hold_bars=hold, sl_pct=sl, tp_pct=tp, seg="VALID")
    ho = eval_signals(p1h, sig_l, sig_s, hold_bars=hold, sl_pct=sl, tp_pct=tp, seg="HOLDOUT")
    
    rec = {
        "theme": theme,
        "level": level,
        "name": name,
        "desc": desc,
        # Train
        "tr_n": tr["n_trades"], "tr_win": round(tr["win_rate"], 1),
        "tr_mean_bp": round(tr["mean_bp"], 1), "tr_pf": round(tr["pf"], 2),
        "tr_t": round(tr["t_stat"], 2), "tr_tot_bp": round(tr["n_trades"] * tr["mean_bp"], 0),
        # Valid
        "va_n": va["n_trades"], "va_win": round(va["win_rate"], 1),
        "va_mean_bp": round(va["mean_bp"], 1), "va_pf": round(va["pf"], 2),
        "va_t": round(va["t_stat"], 2), "va_tot_bp": round(va["n_trades"] * va["mean_bp"], 0),
        # Holdout
        "ho_n": ho["n_trades"], "ho_win": round(ho["win_rate"], 1),
        "ho_mean_bp": round(ho["mean_bp"], 1), "ho_pf": round(ho["pf"], 2),
        "ho_t": round(ho["t_stat"], 2), "ho_tot_bp": round(ho["n_trades"] * ho["mean_bp"], 0),
    }
    records.append(rec)
    print(f"L{level} | {theme:10s} | {name:28s} | TR: n={tr['n_trades']:4d}, win={tr['win_rate']:4.1f}%, pf={tr['pf']:4.2f} | VA: n={va['n_trades']:4d}, win={va['win_rate']:4.1f}%, pf={va['pf']:4.2f} | HO: n={ho['n_trades']:4d}, win={ho['win_rate']:4.1f}%, pf={ho['pf']:4.2f}")


# =====================================================================
# THEME A: MOMENTUM / BREAKOUT CONFLUENCE (5 Levels)
# =====================================================================
print("--- THEME A: Momentum / Breakout Confluence Stacking ---")

# Level 1: Raw 1H Keltner Breakout (No filters)
l1_l = (c1h > kelt_u_1h)
l1_s = (c1h < kelt_l_1h)
run_test("Breakout", 1, "Raw_Keltner_Breakout", "1H Close > Keltner Upper / < Lower", l1_l, l1_s, hold=18, sl=0.07, tp=0.14)

# Level 2A: Breakout + 1H Squeeze
l2a_l = l1_l & sq_1h_prior
l2a_s = l1_s & sq_1h_prior
run_test("Breakout", 2, "Breakout_+_1HSqueeze", "Raw Breakout + 1H Volatility Compression", l2a_l, l2a_s, hold=18, sl=0.07, tp=0.14)

# Level 2B: Breakout + Dual Squeeze (4H & 1H)
l2b_l = l1_l & dual_sq_prior
l2b_s = l1_s & dual_sq_prior
run_test("Breakout", 2, "Breakout_+_DualSqueeze", "Raw Breakout + 4H & 1H Dual Squeeze", l2b_l, l2b_s, hold=18, sl=0.07, tp=0.14)

# Level 3A: Dual Squeeze + Moderate Taker Flow (>55%)
l3a_l = l2b_l & (taker_ratio_1h > 0.55)
l3a_s = l2b_s & (taker_ratio_1h < 0.45)
run_test("Breakout", 3, "DualSq_+_Flow55", "Dual Squeeze + Taker Buy > 55%", l3a_l, l3a_s, hold=18, sl=0.07, tp=0.14)

# Level 3B: Dual Squeeze + Aggressive Taker Flow (>65%)
l3b_l = l2b_l & (taker_ratio_1h > 0.65)
l3b_s = l2b_s & (taker_ratio_1h < 0.35)
run_test("Breakout", 3, "DualSq_+_Flow65", "Dual Squeeze + Taker Buy > 65%", l3b_l, l3b_s, hold=18, sl=0.07, tp=0.14)

# Level 4A: Dual Squeeze + Flow60 + Macro BTC Gate
l4a_l = l2b_l & (taker_ratio_1h > 0.60) & btc_macro_bull
l4a_s = l2b_s & (taker_ratio_1h < 0.40) & btc_macro_bear
run_test("Breakout", 4, "DualSq_+_Flow60_+_BTCGate", "Dual Squeeze + Flow60 + BTC 1D&4H Direction", l4a_l, l4a_s, hold=18, sl=0.07, tp=0.14)

# Level 4B: Dual Squeeze + Flow60 + Volume Surge (>1.5x)
l4b_l = l2b_l & (taker_ratio_1h > 0.60) & vol_surge_15
l4b_s = l2b_s & (taker_ratio_1h < 0.40) & vol_surge_15
run_test("Breakout", 4, "DualSq_+_Flow60_+_Vol1.5x", "Dual Squeeze + Flow60 + 1H Volume > 1.5x MA", l4b_l, l4b_s, hold=18, sl=0.07, tp=0.14)

# Level 5A: Ultra Strict Stack (Dual Squeeze + Flow65 + BTC Gate + Vol 2.0x)
l5a_l = l2b_l & (taker_ratio_1h > 0.65) & btc_macro_bull & vol_surge_25
l5a_s = l2b_s & (taker_ratio_1h < 0.35) & btc_macro_bear & vol_surge_25
run_test("Breakout", 5, "Ultra_Strict_Breakout_Stack", "DualSq + Flow65 + BTCGate + Vol2.5x", l5a_l, l5a_s, hold=18, sl=0.07, tp=0.14)

# Level 5B: Confluence Stack with Causal 4H RS (Dual Squeeze + Flow60 + BTC Gate + RS Lead 2%)
l5b_l = l2b_l & (taker_ratio_1h > 0.60) & btc_macro_bull & rs_lead_20
l5b_s = l2b_s & (taker_ratio_1h < 0.40) & btc_macro_bear & rs_lag_20
run_test("Breakout", 5, "Strict_Stack_+_4HRSLead", "DualSq + Flow60 + BTCGate + 4H RS > 2%", l5b_l, l5b_s, hold=18, sl=0.07, tp=0.14)


# =====================================================================
# THEME B: LIQUIDATION / REVERSAL CONFLUENCE (5 Levels)
# =====================================================================
print("\n--- THEME B: Liquidation / Reversal Confluence Stacking ---")

# Level 1: Raw 12h Price Drop > 8% (Unconstrained dip buying)
empty_s = np.zeros_like(c1h, dtype=bool)
l1_rev_l = (c_ret_12h < -0.08)
run_test("Reversal", 1, "Raw_Price_Drop_8%", "12h Price Drop > 8% alone", l1_rev_l, empty_s, hold=12, sl=0.06, tp=0.12)

# Level 2: Price Drop > 8% + Open Interest Collapse > 15% (Liquidation Flushout)
l2_rev_l = l1_rev_l & (oi_ret_12h < -0.15)
run_test("Reversal", 2, "Drop8%_+_OIFlush15%", "Price Drop 8% + OI Plunge 15% (Forced Liquidations)", l2_rev_l, empty_s, hold=12, sl=0.06, tp=0.12)

# Level 3: Drop8% + OIFlush15% + Technical Oversold (RSI < 30)
l3_rev_l = l2_rev_l & (rsi14_1h < 30)
run_test("Reversal", 3, "Flushout_+_RSI30", "Drop8% + OIFlush15% + 1H RSI < 30", l3_rev_l, empty_s, hold=12, sl=0.06, tp=0.12)

# Level 4: Drop8% + OIFlush15% + RSI < 30 + Green Reversal Candle + Taker Buy > 55%
l4_rev_l = l3_rev_l & (c1h > o1h) & (taker_ratio_1h > 0.55)
run_test("Reversal", 4, "Flushout_RSI30_+_Flow55", "Flushout + RSI30 + Green Bar + Taker > 55%", l4_rev_l, empty_s, hold=12, sl=0.06, tp=0.12)

# Level 5A: Ultra Strict Reversal Stack (Flushout + RSI < 25 + Green Bar + Taker > 65%)
l5a_rev_l = l2_rev_l & (rsi14_1h < 25) & (c1h > o1h) & (taker_ratio_1h > 0.65)
run_test("Reversal", 5, "Ultra_Strict_Flushout", "Flushout + RSI25 + Green + Taker > 65%", l5a_rev_l, empty_s, hold=12, sl=0.06, tp=0.12)

# Level 5B: Flushout Stack + Negative Funding Stress (fr < -0.0003)
l5b_rev_l = l4_rev_l & (fr1h < -0.0003)
run_test("Reversal", 5, "Flushout_+_NegFunding", "Flushout Stack + Funding Rate < -0.03%", l5b_rev_l, empty_s, hold=12, sl=0.06, tp=0.12)


# =====================================================================
# THEME C: HYBRID REGIME COMBINATION
# =====================================================================
print("\n--- THEME C: Hybrid Regime Combinations ---")

# System 1: Dual Squeeze Flow60 Long/Short (Trend/Expansion)
# System 2: OI Flushout Long Only (Dislocation/Reversal)
# Hybrid: Long when EITHER Squeeze Breakout triggers in BTC Bull, OR Flushout triggers after panic
hyb_l = l4a_l | l4_rev_l
hyb_s = l4a_s
run_test("Hybrid", 4, "Hybrid_Squeeze_Flushout", "Squeeze in Trend + Flushout in Panic", hyb_l, hyb_s, hold=18, sl=0.07, tp=0.14)

# Save Master Confluence Study CSV
df_study = pd.DataFrame(records)
csv_file = "/root/freqtrade/user_data/minute_research/r7_confluence_study/confluence_study_results.csv"
df_study.to_csv(csv_file, index=False)
print(f"\nConfluence Study completed in {time.time() - t0:.2f}s! Saved to {csv_file}")

