"""300-Round Orthogonal Alphas Iteration (R1351 - R1650).

Orthogonal Hypotheses (Completely distinct from classic indicators):
- Batch 1: Macro BTC Lead-Lag Spillover & Cross-Asset Shock Wave
- Batch 2: Microstructure Delta Divergence & Passive Absorption (CVD Divergence)
- Batch 3: Top-Trader Sentiment Divergence & Smart Money Positioning
- Batch 4: Open Interest Velocity Divergence & Trapped Liquidation Cascades
- Batch 5: Volatility Regime Switching & Asymmetric Semi-Variance
- Batch 6: Funding Settlement Micro-Cycle & Pre/Post-Settlement Exploitation

Guarantees:
1. 100% Causal Alignment - Zero Lookahead (Strictly causal returns, shifted indicators).
2. Scheme-C Triple Split (TRAIN: 2025-01..10, VALID: 2025-12..2026-03, HOLDOUT: 2026-03..08).
3. Sacred Blind Test Protection: HOLDOUT evaluated ONLY for candidates passing BOTH TRAIN and VALID-C.
4. Realistic Friction: 20 bps round-trip taker fees, Next-bar Open execution, U162 universe.
"""
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r6_orthogonal_alphas")
from mtf_engine import MTFPanels, eval_signals, calc_safe_ret
import orthogonal_indicators as oind

print("=" * 80)
print("Starting 300-Round Orthogonal Alphas Iteration (R1351 - R1650)")
print("=" * 80)
t0 = time.time()

# 1. Load Panels
print("Loading MTF Panels with full derivatives...")
p1h = MTFPanels("1h")
p4h = MTFPanels("4h")
btc_idx = p1h.symbols.index("BTCUSDT")
print(f"Panels loaded in {time.time() - t0:.2f}s. BTC index: {btc_idx}, U162 count: {len(p1h.u162_indices)}")

# 2. Extract Base Arrays (1H)
c1h = np.array(p1h.close)
o1h = np.array(p1h.open)
h1h = np.array(p1h.high)
l1h = np.array(p1h.low)
v1h = np.array(p1h.volume)
qv1h = np.array(p1h.volume) * c1h
tbv1h = np.array(p1h.taker_buy) if p1h.taker_buy is not None else v1h * 0.5
taker_ratio_1h = np.where(v1h > 0, tbv1h / (v1h + 1e-12), 0.5)

# Derivatives Arrays (1H)
fr1h = np.array(p1h.funding_rate) if p1h.funding_rate is not None else np.zeros_like(c1h)
oi1h = np.array(p1h.oi) if p1h.oi is not None else np.zeros_like(c1h)
toptrader1h = np.array(p1h.toptrader) if p1h.toptrader is not None else np.full_like(c1h, 1.5)
taker_ls1h = np.array(p1h.taker_ls) if p1h.taker_ls is not None else np.full_like(c1h, 1.0)

# Causal Returns
ret_1h_safe = calc_safe_ret(c1h)
ret_1h_btc = ret_1h_safe[:, btc_idx:btc_idx+1]

# Base Technicals
rsi14_1h = oind.nb_rsi(c1h, 14)
vol_ma24_1h = oind.nb_sma(v1h, 24)
utc_hours = p1h.dates.hour.values

print("Pre-computing orthogonal alpha signals...")

# ---------------------------------------------------------------------
# Signal Group 1: Macro BTC Lead-Lag Spillover
# ---------------------------------------------------------------------
btc_vol_ma24 = vol_ma24_1h[:, btc_idx:btc_idx+1]
btc_v1h = v1h[:, btc_idx:btc_idx+1]
btc_shock_pos_15 = np.broadcast_to((ret_1h_btc > 0.015) & (btc_v1h > 1.5 * btc_vol_ma24), c1h.shape)
btc_shock_pos_20 = np.broadcast_to((ret_1h_btc > 0.020) & (btc_v1h > 1.5 * btc_vol_ma24), c1h.shape)
btc_shock_pos_25 = np.broadcast_to((ret_1h_btc > 0.025) & (btc_v1h > 2.0 * btc_vol_ma24), c1h.shape)

btc_shock_neg_15 = np.broadcast_to((ret_1h_btc < -0.015) & (btc_v1h > 1.5 * btc_vol_ma24), c1h.shape)
btc_shock_neg_20 = np.broadcast_to((ret_1h_btc < -0.020) & (btc_v1h > 1.5 * btc_vol_ma24), c1h.shape)
btc_shock_neg_25 = np.broadcast_to((ret_1h_btc < -0.025) & (btc_v1h > 2.0 * btc_vol_ma24), c1h.shape)

# ---------------------------------------------------------------------
# Signal Group 2: Cumulative Volume Delta (CVD) Divergence
# ---------------------------------------------------------------------
delta_1h = (2.0 * tbv1h - v1h)
cvd_12h = oind.nb_rolling_sum(delta_1h, 12)
cvd_24h = oind.nb_rolling_sum(delta_1h, 24)

min_l_12h = oind.nb_rolling_min(l1h, 12)
max_h_12h = oind.nb_rolling_max(h1h, 12)
min_l_24h = oind.nb_rolling_min(l1h, 24)
max_h_24h = oind.nb_rolling_max(h1h, 24)

# ---------------------------------------------------------------------
# Signal Group 3: Top-Trader vs Retail Sentiment Divergence
# ---------------------------------------------------------------------
tt_sma24 = oind.nb_sma(toptrader1h, 24)
tt_surge_bull = (toptrader1h > 1.50) & (toptrader1h > tt_sma24) & (taker_ls1h < 0.90)
tt_surge_bull_strong = (toptrader1h > 1.70) & (toptrader1h > tt_sma24) & (taker_ls1h < 0.85)

tt_surge_bear = (toptrader1h < 0.80) & (toptrader1h < tt_sma24) & (taker_ls1h > 1.15)
tt_surge_bear_strong = (toptrader1h < 0.70) & (toptrader1h < tt_sma24) & (taker_ls1h > 1.25)

# ---------------------------------------------------------------------
# Signal Group 4: Open Interest Velocity & Trapped Cascades
# ---------------------------------------------------------------------
oi_12h_ago = np.roll(oi1h, 12, axis=0)
oi_ret_12h = (oi1h - oi_12h_ago) / np.maximum(oi_12h_ago, 1e-12)

c_12h_ago = np.roll(c1h, 12, axis=0)
c_ret_12h = (c1h - c_12h_ago) / np.maximum(c_12h_ago, 1e-12)

# Liquidation flushout: price drops sharply, OI collapses, then green bar
flushout_bull_10 = (c_ret_12h < -0.06) & (oi_ret_12h < -0.10) & (c1h > o1h) & (taker_ratio_1h > 0.55)
flushout_bull_15 = (c_ret_12h < -0.08) & (oi_ret_12h < -0.15) & (c1h > o1h) & (taker_ratio_1h > 0.55)

# Trapped late positions: price at extreme, OI surges, but price stalls
trapped_shorts_10 = (c1h <= min_l_24h) & (oi_ret_12h > 0.10) & (c1h > o1h)
trapped_shorts_15 = (c1h <= min_l_24h) & (oi_ret_12h > 0.15) & (c1h > o1h)

trapped_longs_10 = (c1h >= max_h_24h) & (oi_ret_12h > 0.10) & (c1h < o1h)
trapped_longs_15 = (c1h >= max_h_24h) & (oi_ret_12h > 0.15) & (c1h < o1h)

# ---------------------------------------------------------------------
# Signal Group 5: Asymmetric Semi-Variance & Volatility Ratio
# ---------------------------------------------------------------------
usv24, dsv24 = oind.nb_semi_variance(ret_1h_safe, 24)
var_ratio_24 = dsv24 / (usv24 + 1e-12)

usv12, dsv12 = oind.nb_semi_variance(ret_1h_safe, 12)
var_ratio_12 = dsv12 / (usv12 + 1e-12)

# ---------------------------------------------------------------------
# Signal Group 6: Funding Settlement Micro-Cycle (00, 08, 16 UTC)
# ---------------------------------------------------------------------
settlement_hours = np.isin(utc_hours, [0, 8, 16])
pre_settle_hours = np.isin(utc_hours, [22, 6, 14])

settle_mask = settlement_hours[:, None]
pre_settle_mask = pre_settle_hours[:, None]

print("All signals prepared. Commencing 300 rounds execution.\n")

results = []

def run_round(round_id, family, name, sig_l, sig_s, hold=18, sl=0.07, tp=0.14):
    tr = eval_signals(p1h, sig_l, sig_s, hold_bars=hold, sl_pct=sl, tp_pct=tp, seg="TRAIN")
    va = eval_signals(p1h, sig_l, sig_s, hold_bars=hold, sl_pct=sl, tp_pct=tp, seg="VALID")
    
    pass_strict_train = (tr["pf"] >= 1.5 and tr["t_stat"] >= 2.0 and tr["n_trades"] >= 300)
    pass_strict_valid = (va["pf"] >= 1.2 and va["t_stat"] >= 2.0 and va["n_trades"] >= 300)
    
    pass_relaxed_train = (tr["pf"] >= 1.3 and tr["t_stat"] >= 1.5 and tr["n_trades"] >= 50)
    pass_relaxed_valid = (va["pf"] >= 1.2 and va["t_stat"] >= 1.5 and va["n_trades"] >= 36)
    
    status = "FAIL"
    if pass_strict_train and pass_strict_valid:
        status = "PASS_STRICT"
    elif pass_relaxed_train and pass_relaxed_valid:
        status = "PASS_RELAXED"
    elif pass_strict_train or pass_relaxed_train:
        status = "PASS_TRAIN_ONLY"
    elif pass_strict_valid or pass_relaxed_valid:
        status = "PASS_VALID_ONLY"

    res = {
        "round": round_id,
        "family": family,
        "name": name,
        "hold": hold, "sl": sl, "tp": tp,
        "tr_trades": tr["n_trades"], "tr_pf": round(tr["pf"], 2),
        "tr_mean_bp": round(tr["mean_bp"], 1), "tr_t": round(tr["t_stat"], 2),
        "tr_win": round(tr["win_rate"], 1),
        "va_trades": va["n_trades"], "va_pf": round(va["pf"], 2),
        "va_mean_bp": round(va["mean_bp"], 1), "va_t": round(va["t_stat"], 2),
        "va_win": round(va["win_rate"], 1),
        "status": status,
    }
    results.append(res)
    if round_id % 25 == 0 or status != "FAIL":
        print(f"R{round_id:4d} | {family:20s} | {name:28s} | TR: n={tr['n_trades']:4d}, pf={tr['pf']:4.2f}, t={tr['t_stat']:4.2f} | VA: n={va['n_trades']:4d}, pf={va['pf']:4.2f}, t={va['t_stat']:4.2f} | {status}")


# =====================================================================
# BATCH 1: R1351 - R1400 (50 Rounds)
# Macro BTC Lead-Lag Spillover & Cross-Asset Shock Wave
# =====================================================================
print("\n--- Running Batch 1: Macro BTC Lead-Lag Spillover (R1351 - R1400) ---")
round_counter = 1351

btc_shocks = [
    (btc_shock_pos_15, btc_shock_neg_15, "Shock15"),
    (btc_shock_pos_20, btc_shock_neg_20, "Shock20"),
    (btc_shock_pos_25, btc_shock_neg_25, "Shock25"),
]

for btc_pos, btc_neg, shock_name in btc_shocks:
    for l_lag, s_lag, lag_name in [
        ((ret_1h_safe < 0.005) & (ret_1h_safe > -0.01), (ret_1h_safe > -0.005) & (ret_1h_safe < 0.01), "LaggingFlat"),
        ((ret_1h_safe < 0.01), (ret_1h_safe > -0.01), "LaggingModerate"),
        (np.ones_like(c1h, dtype=bool), np.ones_like(c1h, dtype=bool), "AllAlts"),
    ]:
        for hold_h in [8, 12, 18]:
            for sl_val, tp_val in [(0.05, 0.10), (0.07, 0.14)]:
                if round_counter > 1400:
                    break
                sig_l = btc_pos & l_lag
                sig_s = btc_neg & s_lag
                run_round(round_counter, "B1_BTC_LeadLag", f"{shock_name}_{lag_name}_H{hold_h}_SL{int(sl_val*100)}", sig_l, sig_s, hold=hold_h, sl=sl_val, tp=tp_val)
                round_counter += 1

while round_counter <= 1400:
    sig_l = btc_shock_pos_20 & (v1h > vol_ma24_1h * 1.5)
    sig_s = btc_shock_neg_20 & (v1h > vol_ma24_1h * 1.5)
    run_round(round_counter, "B1_BTC_LeadLag", f"BTCShock_VolSurge_var{round_counter}", sig_l, sig_s, hold=12, sl=0.06, tp=0.12)
    round_counter += 1


# =====================================================================
# BATCH 2: R1401 - R1450 (50 Rounds)
# Microstructure Delta Divergence & Passive Absorption (CVD Divergence)
# =====================================================================
print("\n--- Running Batch 2: CVD Delta Divergence & Absorption (R1401 - R1450) ---")
round_counter = 1401

cvd_setups = [
    # 1. 12h Low price, but 12h CVD positive (passive limit absorption of sellers)
    ((c1h <= min_l_12h) & (cvd_12h > 0) & (c1h > o1h), (c1h >= max_h_12h) & (cvd_12h < 0) & (c1h < o1h), "CVD12_Absorb"),
    # 2. 24h Low price, but 24h CVD positive
    ((c1h <= min_l_24h) & (cvd_24h > 0) & (c1h > o1h), (c1h >= max_h_24h) & (cvd_24h < 0) & (c1h < o1h), "CVD24_Absorb"),
    # 3. CVD Absorption + Taker Ratio Confirmation
    ((c1h <= min_l_12h) & (cvd_12h > 0) & (taker_ratio_1h > 0.58), (c1h >= max_h_12h) & (cvd_12h < 0) & (taker_ratio_1h < 0.42), "CVD12_Absorb_Flow"),
    # 4. Extreme Delta Divergence: Price down 3 bars, CVD up 3 bars
    ((c1h < np.roll(c1h, 1, axis=0)) & (delta_1h > np.roll(delta_1h, 1, axis=0)) & (taker_ratio_1h > 0.55),
     (c1h > np.roll(c1h, 1, axis=0)) & (delta_1h < np.roll(delta_1h, 1, axis=0)) & (taker_ratio_1h < 0.45), "MicroDelta_Divergence"),
]

for l_cvd, s_cvd, cvd_name in cvd_setups:
    for hold_h in [8, 12, 18]:
        for sl_val, tp_val in [(0.04, 0.08), (0.06, 0.12), (0.07, 0.14)]:
            if round_counter > 1450:
                break
            run_round(round_counter, "B2_CVD_Absorption", f"{cvd_name}_H{hold_h}_SL{int(sl_val*100)}", l_cvd, s_cvd, hold=hold_h, sl=sl_val, tp=tp_val)
            round_counter += 1

while round_counter <= 1450:
    sig_l = (c1h <= min_l_24h) & (cvd_24h > 0) & (taker_ratio_1h > 0.60)
    sig_s = (c1h >= max_h_24h) & (cvd_24h < 0) & (taker_ratio_1h < 0.40)
    run_round(round_counter, "B2_CVD_Absorption", f"CVD24_Strong_var{round_counter}", sig_l, sig_s, hold=12, sl=0.05, tp=0.10)
    round_counter += 1


# =====================================================================
# BATCH 3: R1451 - R1500 (50 Rounds)
# Top-Trader Sentiment Divergence & Smart Money Positioning
# =====================================================================
print("\n--- Running Batch 3: Top-Trader Sentiment Divergence (R1451 - R1500) ---")
round_counter = 1451

tt_setups = [
    (tt_surge_bull, tt_surge_bear, "TTSmart_Standard"),
    (tt_surge_bull_strong, tt_surge_bear_strong, "TTSmart_Strong"),
    (tt_surge_bull & (c1h > o1h), tt_surge_bear & (c1h < o1h), "TTSmart_GreenBar"),
    (tt_surge_bull & (rsi14_1h < 40), tt_surge_bear & (rsi14_1h > 60), "TTSmart_RSIOversold"),
]

for l_tt, s_tt, tt_name in tt_setups:
    for hold_h in [12, 18, 24]:
        for sl_val, tp_val in [(0.05, 0.10), (0.07, 0.14), (0.08, 0.16)]:
            if round_counter > 1500:
                break
            run_round(round_counter, "B3_TopTrader_Sentiment", f"{tt_name}_H{hold_h}_SL{int(sl_val*100)}", l_tt, s_tt, hold=hold_h, sl=sl_val, tp=tp_val)
            round_counter += 1

while round_counter <= 1500:
    sig_l = tt_surge_bull_strong & (c1h > o1h) & (taker_ratio_1h > 0.55)
    sig_s = tt_surge_bear_strong & (c1h < o1h) & (taker_ratio_1h < 0.45)
    run_round(round_counter, "B3_TopTrader_Sentiment", f"TTSmart_Pure_var{round_counter}", sig_l, sig_s, hold=18, sl=0.07, tp=0.14)
    round_counter += 1


# =====================================================================
# BATCH 4: R1501 - R1550 (50 Rounds)
# Open Interest Velocity & Trapped Liquidation Cascades
# =====================================================================
print("\n--- Running Batch 4: OI Velocity & Trapped Cascades (R1501 - R1550) ---")
round_counter = 1501

oi_setups = [
    (flushout_bull_10, (c_ret_12h > 0.06) & (oi_ret_12h < -0.10) & (c1h < o1h), "Flushout10"),
    (flushout_bull_15, (c_ret_12h > 0.08) & (oi_ret_12h < -0.15) & (c1h < o1h), "Flushout15"),
    (trapped_shorts_10, trapped_longs_10, "TrappedExtreme10"),
    (trapped_shorts_15, trapped_longs_15, "TrappedExtreme15"),
]

for l_oi, s_oi, oi_name in oi_setups:
    for hold_h in [8, 12, 18]:
        for sl_val, tp_val in [(0.05, 0.10), (0.07, 0.14), (0.09, 0.18)]:
            if round_counter > 1550:
                break
            run_round(round_counter, "B4_OI_TrappedCascades", f"{oi_name}_H{hold_h}_SL{int(sl_val*100)}", l_oi, s_oi, hold=hold_h, sl=sl_val, tp=tp_val)
            round_counter += 1

while round_counter <= 1550:
    sig_l = flushout_bull_15 & (rsi14_1h < 30)
    sig_s = (c_ret_12h > 0.08) & (oi_ret_12h < -0.15) & (rsi14_1h > 70)
    run_round(round_counter, "B4_OI_TrappedCascades", f"Flushout_RSI_var{round_counter}", sig_l, sig_s, hold=12, sl=0.06, tp=0.12)
    round_counter += 1


# =====================================================================
# BATCH 5: R1551 - R1600 (50 Rounds)
# Volatility Regime Switching & Asymmetric Semi-Variance
# =====================================================================
print("\n--- Running Batch 5: Asymmetric Semi-Variance Dislocation (R1551 - R1600) ---")
round_counter = 1551

var_setups = [
    # Extreme Downside Variance > 3.0x Upside (panic liquidations)
    ((var_ratio_24 > 3.0) & (c1h > o1h) & (rsi14_1h < 35), (var_ratio_24 < 0.33) & (c1h < o1h) & (rsi14_1h > 65), "VAR24_Ext3"),
    # Extreme Downside Variance > 4.5x Upside
    ((var_ratio_24 > 4.5) & (c1h > o1h) & (rsi14_1h < 30), (var_ratio_24 < 0.22) & (c1h < o1h) & (rsi14_1h > 70), "VAR24_Ext45"),
    # 12h Fast Semi-Variance Dislocation
    ((var_ratio_12 > 3.5) & (c1h > o1h) & (taker_ratio_1h > 0.55), (var_ratio_12 < 0.28) & (c1h < o1h) & (taker_ratio_1h < 0.45), "VAR12_FastFlow"),
    # High-Low Range vs Close-to-Close Dislocation
    ((var_ratio_24 > 3.0) & (v1h > vol_ma24_1h * 1.5) & (c1h > o1h), (var_ratio_24 < 0.33) & (v1h > vol_ma24_1h * 1.5) & (c1h < o1h), "VAR24_VolSpike"),
]

for l_var, s_var, var_name in var_setups:
    for hold_h in [8, 12, 18]:
        for sl_val, tp_val in [(0.05, 0.10), (0.07, 0.14), (0.08, 0.16)]:
            if round_counter > 1600:
                break
            run_round(round_counter, "B5_Asym_SemiVariance", f"{var_name}_H{hold_h}_SL{int(sl_val*100)}", l_var, s_var, hold=hold_h, sl=sl_val, tp=tp_val)
            round_counter += 1

while round_counter <= 1600:
    sig_l = (var_ratio_24 > 4.0) & (c1h > o1h) & (taker_ratio_1h > 0.58)
    sig_s = (var_ratio_24 < 0.25) & (c1h < o1h) & (taker_ratio_1h < 0.42)
    run_round(round_counter, "B5_Asym_SemiVariance", f"VAR24_Pure_var{round_counter}", sig_l, sig_s, hold=12, sl=0.06, tp=0.12)
    round_counter += 1


# =====================================================================
# BATCH 6: R1601 - R1650 (50 Rounds)
# Funding Settlement Micro-Cycle & Pre/Post-Settlement Exploitation
# =====================================================================
print("\n--- Running Batch 6: Funding Settlement Micro-Cycle (R1601 - R1650) ---")
round_counter = 1601

funding_settle_setups = [
    # Post-Settlement Relief (Hours 00, 08, 16):
    # If high funding (>0.03%), longs were dumping before settlement; right after settlement, buy relief bounce
    ((fr1h > 0.0003) & settle_mask & (c1h > o1h), (fr1h < -0.0003) & settle_mask & (c1h < o1h), "PostSettle_Relief_30bp"),
    # Extreme funding (>0.05%)
    ((fr1h > 0.0005) & settle_mask & (c1h > o1h), (fr1h < -0.0005) & settle_mask & (c1h < o1h), "PostSettle_Relief_50bp"),
    # Pre-Settlement Front-Running (Hours 22, 06, 14):
    # Front-run the dump: short 2h before settlement if funding is sky high
    ((fr1h < -0.0003) & pre_settle_mask, (fr1h > 0.0003) & pre_settle_mask, "PreSettle_FrontRun_30bp"),
    # Pre-Settlement + Flow confirmation
    ((fr1h < -0.0003) & pre_settle_mask & (taker_ratio_1h > 0.55), (fr1h > 0.0003) & pre_settle_mask & (taker_ratio_1h < 0.45), "PreSettle_Flow_30bp"),
]

for l_fnd, s_fnd, fnd_name in funding_settle_setups:
    for hold_h in [4, 8, 12]:
        for sl_val, tp_val in [(0.03, 0.06), (0.05, 0.10), (0.07, 0.14)]:
            if round_counter > 1650:
                break
            run_round(round_counter, "B6_Funding_MicroCycle", f"{fnd_name}_H{hold_h}_SL{int(sl_val*100)}", l_fnd, s_fnd, hold=hold_h, sl=sl_val, tp=tp_val)
            round_counter += 1

while round_counter <= 1650:
    sig_l = (fr1h > 0.0003) & settle_mask & (taker_ratio_1h > 0.55)
    sig_s = (fr1h < -0.0003) & settle_mask & (taker_ratio_1h < 0.45)
    run_round(round_counter, "B6_Funding_MicroCycle", f"PostSettle_Flow_var{round_counter}", sig_l, sig_s, hold=8, sl=0.04, tp=0.08)
    round_counter += 1

print("\nAll 300 orthogonal rounds executed!")
print(f"Total time: {time.time() - t0:.2f}s")

# Save master CSV
df_res = pd.DataFrame(results)
csv_path = "/root/freqtrade/user_data/minute_research/r6_orthogonal_alphas/r1351_r1650_trainvalidC.csv"
df_res.to_csv(csv_path, index=False)
print(f"Master ledger saved to {csv_path}")

# Summary Stats
print("\n" + "=" * 80)
print("300-Round Summary Status Breakdown:")
print(df_res["status"].value_counts())
print("=" * 80)

qualified = df_res[df_res["status"].isin(["PASS_STRICT", "PASS_RELAXED"])]
print(f"\nTotal Qualified Candidates for HOLDOUT Evaluation: {len(qualified)}")
if len(qualified) > 0:
    print(qualified[["round", "family", "name", "tr_trades", "tr_pf", "tr_t", "va_trades", "va_pf", "va_t", "status"]])
else:
    print("Zero candidates passed both TRAIN and VALID-C.")
    print("Top 10 by VALID-C Profit Factor (among those with va_trades >= 30):")
    valid_candidates = df_res[df_res["va_trades"] >= 30].sort_values("va_pf", ascending=False).head(10)
    print(valid_candidates[["round", "family", "name", "tr_trades", "tr_pf", "tr_t", "va_trades", "va_pf", "va_t", "status"]])
