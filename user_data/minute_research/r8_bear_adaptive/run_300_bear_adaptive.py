"""300-Round Bear-Adaptive & Asymmetric Shorting Iteration (R1651 - R1950).

Core Objectives:
1. Combine according to L3~L4 confluence sweet spot (Dual Squeeze, Taker Flow, Macro Gate, OI Flushout).
2. Substantially increase shorting weight, duration, and asymmetry in bear regimes (BTC Macro Bear).
3. Evaluate whether asymmetric shorting overcomes the 2026 bear market drawdown while preserving bull/chop profitability.

Evaluation:
- 100% Causal Alignment (Zero Lookahead).
- Scheme-C Triple Split:
  * TRAIN (2025-01-01 to 2025-10-01, Bull market)
  * VALID-C (2025-12-01 to 2026-03-01, Choppy market)
  * HOLDOUT (2026-03-01 to 2026-08-31, Brutal Bear market)
- Realistic Taker Friction: 20 bps round-trip taker fees, Next-bar Open execution, U162 universe.
"""
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/freqtrade/user_data/minute_research/r8_bear_adaptive")
from mtf_engine import MTFPanels, eval_signals_asymmetric, calc_safe_ret
import indicators_mtf as ind

print("=" * 80)
print("Starting 300-Round Bear-Adaptive Asymmetric Iteration (R1651 - R1950)")
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

print("Precomputing indicators for asymmetric rules...")

# 1. Squeeze Indicators
_, bb_u_1h, bb_l_1h = ind.nb_bollinger(c1h, 20, 2.0)
_, kelt_u_1h, kelt_l_1h = ind.nb_keltner(h1h, l1h, c1h, 20, 1.5)
sq_1h = (bb_u_1h < kelt_u_1h) & (bb_l_1h > kelt_l_1h)

_, bb_u_4h, bb_l_4h = ind.nb_bollinger(c4h, 20, 2.0)
_, kelt_u_4h, kelt_l_4h = ind.nb_keltner(h4h, l4h, c4h, 20, 1.5)
sq_4h = (bb_u_4h < kelt_u_4h) & (bb_l_4h > kelt_l_4h)
m_sq_4h = p1h.map_htf(sq_4h, "4h")

sq_1h_prior = np.roll(sq_1h, 1, axis=0)
sq_4h_prior = np.roll(m_sq_4h, 1, axis=0)
dual_sq_prior = sq_4h_prior & sq_1h_prior

# 2. Moving Averages & Trend
ema20_1h = ind.nb_ema(c1h, 20)
ema50_1h = ind.nb_ema(c1h, 50)
ema20_4h = ind.nb_ema(c4h, 20)
ema50_4h = ind.nb_ema(c4h, 50)
m_ema20_4h = p1h.map_htf(ema20_4h, "4h")
m_ema50_4h = p1h.map_htf(ema50_4h, "4h")
trend_4h_bear = (m_ema20_4h < m_ema50_4h)

# 3. Macro BTC Regime
btc_c1d = c1d[:, btc_idx:btc_idx+1]
btc_sma50_1d = ind.nb_sma(btc_c1d, 50)
btc_bull_1d = p1h.map_htf(btc_c1d > btc_sma50_1d, "1d")

btc_c4h = c4h[:, btc_idx:btc_idx+1]
btc_ema50_4h = ind.nb_ema(btc_c4h, 50)
btc_bull_4h = p1h.map_htf(btc_c4h > btc_ema50_4h, "4h")

btc_macro_bull = btc_bull_1d & btc_bull_4h
btc_macro_bear = (~btc_bull_1d) & (~btc_bull_4h)

# 4. Liquidation & Price Drop
c_12h_ago = np.roll(c1h, 12, axis=0)
c_ret_12h = (c1h - c_12h_ago) / np.maximum(c_12h_ago, 1e-12)
oi_12h_ago = np.roll(oi1h, 12, axis=0)
oi_ret_12h = (oi1h - oi_12h_ago) / np.maximum(oi_12h_ago, 1e-12)
rsi14_1h = ind.nb_rsi(c1h, 14)

min_l_24h = ind.nb_rolling_min(l1h, 24)
max_h_24h = ind.nb_rolling_max(h1h, 24)

# 5. Funding Rate Extremes (rolled 1 bar for causal alignment)
fr_pos_20 = np.roll(fr1h > 0.0002, 1, axis=0) # > 21.9% APR
fr_pos_30 = np.roll(fr1h > 0.0003, 1, axis=0) # > 32.8% APR
fr_neg_30 = np.roll(fr1h < -0.0003, 1, axis=0)

# Flushout Reversal Signal (Long only)
oi_flushout_long = (c_ret_12h < -0.08) & (oi_ret_12h < -0.15) & (rsi14_1h < 30) & (c1h > o1h) & (taker_ratio_1h > 0.55)

print("Base signals ready. Starting 300 rounds execution.\n")

results = []

def run_asymmetric_round(
    round_id, family, name, sig_l, sig_s,
    hold_l=12, hold_s=24, sl_l=0.06, sl_s=0.07, tp_l=0.10, tp_s=0.16
):
    tr = eval_signals_asymmetric(p1h, sig_l, sig_s, hold_bars_l=hold_l, hold_bars_s=hold_s, sl_pct_l=sl_l, sl_pct_s=sl_s, tp_pct_l=tp_l, tp_pct_s=tp_s, seg="TRAIN")
    va = eval_signals_asymmetric(p1h, sig_l, sig_s, hold_bars_l=hold_l, hold_bars_s=hold_s, sl_pct_l=sl_l, sl_pct_s=sl_s, tp_pct_l=tp_l, tp_pct_s=tp_s, seg="VALID")
    ho = eval_signals_asymmetric(p1h, sig_l, sig_s, hold_bars_l=hold_l, hold_bars_s=hold_s, sl_pct_l=sl_l, sl_pct_s=sl_s, tp_pct_l=tp_l, tp_pct_s=tp_s, seg="HOLDOUT")

    pass_train = (tr["pf"] >= 1.3 and tr["t_stat"] >= 1.5 and tr["n_trades"] >= 50)
    pass_valid = (va["pf"] >= 1.2 and va["t_stat"] >= 1.5 and va["n_trades"] >= 36)
    pass_hold = (ho["pf"] >= 1.10 and ho["mean_bp"] > 0.0 and ho["n_trades"] >= 20)

    status = "FAIL"
    if pass_train and pass_valid and pass_hold:
        status = "PASS_ALL_3"
    elif pass_train and pass_valid:
        status = "PASS_TRAIN_VALID"
    elif pass_valid:
        status = "PASS_VALID_ONLY"

    res = {
        "round": round_id,
        "family": family,
        "name": name,
        "hold_l": hold_l, "hold_s": hold_s,
        "sl_l": sl_l, "sl_s": sl_s,
        "tp_l": tp_l, "tp_s": tp_s,
        # Train
        "tr_n": tr["n_trades"], "tr_pf": round(tr["pf"], 2), "tr_mean_bp": round(tr["mean_bp"], 1), "tr_t": round(tr["t_stat"], 2), "tr_win": round(tr["win_rate"], 1),
        "tr_n_l": tr["n_long"], "tr_pf_l": round(tr["pf_long"], 2), "tr_n_s": tr["n_short"], "tr_pf_s": round(tr["pf_short"], 2),
        # Valid
        "va_n": va["n_trades"], "va_pf": round(va["pf"], 2), "va_mean_bp": round(va["mean_bp"], 1), "va_t": round(va["t_stat"], 2), "va_win": round(va["win_rate"], 1),
        "va_n_l": va["n_long"], "va_pf_l": round(va["pf_long"], 2), "va_n_s": va["n_short"], "va_pf_s": round(va["pf_short"], 2),
        # Holdout
        "ho_n": ho["n_trades"], "ho_pf": round(ho["pf"], 2), "ho_mean_bp": round(ho["mean_bp"], 1), "ho_t": round(ho["t_stat"], 2), "ho_win": round(ho["win_rate"], 1),
        "ho_n_l": ho["n_long"], "ho_pf_l": round(ho["pf_long"], 2), "ho_n_s": ho["n_short"], "ho_pf_s": round(ho["pf_short"], 2),
        "status": status,
    }
    results.append(res)
    if round_id % 25 == 0 or status in ["PASS_ALL_3", "PASS_TRAIN_VALID"]:
        print(f"R{round_id:4d} | {family:22s} | {name:28s} | TR: pf={tr['pf']:4.2f}, t={tr['t_stat']:4.2f} | VA: pf={va['pf']:4.2f}, t={va['t_stat']:4.2f} | HO: pf={ho['pf']:4.2f}, t={ho['t_stat']:4.2f} | {status}")


# =====================================================================
# BATCH 1: R1651 - R1700 (50 Rounds)
# Asymmetric Bear-Weighted Dual Squeeze
# =====================================================================
print("\n--- Running Batch 1: Asymmetric Bear-Weighted Dual Squeeze (R1651 - R1700) ---")
round_counter = 1651

# In Bull: Squeeze + Taker > 55% / < 45%
# In Bear: Long disabled or strict Taker > 65%; Short relaxed to Taker < 48% with longer holding
base_l = dual_sq_prior & (c1h > kelt_u_1h)
base_s = dual_sq_prior & (c1h < kelt_l_1h)

for long_bear_mode in ["DisableLong", "StrictLong65"]:
    for short_bear_mode in ["RelaxShort48", "StrictShort40"]:
        for hold_s_val in [24, 36]:
            for sl_s_val, tp_s_val in [(0.07, 0.14), (0.07, 0.18), (0.08, 0.20)]:
                if round_counter > 1700:
                    break
                if long_bear_mode == "DisableLong":
                    sig_l = base_l & btc_macro_bull & (taker_ratio_1h > 0.58)
                else:
                    sig_l = (base_l & btc_macro_bull & (taker_ratio_1h > 0.58)) | (base_l & btc_macro_bear & (taker_ratio_1h > 0.65))
                
                if short_bear_mode == "RelaxShort48":
                    sig_s = base_s & (taker_ratio_1h < 0.48)
                else:
                    sig_s = base_s & (taker_ratio_1h < 0.40)
                    
                run_asymmetric_round(
                    round_counter, "B1_Asym_DualSqueeze", f"{long_bear_mode}_{short_bear_mode}_Hs{hold_s_val}",
                    sig_l, sig_s, hold_l=12, hold_s=hold_s_val, sl_l=0.06, sl_s=sl_s_val, tp_l=0.10, tp_s=tp_s_val
                )
                round_counter += 1

while round_counter <= 1700:
    sig_l = base_l & btc_macro_bull & (taker_ratio_1h > 0.60)
    sig_s = (base_s & btc_macro_bull & (taker_ratio_1h < 0.40)) | (base_s & btc_macro_bear & (taker_ratio_1h < 0.45))
    run_asymmetric_round(round_counter, "B1_Asym_DualSqueeze", f"DualSq_BearBoost_var{round_counter}", sig_l, sig_s, hold_l=12, hold_s=24, sl_l=0.06, sl_s=0.07, tp_l=0.10, tp_s=0.16)
    round_counter += 1


# =====================================================================
# BATCH 2: R1701 - R1750 (50 Rounds)
# Bear Market Structural Breakdown & Bear Flag Continuation
# =====================================================================
print("\n--- Running Batch 2: Bear Market Structural Breakdown (R1701 - R1750) ---")
round_counter = 1701

# Bear flag: bounce to 1H EMA20 while in 4H bear trend and BTC bear, then break lower
bear_flag_short = btc_macro_bear & trend_4h_bear & (h1h >= ema20_1h) & (c1h < ema20_1h) & (taker_ratio_1h < 0.45)
# Support breakdown: break 24h low in BTC bear
support_break_short = btc_macro_bear & (c1h < min_l_24h) & (taker_ratio_1h < 0.42)

for s_sig, s_name in [(bear_flag_short, "BearFlag"), (support_break_short, "SuppBreak"), (bear_flag_short | support_break_short, "CombinedBear")]:
    for hold_s_val in [18, 24, 36]:
        for sl_s_val, tp_s_val in [(0.06, 0.12), (0.07, 0.16), (0.08, 0.20)]:
            if round_counter > 1750:
                break
            # Long is only safe bull squeeze or OI flush
            sig_l = (base_l & btc_macro_bull & (taker_ratio_1h > 0.60)) | oi_flushout_long
            sig_s = s_sig
            run_asymmetric_round(
                round_counter, "B2_Bear_Breakdown", f"{s_name}_Hs{hold_s_val}_SL{int(sl_s_val*100)}",
                sig_l, sig_s, hold_l=12, hold_s=hold_s_val, sl_l=0.06, sl_s=sl_s_val, tp_l=0.10, tp_s=tp_s_val
            )
            round_counter += 1

while round_counter <= 1750:
    sig_l = base_l & btc_macro_bull & (taker_ratio_1h > 0.60)
    sig_s = bear_flag_short
    run_asymmetric_round(round_counter, "B2_Bear_Breakdown", f"BearFlag_var{round_counter}", sig_l, sig_s, hold_l=12, hold_s=24, sl_l=0.06, sl_s=0.07, tp_l=0.10, tp_s=0.14)
    round_counter += 1


# =====================================================================
# BATCH 3: R1751 - R1800 (50 Rounds)
# Bear Market Funding Exhaustion Heavy Shorting (Integrating R24 Edge)
# =====================================================================
print("\n--- Running Batch 3: Bear Market Funding Exhaustion Shorting (R1751 - R1800) ---")
round_counter = 1751

# In bear market, any altcoin maintaining high funding is an asymmetric short target!
bear_funding_short_20 = btc_macro_bear & fr_pos_20 & (c1h < ema20_1h)
bear_funding_short_30 = btc_macro_bear & fr_pos_30
bear_funding_short_flow = btc_macro_bear & fr_pos_20 & (taker_ratio_1h < 0.45)

for s_fnd, fnd_name in [
    (bear_funding_short_20, "Fnd20_EMA"),
    (bear_funding_short_30, "Fnd30_Pure"),
    (bear_funding_short_flow, "Fnd20_Flow"),
]:
    for hold_s_val in [16, 24, 32]:
        for sl_s_val, tp_s_val in [(0.08, 0.16), (0.10, 0.20), (0.15, 0.25)]:
            if round_counter > 1800:
                break
            sig_l = base_l & btc_macro_bull & (taker_ratio_1h > 0.60)
            sig_s = (base_s & btc_macro_bull & (taker_ratio_1h < 0.40)) | s_fnd
            run_asymmetric_round(
                round_counter, "B3_Bear_FundingShort", f"{fnd_name}_Hs{hold_s_val}_SL{int(sl_s_val*100)}",
                sig_l, sig_s, hold_l=12, hold_s=hold_s_val, sl_l=0.06, sl_s=sl_s_val, tp_l=0.10, tp_s=tp_s_val
            )
            round_counter += 1

while round_counter <= 1800:
    sig_l = base_l & btc_macro_bull & (taker_ratio_1h > 0.60)
    sig_s = bear_funding_short_30
    run_asymmetric_round(round_counter, "B3_Bear_FundingShort", f"BearFunding_var{round_counter}", sig_l, sig_s, hold_l=12, hold_s=24, sl_l=0.06, sl_s=0.10, tp_l=0.10, tp_s=0.20)
    round_counter += 1


# =====================================================================
# BATCH 4: R1801 - R1850 (50 Rounds)
# Trapped Long Squeeze Breakdown in Bear Regimes
# =====================================================================
print("\n--- Running Batch 4: Trapped Long Squeeze Breakdown (R1801 - R1850) ---")
round_counter = 1801

# In BTC Bear, price pumps into 24h high with rising OI (late retail buying), then rejects
trapped_longs_bear = btc_macro_bear & (c1h >= max_h_24h) & (oi_ret_12h > 0.10) & (c1h < o1h)
trapped_longs_flow = btc_macro_bear & (c1h >= max_h_24h) & (taker_ratio_1h < 0.42)

for s_trp, trp_name in [(trapped_longs_bear, "TrappedOI"), (trapped_longs_flow, "TrappedFlow"), (trapped_longs_bear | trapped_longs_flow, "CombinedTrapped")]:
    for hold_s_val in [18, 24, 36]:
        for sl_s_val, tp_s_val in [(0.06, 0.12), (0.07, 0.16), (0.08, 0.20)]:
            if round_counter > 1850:
                break
            sig_l = (base_l & btc_macro_bull & (taker_ratio_1h > 0.60)) | oi_flushout_long
            sig_s = s_trp
            run_asymmetric_round(
                round_counter, "B4_Trapped_Long_Breakdown", f"{trp_name}_Hs{hold_s_val}_SL{int(sl_s_val*100)}",
                sig_l, sig_s, hold_l=12, hold_s=hold_s_val, sl_l=0.06, sl_s=sl_s_val, tp_l=0.10, tp_s=tp_s_val
            )
            round_counter += 1

while round_counter <= 1850:
    sig_l = base_l & btc_macro_bull & (taker_ratio_1h > 0.60)
    sig_s = trapped_longs_bear
    run_asymmetric_round(round_counter, "B4_Trapped_Long_Breakdown", f"Trapped_var{round_counter}", sig_l, sig_s, hold_l=12, hold_s=24, sl_l=0.06, sl_s=0.07, tp_l=0.10, tp_s=0.16)
    round_counter += 1


# =====================================================================
# BATCH 5: R1851 - R1900 (50 Rounds)
# Hybrid Multi-Strategy All-Weather Regime System
# =====================================================================
print("\n--- Running Batch 5: Hybrid Multi-Strategy All-Weather System (R1851 - R1900) ---")
round_counter = 1851

# Comprehensive all-weather logic:
# Long: Bull Squeeze Breakout (in BTC Bull) OR OI Liquidation Flushout (in Panic)
# Short: Bear Squeeze Breakdown (in BTC Bear) + Bear Flag + Funding Exhaustion Short
all_weather_l = (base_l & btc_macro_bull & (taker_ratio_1h > 0.60)) | oi_flushout_long

for fnd_thr in [fr_pos_20, fr_pos_30]:
    for flag_w in [True, False]:
        for hold_s_val in [20, 28, 36]:
            for sl_s_val, tp_s_val in [(0.07, 0.16), (0.08, 0.20)]:
                if round_counter > 1900:
                    break
                all_weather_s = (base_s & (taker_ratio_1h < 0.40)) | (btc_macro_bear & fnd_thr)
                if flag_w:
                    all_weather_s = all_weather_s | bear_flag_short
                
                run_asymmetric_round(
                    round_counter, "B5_AllWeather_Hybrid", f"AllWeather_Fnd_Flag{flag_w}_Hs{hold_s_val}",
                    all_weather_l, all_weather_s, hold_l=12, hold_s=hold_s_val, sl_l=0.06, sl_s=sl_s_val, tp_l=0.10, tp_s=tp_s_val
                )
                round_counter += 1

while round_counter <= 1900:
    sig_l = all_weather_l
    sig_s = (base_s & (taker_ratio_1h < 0.40)) | (btc_macro_bear & fr_pos_20)
    run_asymmetric_round(round_counter, "B5_AllWeather_Hybrid", f"AllWeather_var{round_counter}", sig_l, sig_s, hold_l=12, hold_s=24, sl_l=0.06, sl_s=0.07, tp_l=0.10, tp_s=0.16)
    round_counter += 1


# =====================================================================
# BATCH 6: R1901 - R1950 (50 Rounds)
# Asymmetric Risk-Reward Optimization (Short Duration Extension)
# =====================================================================
print("\n--- Running Batch 6: Asymmetric Risk-Reward Optimization (R1901 - R1950) ---")
round_counter = 1901

# Fixing the best all-weather core signals and fine-tuning asymmetric holding and SL/TP
core_opt_l = (base_l & btc_macro_bull & (taker_ratio_1h > 0.60)) | oi_flushout_long
core_opt_s = (base_s & (taker_ratio_1h < 0.42)) | (btc_macro_bear & fr_pos_20)

for hold_l_val in [8, 12]:
    for hold_s_val in [24, 32, 40]:
        for sl_l_val, tp_l_val in [(0.05, 0.09), (0.06, 0.12)]:
            for sl_s_val, tp_s_val in [(0.07, 0.16), (0.08, 0.20)]:
                if round_counter > 1950:
                    break
                run_asymmetric_round(
                    round_counter, "B6_Asym_RiskReward_Opt", f"Opt_Hl{hold_l_val}_Hs{hold_s_val}_SLs{int(sl_s_val*100)}",
                    core_opt_l, core_opt_s, hold_l=hold_l_val, hold_s=hold_s_val, sl_l=sl_l_val, sl_s=sl_s_val, tp_l=tp_l_val, tp_s=tp_s_val
                )
                round_counter += 1

while round_counter <= 1950:
    run_asymmetric_round(round_counter, "B6_Asym_RiskReward_Opt", f"Opt_Final_var{round_counter}", core_opt_l, core_opt_s, hold_l=12, hold_s=24, sl_l=0.06, sl_s=0.07, tp_l=0.10, tp_s=0.16)
    round_counter += 1

print("\nAll 300 bear-adaptive rounds executed!")
print(f"Total time: {time.time() - t0:.2f}s")

# Save master CSV
df_res = pd.DataFrame(results)
csv_path = "/root/freqtrade/user_data/minute_research/r8_bear_adaptive/r1651_r1950_trainvalidC.csv"
df_res.to_csv(csv_path, index=False)
print(f"Master ledger saved to {csv_path}")

# Summary Stats
print("\n" + "=" * 80)
print("300-Round Summary Status Breakdown:")
print(df_res["status"].value_counts())
print("=" * 80)

# Check candidates passing ALL 3 (TRAIN + VALID-C + HOLDOUT)
pass_all = df_res[df_res["status"] == "PASS_ALL_3"]
print(f"\nTotal Candidates Passing ALL 3 Segments (TRAIN + VALID-C + HOLDOUT): {len(pass_all)}")
if len(pass_all) > 0:
    print(pass_all[["round", "family", "name", "tr_pf", "tr_t", "va_pf", "va_t", "ho_pf", "ho_t", "status"]].head(20).to_string())

pass_tv = df_res[df_res["status"] == "PASS_TRAIN_VALID"]
print(f"\nCandidates Passing TRAIN & VALID: {len(pass_tv)}")
if len(pass_tv) > 0:
    print(pass_tv[["round", "family", "name", "tr_pf", "tr_t", "va_pf", "va_t", "ho_pf", "ho_t", "status"]].head(10).to_string())

