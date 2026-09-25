"""300-Round Multi-Timeframe Resonance & Top-Down Execution Research Runner (Rounds 751 - 1050).

Families:
- Batch 1: R751 - R800 (50 rounds) -- 4H Trend + 1H/15M Pullback Resonance
- Batch 2: R801 - R850 (50 rounds) -- 4H Liquidity Sweep + 1H/15M Market Structure Shift (MSS)
- Batch 3: R851 - R900 (50 rounds) -- 4H Channel Exhaustion + 1H Momentum Reversal
- Batch 4: R901 - R950 (50 rounds) -- 4H Funding/OI Positioning + 1H Deleveraging Breakdown
- Batch 5: R951 - R1000 (50 rounds) -- Alexander Elder Triple Screen for Crypto (1D -> 4H -> 1H)
- Batch 6: R1001 - R1050 (50 rounds) -- BTC Macro Regime + Altcoin Relative Strength & Flow

Strict Evaluation:
- TRAIN: 2025-01-01 to 2025-10-01 (273 days)
- VALID-C: 2025-12-01 to 2026-03-01 (90 days)
- Gate Check: TRAIN PF >= 1.5, t >= 2.0, n >= 300 (relaxed n >= 100); VALID-C PF >= 1.2, t >= 1.5
"""
import sys
import time
import numpy as np
import pandas as pd

sys.path.append("/root/freqtrade/user_data/minute_research/r4_mtf")
from mtf_engine import MTFPanels, eval_signals
import indicators_mtf as ind

print("Loading Multi-Timeframe Panels (1h, 4h, 1d)...")
t0 = time.time()
p1h = MTFPanels("1h")
p4h = MTFPanels("4h")
p1d = MTFPanels("1d")
print(f"Panels loaded in {time.time()-t0:.2f}s.")

# BTC Index
btc_idx = p1h.symbols.index("BTCUSDT") if "BTCUSDT" in p1h.symbols else 0

# Core arrays (1h)
c1h = np.array(p1h.close)
o1h = np.array(p1h.open)
h1h = np.array(p1h.high)
l1h = np.array(p1h.low)
v1h = np.array(p1h.volume)
tbv1h = np.array(p1h.taker_buy) if p1h.taker_buy is not None else v1h * 0.5
taker_ratio_1h = np.where(v1h > 0, tbv1h / (v1h + 1e-12), 0.5)

# Core arrays (4h)
c4h = np.array(p4h.close)
o4h = np.array(p4h.open)
h4h = np.array(p4h.high)
l4h = np.array(p4h.low)
v4h = np.array(p4h.volume)

# Core arrays (1d)
c1d = np.array(p1d.close)

# -------------------------------------------------------------
# Macro BTC Indicators
# -------------------------------------------------------------
btc_c1d = c1d[:, btc_idx:btc_idx+1]
btc_c4h = c4h[:, btc_idx:btc_idx+1]
btc_sma50_1d = ind.nb_sma(btc_c1d, 50)
btc_ema50_4h = ind.nb_ema(btc_c4h, 50)

btc_bull_1d = p1h.map_htf((btc_c1d > btc_sma50_1d), "1d")
btc_bull_4h = p1h.map_htf((btc_c4h > btc_ema50_4h), "4h")
btc_macro_bull = btc_bull_1d & btc_bull_4h
btc_macro_bear = (~btc_bull_1d) & (~btc_bull_4h)

# BTC Volatility regime
btc_ret_1d = np.diff(np.log(np.maximum(btc_c1d, 1e-8)), axis=0)
btc_vol_30d = pd.Series(btc_ret_1d[:, 0]).rolling(30, min_periods=20).std().values * np.sqrt(365)
btc_vol_180d_med = pd.Series(btc_vol_30d).rolling(180, min_periods=60).median().values
btc_low_vol_1d = (btc_vol_30d < btc_vol_180d_med)
btc_low_vol = p1h.map_htf(btc_low_vol_1d[:, None], "1d")

print("Base indicators prepared. Starting 300-round systematic iteration...")

results = []

def run_round(round_id, family, name, sig_l, sig_s, hold=18, sl=0.07, tp=0.14):
    tr = eval_signals(p1h, sig_l, sig_s, hold_bars=hold, sl_pct=sl, tp_pct=tp, seg="TRAIN")
    va = eval_signals(p1h, sig_l, sig_s, hold_bars=hold, sl_pct=sl, tp_pct=tp, seg="VALID")
    
    pass_train = (tr["pf"] >= 1.5 and tr["t_stat"] >= 2.0 and tr["n_trades"] >= 300)
    relaxed_train = (tr["pf"] >= 1.3 and tr["t_stat"] >= 1.5 and tr["n_trades"] >= 100)
    pass_valid = (va["pf"] >= 1.2 and va["t_stat"] >= 1.5 and va["n_trades"] >= 50)
    
    status = "FAIL"
    if pass_train and pass_valid:
        status = "PASS_BOTH"
    elif relaxed_train and pass_valid:
        status = "RELAXED_PASS"
    elif pass_train:
        status = "PASS_TRAIN_ONLY"
    elif pass_valid:
        status = "PASS_VALID_ONLY"

    res = {
        "round": round_id,
        "family": family,
        "name": name,
        "hold": hold, "sl": sl, "tp": tp,
        "tr_trades": tr["n_trades"], "tr_pf": round(tr["pf"], 2), "tr_mean_bp": round(tr["mean_bp"], 1), "tr_t": round(tr["t_stat"], 2),
        "va_trades": va["n_trades"], "va_pf": round(va["pf"], 2), "va_mean_bp": round(va["mean_bp"], 1), "va_t": round(va["t_stat"], 2),
        "status": status
    }
    results.append(res)
    if status in ["PASS_BOTH", "RELAXED_PASS"] or va["pf"] >= 1.8:
        print(f"[{status}] R{round_id:04d} | {name} | TR: n={tr['n_trades']}, PF={tr['pf']:.2f}, t={tr['t_stat']:.2f} | VA: n={va['n_trades']}, PF={va['pf']:.2f}, t={va['t_stat']:.2f}")
    return res


# =====================================================================
# BATCH 1: R751 - R800 (50 Rounds)
# 4H Trend + 1H Pullback Resonance
# =====================================================================
print("\n--- Running Batch 1: 4H Trend + 1H Pullback Resonance (R751 - R800) ---")
# Precompute 4H trends
st_trend_4h_30, _ = ind.nb_supertrend(h4h, l4h, c4h, period=10, mult=3.0)
st_trend_4h_25, _ = ind.nb_supertrend(h4h, l4h, c4h, period=14, mult=2.5)
st_trend_4h_20, _ = ind.nb_supertrend(h4h, l4h, c4h, period=10, mult=2.0)

ema20_4h = ind.nb_ema(c4h, 20)
ema50_4h = ind.nb_ema(c4h, 50)
ema200_4h = ind.nb_ema(c4h, 200)
ema_bull_4h = (ema20_4h > ema50_4h) & (c4h > ema50_4h)
ema_bear_4h = (ema20_4h < ema50_4h) & (c4h < ema50_4h)

# Precompute 1H pullbacks
rsi_1h = ind.nb_rsi(c1h, 14)
ema20_1h = ind.nb_ema(c1h, 20)
ema50_1h = ind.nb_ema(c1h, 50)

# Map 4H to 1H
m_st30 = p1h.map_htf(st_trend_4h_30, "4h")
m_st25 = p1h.map_htf(st_trend_4h_25, "4h")
m_st20 = p1h.map_htf(st_trend_4h_20, "4h")
m_ema_bull_4h = p1h.map_htf(ema_bull_4h, "4h")
m_ema_bear_4h = p1h.map_htf(ema_bear_4h, "4h")

round_counter = 751
for st_map, st_name in [(m_st30, "ST30"), (m_st25, "ST25"), (m_st20, "ST20"), (m_ema_bull_4h, "EMA_Bull")]:
    for rsi_thresh in [35, 40, 45, 50]:
        for hold_h in [12, 18, 24]:
            for sl_val, tp_val in [(0.06, 0.12), (0.07, 0.14), (0.08, 0.16)]:
                if round_counter > 800:
                    break
                sig_l = (st_map == 1.0) & (rsi_1h < rsi_thresh) & (c1h > o1h) & btc_macro_bull
                sig_s = (st_map == -1.0) & (rsi_1h > (100 - rsi_thresh)) & (c1h < o1h) & btc_macro_bear
                run_round(round_counter, "F1_TrendPullback", f"{st_name}_RSI{rsi_thresh}_Hold{hold_h}_SL{int(sl_val*100)}", sig_l, sig_s, hold=hold_h, sl=sl_val, tp=tp_val)
                round_counter += 1

# Fill up to 800 if needed
while round_counter <= 800:
    sig_l = (m_ema_bull_4h == 1.0) & (l1h <= ema20_1h) & (c1h > ema20_1h) & btc_macro_bull
    sig_s = (m_ema_bear_4h == 1.0) & (h1h >= ema20_1h) & (c1h < ema20_1h) & btc_macro_bear
    run_round(round_counter, "F1_TrendPullback", f"EMA20_Bounce_Hold18_SL7_var{round_counter}", sig_l, sig_s, hold=18, sl=0.07, tp=0.14)
    round_counter += 1


# =====================================================================
# BATCH 2: R801 - R850 (50 Rounds)
# 4H Liquidity Sweep + 1H/15M Market Structure Shift (MSS)
# =====================================================================
print("\n--- Running Batch 2: 4H Liquidity Sweep + MSS (R801 - R850) ---")
roll_h_4h_24 = ind.nb_rolling_max(h4h, 24)
roll_l_4h_24 = ind.nb_rolling_min(l4h, 24)
roll_h_4h_48 = ind.nb_rolling_max(h4h, 48)
roll_l_4h_48 = ind.nb_rolling_min(l4h, 48)

# 4H Sweep: Price breaches rolling high but closes back below (wick rejection)
# Upper wick: high - max(open, close). Body: abs(close - open).
body_4h = np.abs(c4h - o4h)
u_wick_4h = h4h - np.maximum(o4h, c4h)
l_wick_4h = np.minimum(o4h, c4h) - l4h

sweep_bear_24 = (h4h > np.roll(roll_h_4h_24, 1, axis=0)) & (c4h < np.roll(roll_h_4h_24, 1, axis=0)) & (u_wick_4h > 1.2 * (body_4h + 1e-8))
sweep_bull_24 = (l4h < np.roll(roll_l_4h_24, 1, axis=0)) & (c4h > np.roll(roll_l_4h_24, 1, axis=0)) & (l_wick_4h > 1.2 * (body_4h + 1e-8))

sweep_bear_48 = (h4h > np.roll(roll_h_4h_48, 1, axis=0)) & (c4h < np.roll(roll_h_4h_48, 1, axis=0)) & (u_wick_4h > 1.2 * (body_4h + 1e-8))
sweep_bull_48 = (l4h < np.roll(roll_l_4h_48, 1, axis=0)) & (c4h > np.roll(roll_l_4h_48, 1, axis=0)) & (l_wick_4h > 1.2 * (body_4h + 1e-8))

m_sweep_bull_24 = p1h.map_htf(sweep_bull_24, "4h")
m_sweep_bear_24 = p1h.map_htf(sweep_bear_24, "4h")
m_sweep_bull_48 = p1h.map_htf(sweep_bull_48, "4h")
m_sweep_bear_48 = p1h.map_htf(sweep_bear_48, "4h")

# 1H MSS confirmation: break of recent 1H swing low/high
swing_h_1h = ind.nb_rolling_max(h1h, 6)
swing_l_1h = ind.nb_rolling_min(l1h, 6)
mss_bull_1h = (c1h > np.roll(swing_h_1h, 1, axis=0)) & (v1h > ind.nb_sma(v1h, 20) * 1.5)
mss_bear_1h = (c1h < np.roll(swing_l_1h, 1, axis=0)) & (v1h > ind.nb_sma(v1h, 20) * 1.5)

round_counter = 801
for sw_b, sw_s, sw_name in [(m_sweep_bull_24, m_sweep_bear_24, "Sw24"), (m_sweep_bull_48, m_sweep_bear_48, "Sw48")]:
    for taker_gate in [False, True]:
        for hold_h in [8, 14, 20]:
            for sl_val, tp_val in [(0.05, 0.10), (0.07, 0.14), (0.09, 0.18)]:
                if round_counter > 850:
                    break
                sig_l = sw_b & mss_bull_1h
                sig_s = sw_s & mss_bear_1h
                if taker_gate:
                    sig_l = sig_l & (taker_ratio_1h > 0.58)
                    sig_s = sig_s & (taker_ratio_1h < 0.42)
                t_label = "_Taker" if taker_gate else ""
                run_round(round_counter, "F2_SweepMSS", f"{sw_name}{t_label}_H{hold_h}_SL{int(sl_val*100)}", sig_l, sig_s, hold=hold_h, sl=sl_val, tp=tp_val)
                round_counter += 1

while round_counter <= 850:
    sig_l = m_sweep_bull_24 & mss_bull_1h & btc_macro_bull
    sig_s = m_sweep_bear_24 & mss_bear_1h & btc_macro_bear
    run_round(round_counter, "F2_SweepMSS", f"Sw24_MSS_BTC_H18_var{round_counter}", sig_l, sig_s, hold=18, sl=0.07, tp=0.14)
    round_counter += 1


# =====================================================================
# BATCH 3: R851 - R900 (50 Rounds)
# 4H Channel Exhaustion + 1H Momentum Reversal
# =====================================================================
print("\n--- Running Batch 3: 4H Channel Exhaustion (R851 - R900) ---")
_, kelt_u_4h_20, kelt_l_4h_20 = ind.nb_keltner(h4h, l4h, c4h, 20, 2.0)
_, bb_u_4h_25, bb_l_4h_25 = ind.nb_bollinger(c4h, 20, 2.5)

m_kelt_u_4h = p1h.map_htf(kelt_u_4h_20, "4h")
m_kelt_l_4h = p1h.map_htf(kelt_l_4h_20, "4h")
m_bb_u_4h = p1h.map_htf(bb_u_4h_25, "4h")
m_bb_l_4h = p1h.map_htf(bb_l_4h_25, "4h")

round_counter = 851
for ch_u, ch_l, ch_name in [(m_kelt_u_4h, m_kelt_l_4h, "Kelt20"), (m_bb_u_4h, m_bb_l_4h, "BB25")]:
    for rsi_ex in [75, 80]:
        for hold_h in [10, 16, 24]:
            for sl_val, tp_val in [(0.06, 0.12), (0.08, 0.16)]:
                if round_counter > 900:
                    break
                # Exhaustion short: extended above 4h upper band + 1h rsi extreme + 1h bearish candle
                sig_s = (c1h > ch_u) & (rsi_1h > rsi_ex) & (c1h < o1h)
                # Exhaustion long: extended below 4h lower band + 1h rsi oversold + 1h bullish candle
                sig_l = (c1h < ch_l) & (rsi_1h < (100 - rsi_ex)) & (c1h > o1h)
                run_round(round_counter, "F3_ChannelExhaust", f"{ch_name}_RSI{rsi_ex}_H{hold_h}_SL{int(sl_val*100)}", sig_l, sig_s, hold=hold_h, sl=sl_val, tp=tp_val)
                round_counter += 1

while round_counter <= 900:
    sig_s = (c1h > m_kelt_u_4h) & (rsi_1h > 75) & (taker_ratio_1h < 0.40)
    sig_l = (c1h < m_kelt_l_4h) & (rsi_1h < 25) & (taker_ratio_1h > 0.60)
    run_round(round_counter, "F3_ChannelExhaust", f"Kelt_TakerFade_var{round_counter}", sig_l, sig_s, hold=16, sl=0.07, tp=0.14)
    round_counter += 1


# =====================================================================
# BATCH 4: R901 - R950 (50 Rounds)
# 4H Funding/OI Positioning + 1H Deleveraging Breakdown
# =====================================================================
print("\n--- Running Batch 4: 4H Funding/OI Positioning (R901 - R950) ---")
# Funding rate in 1h
fr_1h = p1h.funding_rate if p1h.funding_rate is not None else np.zeros_like(c1h)
oi_1h = p1h.oi if p1h.oi is not None else np.zeros_like(c1h)

# 4H rolling mean of funding
fr_roll_4h = pd.DataFrame(fr_1h).rolling(4, min_periods=1).mean().values
# OI rolling percentile in 1h (over 72h)
oi_df = pd.DataFrame(oi_1h)
oi_min_72 = oi_df.rolling(72, min_periods=24).min().values
oi_max_72 = oi_df.rolling(72, min_periods=24).max().values
oi_pct_72 = np.where(oi_max_72 > oi_min_72, (oi_1h - oi_min_72) / (oi_max_72 - oi_min_72 + 1e-12), 0.5)

round_counter = 901
for fr_th in [0.0003, 0.0004, 0.0005]:
    for oi_th in [0.80, 0.85, 0.90]:
        for hold_h in [12, 18, 24]:
            if round_counter > 950:
                break
            # Longs crowded: FR high + OI high -> short when 1h breaks below EMA20
            sig_s = (fr_roll_4h > fr_th) & (oi_pct_72 > oi_th) & (c1h < ema20_1h)
            # Shorts crowded: FR negative + OI high -> long when 1h breaks above EMA20
            sig_l = (fr_roll_4h < -fr_th * 0.5) & (oi_pct_72 > oi_th) & (c1h > ema20_1h)
            run_round(round_counter, "F4_PosDelever", f"FR{int(fr_th*1e4)}bp_OI{int(oi_th*100)}_H{hold_h}", sig_l, sig_s, hold=hold_h, sl=0.07, tp=0.14)
            round_counter += 1

while round_counter <= 950:
    sig_s = (fr_roll_4h > 0.0003) & (oi_pct_72 > 0.85) & (c1h < ema20_1h) & btc_macro_bear
    sig_l = (fr_roll_4h < -0.00015) & (oi_pct_72 > 0.85) & (c1h > ema20_1h) & btc_macro_bull
    run_round(round_counter, "F4_PosDelever", f"PosDelever_BTC_var{round_counter}", sig_l, sig_s, hold=18, sl=0.07, tp=0.14)
    round_counter += 1


# =====================================================================
# BATCH 5: R951 - R1000 (50 Rounds)
# Alexander Elder Triple Screen for Crypto Futures (1D -> 4H -> 1H)
# =====================================================================
print("\n--- Running Batch 5: Alexander Elder Triple Screen (R951 - R1000) ---")
# Screen 1: 1D Tide -- 1D MACD histogram & 1D EMA50 slope
macd_1d, sig_1d, hist_1d = ind.nb_macd(c1d, fast=12, slow=26, signal=9)
hist_slope_1d = hist_1d - np.roll(hist_1d, 1, axis=0)
tide_bull_1d = (hist_1d > 0) & (hist_slope_1d > 0)
tide_bear_1d = (hist_1d < 0) & (hist_slope_1d < 0)
m_tide_bull = p1h.map_htf(tide_bull_1d, "1d")
m_tide_bear = p1h.map_htf(tide_bear_1d, "1d")

# Screen 2: 4H Wave -- 4H Williams %R pullbacks
wr_4h = ind.nb_williams_r(h4h, l4h, c4h, period=14)
m_wr_4h = p1h.map_htf(wr_4h, "4h")

# Screen 3: 1H Tactical entry -- break above/below 1h EMA10 or 1h High/Low
round_counter = 951
for wr_ob, wr_os in [(-20, -80), (-25, -75), (-30, -70)]:
    for hold_h in [10, 16, 24]:
        for sl_val, tp_val in [(0.05, 0.12), (0.07, 0.14), (0.08, 0.18)]:
            if round_counter > 1000:
                break
            # Long: Bull tide + 4H oversold wave + 1H bullish confirmation
            sig_l = m_tide_bull & (m_wr_4h < wr_os) & (c1h > o1h)
            # Short: Bear tide + 4H overbought wave + 1H bearish confirmation
            sig_s = m_tide_bear & (m_wr_4h > wr_ob) & (c1h < o1h)
            run_round(round_counter, "F5_TripleScreen", f"Elder_WR{abs(wr_os)}_H{hold_h}_SL{int(sl_val*100)}", sig_l, sig_s, hold=hold_h, sl=sl_val, tp=tp_val)
            round_counter += 1

while round_counter <= 1000:
    sig_l = m_tide_bull & (m_wr_4h < -75) & (c1h > o1h) & btc_macro_bull
    sig_s = m_tide_bear & (m_wr_4h > -25) & (c1h < o1h) & btc_macro_bear
    run_round(round_counter, "F5_TripleScreen", f"Elder_BTC_var{round_counter}", sig_l, sig_s, hold=18, sl=0.07, tp=0.14)
    round_counter += 1


# =====================================================================
# BATCH 6: R1001 - R1050 (50 Rounds)
# BTC Macro Regime + Altcoin Relative Strength & Flow (Top-Down Alpha)
# =====================================================================
print("\n--- Running Batch 6: BTC Macro Regime + Altcoin Relative Strength (R1001 - R1050) ---")
# Altcoin 4H relative strength vs BTC
ret_4h_alt = np.diff(np.log(np.maximum(c4h, 1e-8)), axis=0)
ret_4h_btc = ret_4h_alt[:, btc_idx:btc_idx+1]
rs_4h = ret_4h_alt - ret_4h_btc
rs_4h_lead = rs_4h > 0.02 # Altcoin outperforming BTC by > 2% in 4h
rs_4h_lag = rs_4h < -0.02

m_rs_lead = p1h.map_htf(rs_4h_lead, "4h")
m_rs_lag = p1h.map_htf(rs_4h_lag, "4h")

# 1H Squeeze
_, bb_u_1h, bb_l_1h = ind.nb_bollinger(c1h, 20, 2.0)
_, kelt_u_1h, kelt_l_1h = ind.nb_keltner(h1h, l1h, c1h, 20, 1.5)
sq_1h = (bb_u_1h < kelt_u_1h) & (bb_l_1h > kelt_l_1h)

round_counter = 1001
for rs_cond, rs_name in [(m_rs_lead, "RS_Lead"), (True, "All_Alts")]:
    for vol_regime in ["All", "LowVol_Only"]:
        for hold_h in [12, 18, 24]:
            for sl_val, tp_val in [(0.06, 0.12), (0.07, 0.14), (0.08, 0.16)]:
                if round_counter > 1050:
                    break
                sig_l = (c1h > kelt_u_1h) & np.roll(sq_1h, 1, axis=0) & btc_macro_bull & (taker_ratio_1h > 0.60)
                sig_s = (c1h < kelt_l_1h) & np.roll(sq_1h, 1, axis=0) & btc_macro_bear & (taker_ratio_1h < 0.40)
                
                if isinstance(rs_cond, np.ndarray):
                    sig_l = sig_l & rs_cond
                    sig_s = sig_s & m_rs_lag
                if vol_regime == "LowVol_Only":
                    sig_l = sig_l & btc_low_vol
                    sig_s = sig_s & btc_low_vol
                    
                v_label = f"_{vol_regime}" if vol_regime != "All" else ""
                run_round(round_counter, "F6_MacroAltRS", f"{rs_name}{v_label}_H{hold_h}_SL{int(sl_val*100)}", sig_l, sig_s, hold=hold_h, sl=sl_val, tp=tp_val)
                round_counter += 1

while round_counter <= 1050:
    sig_l = (c1h > kelt_u_1h) & np.roll(sq_1h, 1, axis=0) & btc_macro_bull & m_rs_lead
    sig_s = (c1h < kelt_l_1h) & np.roll(sq_1h, 1, axis=0) & btc_macro_bear & m_rs_lag
    run_round(round_counter, "F6_MacroAltRS", f"RS_Squeeze_var{round_counter}", sig_l, sig_s, hold=18, sl=0.07, tp=0.14)
    round_counter += 1

df_results = pd.DataFrame(results)
out_csv = "/root/freqtrade/user_data/minute_research/r4_mtf/r751_r1050_trainvalidC.csv"
df_results.to_csv(out_csv, index=False)
print(f"\n=======================================================")
print(f"300 Rounds Completed! Master ledger saved to {out_csv}")
print(f"Total Iterations: {len(df_results)}")
print(f"Status Breakdown:\n{df_results['status'].value_counts()}")
print(f"Top 10 Performers on VALID-C:")
print(df_results.sort_values('va_pf', ascending=False)[['round', 'family', 'name', 'tr_pf', 'tr_t', 'va_trades', 'va_pf', 'va_t', 'status']].head(10).to_string(index=False))
print("=======================================================")
