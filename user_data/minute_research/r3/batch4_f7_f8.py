"""Batch 4: Family 7 (R171-R190: Price Action & Swings) & Family 8 (R191-R210: Cross-Sectional Portfolios)."""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append("/root/freqtrade/user_data/minute_research/r3")
from engine200 import PanelData, evaluate_strategy
import indicators as ind


def run_batch4():
    print("Loading 1h, 4h, 1d panels for Batch 4...")
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
    qv1h = np.array(p1h.quote_volume) if p1h.quote_volume is not None else v1h * c1h
    fr1h = np.array(p1h.funding_rate) if p1h.funding_rate is not None else np.zeros_like(c1h)
    oi1h = np.array(p1h.oi) if p1h.oi is not None else np.zeros_like(c1h)
    tt_ls1h = np.array(p1h.toptrader_ls) if p1h.toptrader_ls is not None else np.ones_like(c1h)
    
    c4h = np.array(p4h.close)
    h4h = np.array(p4h.high)
    l4h = np.array(p4h.low)
    
    c1d = np.array(p1d.close)
    
    print("Pre-computing multi-timeframe indicators for Batch 4...")
    ema20_1h = ind.nb_ema(c1h, 20)
    ema50_1h = ind.nb_ema(c1h, 50)
    atr14_1h = ind.nb_atr(h1h, l1h, c1h, 14)
    rsi14_1h = ind.nb_rsi(c1h, 14)
    don_u_24, don_l_24 = ind.nb_donchian(h1h, l1h, 24)
    don_u_48, don_l_48 = ind.nb_donchian(h1h, l1h, 48)
    
    # 4h indicators mapped causally
    ema50_4h = ind.nb_ema(c4h, 50)
    ema100_4h = ind.nb_ema(c4h, 100)
    m_ema50_4h = p1h.map_htf_to_ltf(ema50_4h, "4h")
    m_ema100_4h = p1h.map_htf_to_ltf(ema100_4h, "4h")
    m_c4h = p1h.map_htf_to_ltf(c4h, "4h")
    
    st_trend_4h, _ = ind.nb_supertrend(h4h, l4h, c4h, 10, 3.0)
    m_st_trend_4h = p1h.map_htf_to_ltf(st_trend_4h.astype(np.float64), "4h")
    
    # 1d indicators mapped causally
    sma100_1d = ind.nb_sma(c1d, 100)
    m_sma100_1d = p1h.map_htf_to_ltf(sma100_1d, "1d")
    m_c1d = p1h.map_htf_to_ltf(c1d, "1d")
    
    results = []
    T, N = c1h.shape
    empty_sig = np.zeros((T, N), dtype=bool)
    c_prev = np.roll(c1h, 1, axis=0)

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
        print(f"[{r_id:03d}] {name[:35]:35s} | n={row['n']:5d} | mean={row['mean_bp']:6.1f}bp | pf={row['pf']:5.2f} | day_t={row['day_t']:5.2f} |/d={row['per_day']:4.1f} | {status}")
        return row

    print("Executing Batch 4 (R171-R210) on TRAIN...")
    
    # --- FAMILY 7: Price Action, Support/Resistance & Swings (R171 - R190) ---
    # R171: Multi-Timeframe Fair Value Gap (FVG) Retest with 4h Trend
    # Bullish FVG: Low of bar i > High of bar i-2
    fvg_bull = np.roll(l1h, 1, axis=0) > np.roll(h1h, 3, axis=0)
    # Retest: current low dips into the gap area
    fvg_top = np.roll(l1h, 1, axis=0)
    fvg_bot = np.roll(h1h, 3, axis=0)
    retest_bull = fvg_bull & (l1h <= fvg_top) & (c1h >= fvg_bot) & (c1h > o1h)
    sig_l = retest_bull & (m_c4h > m_ema50_4h)
    test_round(171, "MTF Fair Value Gap (FVG) Retest", sig_l, empty_sig, hold=18)
    
    # R172: Break of Structure (BOS) on 1h (Close breaks prior 24h high)
    sig_l = (c_prev <= don_u_24) & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = (c_prev >= don_l_24) & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(172, "Break of Structure (BOS) + 4h", sig_l, sig_s, hold=24)
    
    # R173: Liquidity Sweep / Judas Swing (Sweep 24h high/low and immediately close back inside)
    sweep_high = (h1h > don_u_24) & (c1h < don_u_24) & (c1h < o1h)
    sweep_low = (l1h < don_l_24) & (c1h > don_l_24) & (c1h > o1h)
    sig_s = sweep_high & (m_c4h < m_ema50_4h)
    sig_l = sweep_low & (m_c4h > m_ema50_4h)
    test_round(173, "Liquidity Sweep Judas Swing", sig_l, sig_s, hold=18)
    
    # R174: Previous Day High/Low Sweep Rejection (PDH/PDL proxy: 24h extremes)
    sig_s = sweep_high
    sig_l = sweep_low
    test_round(174, "PDH/PDL Sweep Reversal", sig_l, sig_s, hold=12)
    
    # R175: Equal Highs / Equal Lows (EQH/EQL) Liquidity Grab Continuation
    # Double top within 0.2% followed by clean breakout
    dt = np.abs(np.roll(h1h, 12, axis=0) - don_u_24) / don_u_24 < 0.003
    sig_l = dt & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    test_round(175, "EQH Liquidity Grab Breakout", sig_l, empty_sig, hold=24)
    
    # R176: Asian Session High/Low deviation and reclaim (00-08 UTC range sweep during London 08-16 UTC)
    # Using UTC hours from index
    hours = pd.to_datetime(p1h.dates).hour.values
    is_london = (hours >= 8) & (hours < 16)
    is_london_2d = np.repeat(is_london[:, None], N, axis=1)
    sig_l = is_london_2d & sweep_low & (m_c4h > m_ema50_4h)
    sig_s = is_london_2d & sweep_high & (m_c4h < m_ema50_4h)
    test_round(176, "London Session Sweep Reclaim", sig_l, sig_s, hold=12)
    
    # R177: Inside Bar 1h Breakout with 4h Trend
    inside_bar = (np.roll(h1h, 1, axis=0) < np.roll(h1h, 2, axis=0)) & (np.roll(l1h, 1, axis=0) > np.roll(l1h, 2, axis=0))
    sig_l = inside_bar & (c1h > np.roll(h1h, 1, axis=0)) & (m_c4h > m_ema50_4h)
    sig_s = inside_bar & (c1h < np.roll(l1h, 1, axis=0)) & (m_c4h < m_ema50_4h)
    test_round(177, "Inside Bar Breakout + 4h Trend", sig_l, sig_s, hold=18)
    
    # R178: Bullish / Bearish Engulfing with 4h Trend
    engulf_bull = (c_prev < np.roll(o1h, 1, axis=0)) & (c1h > o1h) & (c1h > np.roll(o1h, 1, axis=0)) & (o1h < c_prev)
    engulf_bear = (c_prev > np.roll(o1h, 1, axis=0)) & (c1h < o1h) & (c1h < np.roll(o1h, 1, axis=0)) & (o1h > c_prev)
    sig_l = engulf_bull & (m_c4h > m_ema50_4h)
    sig_s = engulf_bear & (m_c4h < m_ema50_4h)
    test_round(178, "Engulfing Bar + 4h Trend", sig_l, sig_s, hold=18)
    
    # R179: 3-Bar Pin Bar Reversal at 4h Support
    lower_wick = np.minimum(o1h, c1h) - l1h
    candle_body = np.maximum(np.abs(c1h - o1h), 1e-8)
    pin_bar = (lower_wick > candle_body * 2.5) & (l1h < ema50_1h) & (c1h > ema50_1h)
    sig_l = pin_bar & (m_c4h > m_ema50_4h)
    test_round(179, "Pin Bar Rejection at 50 EMA", sig_l, empty_sig, hold=18)
    
    # R180: Swing Failure Pattern (SFP) at 48h High/Low
    sfp_high = (h1h > don_u_48) & (c1h < don_u_48) & (c1h < o1h)
    sfp_low = (l1h < don_l_48) & (c1h > don_l_48) & (c1h > o1h)
    sig_s = sfp_high & (m_c4h < m_ema50_4h)
    sig_l = sfp_low & (m_c4h > m_ema50_4h)
    test_round(180, "Swing Failure Pattern (SFP) 48h", sig_l, sig_s, hold=24)
    
    # R181: Narrow Range 4 (NR4) Breakout with 4h Trend
    candle_range = h1h - l1h
    range_min4 = ind.nb_donchian(candle_range, candle_range, 4)[1]
    nr4 = np.roll(candle_range, 1, axis=0) <= np.roll(range_min4, 1, axis=0) * 1.02
    sig_l = nr4 & (c1h > np.roll(h1h, 1, axis=0)) & (m_c4h > m_ema50_4h)
    sig_s = nr4 & (c1h < np.roll(l1h, 1, axis=0)) & (m_c4h < m_ema50_4h)
    test_round(181, "NR4 Volatility Expansion Break", sig_l, sig_s, hold=18)
    
    # R182: Wide Range Expansion Bar Follow-Through (Range > 2.5x ATR, close near extreme)
    wr_up = (candle_range > atr14_1h * 2.5) & (c1h > o1h) & ((h1h - c1h) < candle_range * 0.15)
    wr_dn = (candle_range > atr14_1h * 2.5) & (c1h < o1h) & ((c1h - l1h) < candle_range * 0.15)
    sig_l = wr_up & (m_c4h > m_ema50_4h)
    sig_s = wr_dn & (m_c4h < m_ema50_4h)
    test_round(182, "Wide Range Bar Follow Through", sig_l, sig_s, hold=18)
    
    # R183: Consolidation Box Breakout (Donchian width < 3% for 24h, then breaks out)
    don_width = np.where(don_l_24 > 0, (don_u_24 - don_l_24) / don_l_24, 0.0)
    tight_box = don_width < 0.03
    sig_l = np.roll(tight_box, 1, axis=0) & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = np.roll(tight_box, 1, axis=0) & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(183, "Tight Box (<3%) Breakout", sig_l, sig_s, hold=24)
    
    # R184: Consecutive Higher Lows Continuation (3 consecutive 4h bars making higher lows)
    hl3 = (l4h > np.roll(l4h, 1, axis=0)) & (np.roll(l4h, 1, axis=0) > np.roll(l4h, 2, axis=0))
    m_hl3 = p1h.map_htf_to_ltf(hl3.astype(np.float64), "4h")
    sig_l = (m_hl3 == 1.0) & (c_prev < ema20_1h) & (c1h > ema20_1h)
    test_round(184, "3-Bar Higher Lows 4h Trend", sig_l, empty_sig, hold=24)
    
    # R185: Price Action Channel (PAC: High 20 EMA, Low 20 EMA) Breakout
    ema20_h = ind.nb_ema(h1h, 20)
    ema20_l = ind.nb_ema(l1h, 20)
    sig_l = (c1h > ema20_h) & (c_prev <= ema20_h) & (m_c4h > m_ema50_4h)
    sig_s = (c1h < ema20_l) & (c_prev >= ema20_l) & (m_c4h < m_ema50_4h)
    test_round(185, "Price Action Channel (PAC) Break", sig_l, sig_s, hold=24)
    
    # R186: Wick Exhaustion Fade (Wick > 70% of total bar range)
    upper_wick = h1h - np.maximum(o1h, c1h)
    sig_s = (upper_wick > candle_range * 0.7) & (candle_range > atr14_1h * 1.5)
    sig_l = (lower_wick > candle_range * 0.7) & (candle_range > atr14_1h * 1.5)
    test_round(186, "Extreme Wick Exhaustion Fade", sig_l, sig_s, hold=12)
    
    # R187: Symmetrical Compression Triangle Breakout
    # Highs making lower highs while lows making higher lows
    comp = (don_u_24 < np.roll(don_u_24, 12, axis=0)) & (don_l_24 > np.roll(don_l_24, 12, axis=0))
    sig_l = comp & (c1h > don_u_24) & (m_c4h > m_ema50_4h)
    sig_s = comp & (c1h < don_l_24) & (m_c4h < m_ema50_4h)
    test_round(187, "Triangle Compression Breakout", sig_l, sig_s, hold=24)
    
    # R188: FVG + Donchian Break Confluence
    sig_l = retest_bull & (c1h > don_u_24)
    test_round(188, "FVG + Donchian Confluence", sig_l, empty_sig, hold=24)
    
    # R189: SFP Reversal with Dynamic ATR Stop/TP
    test_round(189, "SFP Reversal + Dynamic SL/TP", sig_l, sig_s, hold=24, sl=0.025, tp=0.05)
    
    # R190: 24h Breakout + SuperTrend 4h + Trailing Stop
    sig_l = (c1h > don_u_24) & (m_st_trend_4h == 1)
    sig_s = (c1h < don_l_24) & (m_st_trend_4h == -1)
    test_round(190, "24h Break + 4h ST + SL/TP", sig_l, sig_s, hold=36, sl=0.03, tp=0.06)

    
    # --- FAMILY 8: Cross-Sectional Factor Portfolios (R191 - R210) ---
    # R191: Cross-Sectional 24h Momentum (Long Top 10% highest 24h return, Short Bottom 10%)
    ret24 = (c1h - np.roll(c1h, 24, axis=0)) / np.roll(c1h, 24, axis=0)
    # At each bar, rank across universe
    sig_l = np.zeros_like(c1h, dtype=bool)
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24): # Rebalance every 24h
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(ret24[i, j])]
        if len(valid_coins) >= 20:
            scores = [(ret24[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            n_sel = max(len(valid_coins) // 10, 3)
            for _, j in scores[-n_sel:]:
                sig_l[i, j] = True
            for _, j in scores[:n_sel]:
                sig_s[i, j] = True
    test_round(191, "XS 24h Momentum Top/Bottom 10%", sig_l, sig_s, hold=24)
    
    # R192: Cross-Sectional 7d Momentum with 24h Reversal Filter (Long 7d winners that dipped in 24h)
    ret168 = (c1h - np.roll(c1h, 168, axis=0)) / np.roll(c1h, 168, axis=0)
    sig_l = np.zeros_like(c1h, dtype=bool)
    for i in range(168, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(ret168[i, j]) and not np.isnan(ret24[i, j])]
        if len(valid_coins) >= 20:
            # 7d winners (top 20%) that pulled back in 24h (ret24 < 0)
            scores = [(ret168[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            top_coins = [j for _, j in scores[-int(len(valid_coins) * 0.25):] if ret24[i, j] < -0.01]
            for j in top_coins[:5]:
                sig_l[i, j] = True
    test_round(192, "XS 7d Mom Leader Dip Buyer", sig_l, empty_sig, hold=24)
    
    # R193: Cross-Sectional Quote Volume Share Surge
    qv_share = np.zeros_like(qv1h)
    tot_qv = np.nansum(qv1h, axis=1, keepdims=True)
    qv_share = np.where(tot_qv > 0, qv1h / tot_qv, 0.0)
    qv_share_ma = ind.nb_sma(qv_share, 168)
    qv_surge_ratio = np.where(qv_share_ma > 0, qv_share / qv_share_ma, 1.0)
    sig_l = np.zeros_like(c1h, dtype=bool)
    for i in range(168, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(qv_surge_ratio[i, j])]
        if len(valid_coins) >= 20:
            scores = [(qv_surge_ratio[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            for _, j in scores[-5:]:
                sig_l[i, j] = True
    test_round(193, "XS Volume Share Surge Leaders", sig_l, empty_sig, hold=24)
    
    # R194: Cross-Sectional Volatility-Adjusted Momentum (Return / ATR)
    mom_vol_adj = np.where(atr14_1h > 0, ret24 / (atr14_1h / c1h), 0.0)
    sig_l = np.zeros_like(c1h, dtype=bool)
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(mom_vol_adj[i, j])]
        if len(valid_coins) >= 20:
            scores = [(mom_vol_adj[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            n_sel = max(len(valid_coins) // 10, 3)
            for _, j in scores[-n_sel:]:
                sig_l[i, j] = True
            for _, j in scores[:n_sel]:
                sig_s[i, j] = True
    test_round(194, "XS Vol-Adjusted Momentum Spread", sig_l, sig_s, hold=24)
    
    # R195: Cross-Sectional Top-Trader Long/Short Ratio Spread
    sig_l = np.zeros_like(c1h, dtype=bool)
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(tt_ls1h[i, j])]
        if len(valid_coins) >= 20:
            scores = [(tt_ls1h[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            n_sel = max(len(valid_coins) // 10, 3)
            for _, j in scores[-n_sel:]:
                sig_l[i, j] = True
            for _, j in scores[:n_sel]:
                sig_s[i, j] = True
    test_round(195, "XS Top-Trader LS Factor Spread", sig_l, sig_s, hold=24)
    
    # R196: Cross-Sectional Funding Rate Carry Factor
    sig_l = np.zeros_like(c1h, dtype=bool)
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(fr1h[i, j])]
        if len(valid_coins) >= 20:
            scores = [(fr1h[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            n_sel = max(len(valid_coins) // 10, 3)
            for _, j in scores[:n_sel]: # Lowest funding -> Long
                sig_l[i, j] = True
            for _, j in scores[-n_sel:]: # Highest funding -> Short
                sig_s[i, j] = True
    test_round(196, "XS Funding Rate Carry Spread", sig_l, sig_s, hold=24)
    
    # R197: Cross-Sectional Low-Volatility Anomaly
    natr = np.where(c1h > 0, atr14_1h / c1h, 0.0)
    sig_l = np.zeros_like(c1h, dtype=bool)
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(natr[i, j])]
        if len(valid_coins) >= 20:
            scores = [(natr[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            n_sel = max(len(valid_coins) // 10, 3)
            for _, j in scores[:n_sel]: # Lowest vol -> Long
                sig_l[i, j] = True
            for _, j in scores[-n_sel:]: # Highest vol -> Short
                sig_s[i, j] = True
    test_round(197, "XS Low-Volatility Anomaly", sig_l, sig_s, hold=24)
    
    # R198: Cross-Sectional OI Growth Factor (Long highest 24h OI growth, Short lowest)
    oi_prev24 = np.roll(oi1h, 24, axis=0)
    oi_pct_chg24 = np.where(oi_prev24 > 0, (oi1h - oi_prev24) / oi_prev24, 0.0)
    sig_l = np.zeros_like(c1h, dtype=bool)
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(oi_pct_chg24[i, j])]
        if len(valid_coins) >= 20:
            scores = [(oi_pct_chg24[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            n_sel = max(len(valid_coins) // 10, 3)
            for _, j in scores[-n_sel:]:
                sig_l[i, j] = True
            for _, j in scores[:n_sel]:
                sig_s[i, j] = True
    test_round(198, "XS OI 24h Growth Spread", sig_l, sig_s, hold=24)
    
    # R199: Cross-Sectional Mean Reversion Extreme (Long biggest 24h losers, Short biggest winners)
    sig_l = np.zeros_like(c1h, dtype=bool)
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(ret24[i, j])]
        if len(valid_coins) >= 20:
            scores = [(ret24[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            n_sel = max(len(valid_coins) // 10, 3)
            for _, j in scores[:n_sel]: # Losers -> Long
                sig_l[i, j] = True
            for _, j in scores[-n_sel:]: # Winners -> Short
                sig_s[i, j] = True
    test_round(199, "XS 24h Mean Reversion Spread", sig_l, sig_s, hold=24)
    
    # R200: Cross-Sectional Acceleration Factor (Second derivative of price)
    ret12 = (c1h - np.roll(c1h, 12, axis=0)) / np.roll(c1h, 12, axis=0)
    ret12_prev = np.roll(ret12, 12, axis=0)
    accel = ret12 - ret12_prev
    sig_l = np.zeros_like(c1h, dtype=bool)
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(accel[i, j])]
        if len(valid_coins) >= 20:
            scores = [(accel[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            n_sel = max(len(valid_coins) // 10, 3)
            for _, j in scores[-n_sel:]:
                sig_l[i, j] = True
            for _, j in scores[:n_sel]:
                sig_s[i, j] = True
    test_round(200, "XS Price Acceleration Factor", sig_l, sig_s, hold=24)
    
    # R201: Cross-Sectional Multi-Factor (Mom + Low FR + Rising OI)
    score_mf = np.zeros_like(c1h)
    for j in range(N):
        score_mf[:, j] = ret24[:, j] - fr1h[:, j] * 100.0 + oi_pct_chg24[:, j] * 0.5
    sig_l = np.zeros_like(c1h, dtype=bool)
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(score_mf[i, j])]
        if len(valid_coins) >= 20:
            scores = [(score_mf[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            n_sel = max(len(valid_coins) // 10, 3)
            for _, j in scores[-n_sel:]:
                sig_l[i, j] = True
            for _, j in scores[:n_sel]:
                sig_s[i, j] = True
    test_round(201, "XS Multi-Factor Composite Spread", sig_l, sig_s, hold=24)
    
    # R202: Cross-Sectional Long-Only Momentum in Macro Bull
    sig_l = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(ret24[i, j])]
        if len(valid_coins) >= 20 and m_c1d[i, p1h.u162_indices[0]] > m_sma100_1d[i, p1h.u162_indices[0]]:
            scores = [(ret24[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            for _, j in scores[-5:]:
                sig_l[i, j] = True
    test_round(202, "XS Long-Only Mom in 1d Bull", sig_l, empty_sig, hold=24)
    
    # R203: Cross-Sectional Short-Only Laggards in Macro Bear
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(ret24[i, j])]
        if len(valid_coins) >= 20 and m_c1d[i, p1h.u162_indices[0]] < m_sma100_1d[i, p1h.u162_indices[0]]:
            scores = [(ret24[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            for _, j in scores[:5]:
                sig_s[i, j] = True
    test_round(203, "XS Short-Only Laggards in 1d Bear", empty_sig, sig_s, hold=24)
    
    # R204: Cross-Sectional 48h Momentum with 48h Hold
    ret48 = (c1h - np.roll(c1h, 48, axis=0)) / np.roll(c1h, 48, axis=0)
    sig_l = np.zeros_like(c1h, dtype=bool)
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(48, T, 48):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(ret48[i, j])]
        if len(valid_coins) >= 20:
            scores = [(ret48[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            n_sel = max(len(valid_coins) // 10, 3)
            for _, j in scores[-n_sel:]:
                sig_l[i, j] = True
            for _, j in scores[:n_sel]:
                sig_s[i, j] = True
    test_round(204, "XS 48h Momentum 48h Rebalance", sig_l, sig_s, hold=48)
    
    # R205: Cross-Sectional High Turnover Velocity
    turnover = np.where(oi1h > 0, qv1h / oi1h, 0.0)
    sig_l = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(turnover[i, j])]
        if len(valid_coins) >= 20:
            scores = [(turnover[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            for _, j in scores[-5:]:
                sig_l[i, j] = True
    test_round(205, "XS Turnover Velocity Factor", sig_l, empty_sig, hold=24)
    
    # R206: Cross-Sectional 7d Low-Beta Safe Havens
    sig_l = np.zeros_like(c1h, dtype=bool)
    for i in range(168, T, 168):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(natr[i, j])]
        if len(valid_coins) >= 20:
            scores = [(natr[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            for _, j in scores[:5]:
                sig_l[i, j] = True
    test_round(206, "XS Low-Beta 7d Safe Havens", sig_l, empty_sig, hold=168)
    
    # R207: Cross-Sectional High-Beta Catch-up Trade
    sig_l = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(natr[i, j]) and not np.isnan(ret24[i, j])]
        if len(valid_coins) >= 20:
            # High beta coins that have not moved yet
            scores = [(natr[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            high_beta_laggards = [j for _, j in scores[-int(len(valid_coins) * 0.3):] if abs(ret24[i, j]) < 0.015]
            for j in high_beta_laggards[:5]:
                sig_l[i, j] = True
    test_round(207, "XS High-Beta Laggard Catch-Up", sig_l, empty_sig, hold=24)
    
    # R208: Cross-Sectional Momentum with Dynamic SL/TP (3% SL, 6% TP)
    test_round(208, "XS 24h Mom + SL/TP (3:6)", sig_l, sig_s, hold=48, sl=0.03, tp=0.06)
    
    # R209: Cross-Sectional Quintile Spread Trading (Top 20% vs Bottom 20%)
    sig_l = np.zeros_like(c1h, dtype=bool)
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(ret24[i, j])]
        if len(valid_coins) >= 20:
            scores = [(ret24[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            q = len(valid_coins) // 5
            for _, j in scores[-q:]:
                sig_l[i, j] = True
            for _, j in scores[:q]:
                sig_s[i, j] = True
    test_round(209, "XS Quintile Momentum Spread", sig_l, sig_s, hold=24)
    
    # R210: Cross-Sectional Factor Tilt with 4h BTC Filter
    sig_l = np.zeros_like(c1h, dtype=bool)
    sig_s = np.zeros_like(c1h, dtype=bool)
    for i in range(24, T, 24):
        valid_coins = [j for j in p1h.u162_indices if not np.isnan(ret24[i, j])]
        if len(valid_coins) >= 20:
            scores = [(ret24[i, j], j) for j in valid_coins]
            scores.sort(key=lambda x: x[0])
            n_sel = max(len(valid_coins) // 10, 3)
            # Only long winners if BTC > 50 EMA, only short losers if BTC < 50 EMA
            btc_col = p1h.u162_indices[0]
            if m_c4h[i, btc_col] > m_ema50_4h[i, btc_col]:
                for _, j in scores[-n_sel:]:
                    sig_l[i, j] = True
            else:
                for _, j in scores[:n_sel]:
                    sig_s[i, j] = True
    test_round(210, "XS Mom Filtered by BTC 4h Trend", sig_l, sig_s, hold=24)
    
    df_res = pd.DataFrame(results)
    seg = os.environ.get("RUN_SEG", "TRAIN")
    out_csv = "/root/freqtrade/user_data/minute_research/r3/batch4_results.csv" if seg == "TRAIN" else f"/root/freqtrade/user_data/minute_research/r3/batch4_valid_results.csv"
    df_res.to_csv(out_csv, index=False)
    print("\nBatch 4 completed! Saved to batch4_results.csv")


if __name__ == "__main__":
    run_batch4()
