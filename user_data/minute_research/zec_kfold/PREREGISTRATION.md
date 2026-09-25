# ZEC single-coin, relative-threshold, time-series k-fold screen (pre-registered 2026-09-25, session freqtrade-1c)

Written before any result of this study was computed. ZEC was never in U162, so no earlier strategy selection used
its data; its HOLDOUT (2026-03-01..08-31) is unread.

## Data and fills
ZECUSDT perpetual, 1h bars built from Binance Vision 1m klines (with taker buy volume and trade count); funding
(8h settlements) charged on positions open across a settlement; mark price (basis); 5m metrics (OI, long/short ratios,
last value per hour); BTCUSDT 1h. Signal on bar close, entry next bar open, exit at the open `hold` bars later,
one position at a time, cost 10 bp per side (15 bp stress). No stop / take-profit.

## Families (score s_t on the closed bar; long when high, short when low unless marked; hold in hours)
F01 mom24 ret24 (12) · F02 rev24 −ret24 (12) · F03 mom4 ret4 (6) · F04 rev1 −ret1 (4) · F05 rsi_rev −(RSI14−50) (12) ·
F06 bb_rev −(%B−0.5) (12) · F07 donchian break of prior 24h high/low in ATR units (12) · F08 squeeze_break
(close−Keltner mid)/ATR when prev bar in BB-inside-Keltner squeeze (18) · F09 vol_follow sign(bar)·vol/SMA24 (12) ·
F10 vol_fade (12) · F11 taker_follow taker ratio−0.5 (12) · F12 taker_fade (12) · F13 cvd_div z(12h taker delta)−z(ret12)
(12) · F14 flushout −ret12 when green bar and vol > 1.5×SMA24, long only (12) · F15 fund_fade −last funding (8) ·
F16 fund_follow +last funding (8) · F17 fund3_exh −mean of last 3 settlements (8) · F18 oi_follow sign(ret4)·ΔOI4 (12) ·
F19 oi_flush −ΔOI12 when ret12 < 0, long only (12) · F20 btc_lead BTC ret1 (4) · F21 rs_follow ret24−BTC ret24 (12) ·
F22 rs_fade (12) · F23 atr_break sign(bar)·ATR14/SMA48(ATR14) (12) · F24 trend_pullback (50−RSI) above EMA200 /
(50−RSI) below EMA200 (12) · F25 basis_fade −(close/mark−1) (8) · F26 count_follow sign(bar)·count/SMA24 (12) ·
F27 toptrader_follow Δ12 top-trader long/short ratio (12) · F28 crowd_fade −(retail long/short ratio − trailing median) (12).

## Relative thresholds (the only tuned parameter)
Trailing 90-day (2160 bars, min 240) quantile of the non-zero score, lagged one bar. Long if s ≥ Q_q, short if s ≤ Q_{1−q}.
Strictness q ∈ {0.80, 0.90, 0.95, 0.98}. 28 × 4 = 112 combinations.

## Folds (TRAIN + VALID-C only; 2025-10..11 excluded; HOLDOUT untouched)
F1 2025-01..02, F2 03..04, F3 05..06, F4 07..08, F5 2025-09 + 2025-12, F6 2026-01..02 (trade assigned by entry time).
Thresholds are causal rolling quantiles, so no per-fold fitting and no purging needed; folds measure stability and
feed CSCV.

## Selection (all must hold, pooled over the six folds)
net mean > 0, PF ≥ 1.2, day-clustered t ≥ 2.0, ≥ 4/6 folds positive, ≥ 0.4 trades/day (1/day reported as standard),
mean > 0 at 15 bp/side, and mean > 0 after subtracting ZEC's unconditional same-side, same-hold drift in the fold.
Overfitting: CSCV over the 20 splits of 6 folds into 3 + 3, metric = daily Sharpe; report PBO for the 112-combo set.
At most three survivors (distinct families, highest t) are read once on HOLDOUT, reported raw and drift-adjusted.

## Screen result (2026-09-25, before HOLDOUT)
112 combos, PBO 0.25. One survivor: F11_taker_follow q 0.95 hold 12 — 372 trades (1.02/day), +44 bp, PF 1.39, t 2.24,
4/6 folds, +34 bp at 15 bp/side, drift-adjusted +41 bp, long 194 / short 178. Plateau (q 0.90–0.98 × hold 10/12/14):
min PF 1.00 within ±20% (q 0.94–0.96, hold 10–14) but hold 10 is ≈ 0 everywhere and fold F3 is negative in almost all cells —
fragile. HOLDOUT read once now for F11_taker_follow:0.95.

## HOLDOUT result (read once, 2026-09-25)
F11_taker_follow q 0.95 hold 12 on 2026-03-01..08-31: 177 trades (0.96/day, long 93 / short 84), +22.7 bp, PF 1.18, day t 0.73;
at 15 bp/side +12.7 bp, PF 1.09, t 0.41; drift-adjusted +20.3 bp. Long leg +44 bp, short leg −1 bp. Monthly sum: Mar +11.8%,
Apr +3.7%, May −6.2%, Jun +37.5%, Jul +1.8%, Aug −8.4% — one month carries the result.
**Verdict: fails (PF < 1.2, t far below 2). No strategy passes this pre-registered ZEC study.**

---
# Addendum (2026-09-25, before any run): shorter timeframes
Timeframes 1m / 5m / 15m / 30m. Same 28 family formulas with the same window lengths **in bars**. Two hold variants:
`bars` (the 1h hold as a bar count, e.g. 12 bars) and `clock` (the 1h hold in hours, e.g. 12 h). Strictness chosen per
timeframe so the expected signal rate is ≈ {0.5, 1, 2, 4} per day: q = 1 − target / bars_per_day. Rolling quantile window
90 days (min 10 days). Funding charged at settlement bars; OI / long-short metrics forward-filled to bar closes
(5m snapshots). Folds, gates, cost and CSCV exactly as above, PBO reported per timeframe. At most three survivors in total
(distinct family × timeframe, highest t) read once on HOLDOUT.

## Selection-rule change requested by the user (2026-09-25, after the 30m/15m/5m screens, before any HOLDOUT read of them)
User: frequency should not be tuned to ≈1/day; it only must not be too low; prefer higher total return and, at similar
total return, higher frequency. New ranking among combos passing all significance gates above (t ≥ 2.0, PF ≥ 1.2, ≥ 4/6
folds, 15 bp stress > 0, drift-adjusted > 0) with the unchanged floor ≥ 0.4 trades/day: sort by pooled total return
(sum of per-trade returns, full stake, non-compounded) over the six folds, ties by frequency. Top three distinct
family × timeframe go to HOLDOUT once. Observed before this change: for F25 basis_fade the edge exists only in the extreme
tail (q ≥ 0.997 on 5m); looser thresholds lose money, so total return does not rise with frequency there.
- 2026-09-25T14:02:58Z HOLDOUT read once now for: 5m F25_basis_fade|clock q=0.996528 hold 96; 1m F27_toptrader_follow|clock q=0.996528 hold 720 (plateau: 5m 9/9 cells PF 1.53-2.43 t 2.15-3.09; 1m 12/12 PF 1.07-1.60, t mostly <2, 1m PBO 0.65).
- 2026-09-25T14:03:25Z ERROR disclosed: the 1m HOLDOUT read above used q=0.999306 (wrong; the registered candidate is q=0.997222, target 4/day). That extra read gave 14 trades, +227 bp, t 1.26. Reading the registered q=0.997222 once now.

## HOLDOUT results for the shorter-timeframe candidates (read once each; plus the disclosed wrong-q read above)
| candidate | trades/day | mean | PF | day t | total | drift-adj | long / short | verdict |
|---|---|---|---|---|---|---|---|---|
| 5m F25 basis_fade clock q 0.996528 hold 8h | 138 / 0.75 | +23.5 bp (15 bp: +13.5) | 1.17 | 0.63 | +32.4% | +22.1 bp | +16 / +31 bp | FAIL (PF < 1.2, t) |
| 1m F27 toptrader_follow clock q 0.997222 hold 12h | 56 / 0.30 | +101.7 bp (15 bp: +91.7) | 1.66 | 1.64 | +57.0% | +93.3 bp | +147 / +37 bp | FAIL (t < 2, 0.30/day < 0.4) — closest |
1m F27 monthly: Mar +31.1%, Apr +7.3%, May +22.5%, Jun −2.2%, Jul −6.1%, Aug +4.3%. 1m screen PBO 0.65.
No combination passes the pre-registered ZEC study at any timeframe (1m/5m/15m/30m/1h).
