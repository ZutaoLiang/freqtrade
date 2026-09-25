# AllWeatherRegimeAdaptive review and V2 (session freqtrade-1c, 2026-09-25)

## Review of the original (commit 4a07b8a83), freqtrade on user_data/data/r3b, 100 USDT, 10 slots
Numbers in skills/fable/allweather-regime-adaptive-dryrun.md reproduce. At 0.06%/side: TRAIN +138.7% PF 1.27 t 2.31;
VALID-C +65.0% PF 1.62 t 2.15 (one day = 70% of profit); HOLDOUT +14.0% PF 1.07 t −0.30 (SWARMS 58%, one day 88%).
At 0.10%/side HOLDOUT +4.1% PF 1.02; ex-SWARMS negative. VALID+HOLDOUT t 1.25 (0.99 at 0.10%).
Implementation defects: R24 engine enters 1h after settlement (validated R24: T+5 min) and gets a TP; global
minimal_roi 14% pre-empts the per-tag TPs; 1.5x leverage turns every margin-based stop/TP into price/1.5;
custom_exit stops only at the 1h close; flushout engine lacks the OI condition it was researched with.
HOLDOUT was already used to design this system (family HOLDOUT medians in r8 REPORT) — it is not blind.

## V2 pre-registration (written before any V2 run)
Fix implementation only; every engine threshold, stop, TP and hold is unchanged from the original.
- timeframe 5m; 1h/4h/1d indicators via merge_informative_pair; 1h-engine signals only on the 5m candle that
  completes the hour (entry at the next hour open, as in the 1h original).
- leverage 1; all stops/TPs are price moves: dual_sq −7%/+14%/18h, flushout −6%/+10%/12h,
  exhaust_short = exact R24 (5m candle at settlement T, entry T+5 min, 8h, no TP, −50% disaster stop).
- stops via custom_stoploss (intrabar in backtest), TPs and time exits via custom_exit; minimal_roi disabled.
- universe U162 static (as backtested), 100 USDT wallet, stake unlimited, max_open_trades 10, fee 0.10%/side
  (0.06% shown for comparison), real funding.
Engine selection rule (TRAIN and VALID-C only, each engine run alone): keep an engine if TRAIN PF ≥ 1.1 and
VALID-C PF ≥ 1.2 with net > 0 in both. Then the combined V2: F (fee ×1.5) and G (±20% on each kept engine's
stop/TP/hold) on VALID-C; HOLDOUT read once, reported as contaminated (seen during the original design).

### Finding during V2 selection (before any HOLDOUT read)
The original's backtest config (`r8_bear_adaptive/config_allweather.json`) uses StaticPairList **without**
`allow_inactive`, so freqtrade silently drops the 8 U162 pairs delisted since (NEIROETH, UXLINK, AI16Z, OM, TON,
VINE, MKR, OMNI) — survivorship bias in every original number. V2 configs set `allow_inactive: true`.
The first V2 selection run inherited the same flaw; it is discarded and rerun.
- 2026-09-25T10:30:02Z selection frozen: only exhaust_short kept (TRAIN PF 1.24 / VALID PF 2.04). Final V2 = exhaust only; reading HOLDOUT once now.

## V2 results (fee 0.10%/side unless noted; 100 USDT, 10 slots, delisted pairs included)
| run | trades | return | PF | day t | notes |
|---|---|---|---|---|---|
| V2 all four engines, TRAIN | 1178 | +85.0% | 1.26 | 2.13 | NEIROETH 17% |
| V2 all four engines, VALID-C | 293 | +9.0% | 1.15 | 1.57 | ARC 112% of profit, one day 65% |
| engine selection (alone, TRAIN / VALID-C PF) | | | | | dual_sq_long 1.41 / 0.43; dual_sq_short 1.80 / 0.95; flushout 0.99 / 0.67; exhaust 1.24 / 2.04 → only exhaust kept |
| final = exhaust only, TRAIN | 220 | +10.6% | 1.24 | 1.53 | 10 slots dilute a single engine; first 100 h lost to warm-up |
| final, VALID-C (0.10% / 0.15%) | 99 | +18.5% / +17.3% | 2.04 / 1.96 | 2.17 / 2.08 | ARC 61% |
| final, HOLDOUT read once (0.06 / 0.10 / 0.15%) | 98 | +5.7 / +4.9 / +3.9% | 1.25 / 1.22 / 1.17 | 0.88 / 0.78 / 0.65 | ARC 102–134% of profit |
| final, VALID-C + HOLDOUT | 197 | | 1.58 | 2.16 | ex-ARC PF 1.31, t 1.11 |

Conclusion: once the implementation is fixed and delisted pairs are included, the three "all-weather" engines
lose on VALID-C; the only surviving engine is the funding-exhaustion short, i.e. the V2 system reduces to R24
(`FundingExhaustionShort5m`, already committed with its own dry-run config). The all-weather design is not supported.
