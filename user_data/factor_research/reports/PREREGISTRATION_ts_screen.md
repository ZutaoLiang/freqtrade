# Pre-registration: time-series factors inside screened universes

Written 2026-09-12, before any of the numbers it describes were computed.

## Question

Earlier rounds (1-4, liq) evaluated factors as **cross-sectional rank books** over the
whole tradeable universe and found that only funding/basis carry clears cost. That is not
the question freqtrade asks. A freqtrade strategy holds one pair at a time on an absolute
signal: "this pair's own factor is extreme now -> enter, hold h bars, exit". The question
here is therefore:

> Does any time-series factor, evaluated as per-pair absolute entries with fixed holds,
> clear realistic cost **inside some causally defined subset of the universe**, and does the
> subset hold up out of sample?

The subset ("screen") is part of the hypothesis, not a free parameter tuned afterwards:
every screen below is fixed in advance and every (screen, factor) cell is reported.

## Data and splits

- Panel `1h` (2025-01-01..2026-08-16, 860 symbols, full fields incl. funding, basis, OI).
- Panel `1h_hist` (2022-11-01..2025-12-31, 598 symbols, OHLCV + funding only) is the
  **historical out-of-sample set**: it is read only for cells that pass the 2025-26 gates,
  and only on the pre-2025 portion, once.
- Splits: `train` 2025-01-01..2025-12-22, `valid` 2025-12-22..2026-04-20,
  `holdout` 2026-04-20..2026-08-16 (already spent once by the carry factors; here it is a
  second look and is labelled as such, never as clean).

## Screens (all causal, all evaluated at every bar)

| family | screens | definition |
|---|---|---|
| liquidity | top50, top100, top200, all | trailing 7d median quote-volume rank inside the tradeable mask (`tiers.py`) |
| age | young, mature | bars since the pair first entered the tradeable mask: < 90 days / >= 90 days |
| volatility | vol_lo, vol_hi | trailing 7d std of 1h returns, cross-sectional bottom / top tercile at that bar |
| funding | fr_neg, fr_pos, fr_flat | trailing 24h mean of the carried funding rate: <= -3 bps, >= +3 bps, between |
| trend | trend_up, trend_dn | `ts_rank(close, 720h)` >= 0.67 / <= 0.33 |
| activity | act_hi, act_lo | quote volume 1d / 30d ratio, cross-sectional top / bottom tercile |

Fifteen screens. Every screen is intersected with the tradeable universe mask.

## Entry rule (fixed)

For factor `f` and its trailing 720-bar percentile rank `r = ts_rank(f, 720)`:
long entry when `r >= 0.90`, short entry when `r <= 0.10`, evaluated on bar close, filled at
the **next bar's open**, held exactly `h` bars, exited at the open of bar `t+1+h`. One open
trade per pair (no re-entry while open). No stop: the hold is the risk control being tested;
stops are a freqtrade-stage decision. Holds: 6h, 24h, 72h. Sides reported separately;
the sign is **not** fitted -- long and short are two hypotheses each.

Cost: 5 bps per side (taker fee) plus funding actually carried (rate x bar_hours / interval,
pro rata over the hold). Stress at 10 bps per side is reported alongside.

## Specs

Round-2 filtered cartesian (520 specs: 8 operators x 20 base series x 7 windows, filtered)
on the `1h` panel. Nothing is added after the first result is seen.

## Gates (fixed)

A cell (screen, spec, side, hold) passes **stage 1** on `train` when all of:
- n_trades >= 300;
- mean net return per trade >= 30 bps at 5 bps/side (>= 0 at 10 bps/side);
- Newey-West t of the equal-weight open-book daily P&L >= 3.0;
- at least 3 of the 4 train quarters have positive mean net per trade.

Stage 2 on `valid`: same sign, mean net per trade > 0, NW t >= 1.5, n_trades >= 60.

Stage 3: the surviving cells are decorrelated (spec correlation > 0.7 inside a screen counts
as one), then evaluated on `1h_hist` restricted to 2022-11..2024-12 with the screen rebuilt
from that panel's own fields. A cell that fails there is reported as "2025-26 only".

Stage 4: the best 1-3 cells become a freqtrade strategy (screen = pairlist + in-strategy
regime gate; factor in `populate_indicators`; entry/exit as above), backtested natively on
1h data with `--timeframe-detail 1m` disabled (hold-based exits do not need it), and
reconciled trade-by-trade against the research simulation.

## Multiple testing

15 screens x 520 specs x 2 sides x 3 holds = 46,800 cells on train. Under pure noise with
the stage-1 gates roughly 5-20 cells would pass by chance (t >= 3 one-sided ~ 0.13%, before
the other gates). Stage 2 and 3 exist for that reason; a cell is only called an edge after
stage 3, and the count of stage-1 survivors is reported so the reader can judge the base rate.

## What I will not do

- Add screens, windows, thresholds or holds after seeing results.
- Fit the sign per screen.
- Use the 2026 holdout for selection.
- Report a cell without its neighbours (adjacent windows and holds are shown next to it).
