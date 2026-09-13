# Pre-registration: time-series factors x screens, round 2 -- longer holds, longer history, coarser bars

Written 2026-09-13 before any number below was computed. Round 1 (`PREREGISTRATION_ts_screen.md`)
found nothing at 1h bars with holds <= 72h on a one-year training set. The user's objection is that
the published price-volume anomalies in crypto live at daily bars, weekly holds, liquid names and
multi-year samples -- none of which round 1 covered. This round changes exactly those axes and
nothing else. **No new data is downloaded**: the panels are built from the existing `1h_hist`
(2022-11..2025-12, 598 symbols, OHLCV + funding) and `1h` (2026-01..2026-08 portion, 860 symbols).

## Panels

- `1d_long`: daily bars 2022-11-01..2026-08-16, symbol union (~900), fields open/high/low/close/
  volume/quote_volume/funding_rate (mean settled rate of the day, carried). Universe mask: >= 30 days
  of history and trailing 7-day median daily quote volume >= 2.4M USDT (same rule as the 1h mask).
- `4h_long`: the same at 4h bars.

## Splits (by date, identical for both panels)

train 2022-11-01..2024-12-31 (26 months) / valid 2025-01-01..2025-12-31 / holdout 2026-01-01..2026-08-16.
The holdout has never been used at these bar sizes and is read once, for the cells that pass valid.

## Screens

The 15 of round 1 plus `top30`. Liquidity ranks use trailing 7-day median quote volume.

## Entry / exit

`r = ts_rank(f, W)` with W = 90 bars at 1d, 180 bars at 4h (both = the round-1 30-day window in
calendar terms at 1h, scaled to keep the count of observations comparable: 90 days at 1d is the
shortest window that gives a stable decile). Long at r >= 0.90, short at r <= 0.10, fill next open,
fixed hold, one trade per pair, no stop. Holds at 1d: 1, 3, 7, 14, 28 days; at 4h: 12h, 1d, 3d, 7d, 14d.
Cost 5 bps/side (10 stress) + carried funding.

## Specs

Daily families with literature support, windows in days {3, 7, 14, 28, 56, 91, 182}:
operators {ts_mean, ts_std, ts_zscore, ts_rank, ts_slope, ts_delta, decay_linear} over bases
{ret1, gap, body, hl_range, upper_wick, lower_wick, close_loc, vwap_dev, log_qv, vol_ratio, qv_ratio,
amihud, close, funding_rate, funding_z} plus ts_corr pairs (ret1,vol_ratio), (close,log_qv),
(ret1,funding_rate). About 750 specs. At 4h the same windows expressed in hours.

## Gates

Stage 1 (train): n_trades >= 200, mean net >= 30 bps (>= 0 at 10 bps), NW t (daily book) >= 3,
>= 6 of the 8-9 train quarters positive.  Stage 2 (valid 2025): same sign, mean > 0, t >= 1.5, n >= 40.
Stage 3 (holdout 2026): reported once for stage-2 survivors, mean > 0 and t >= 1.0 counts as held.
Stage 4: freqtrade port of the best implementable cell(s) if any pass stage 3.

Cells: 16 screens x ~750 specs x 2 sides x 5 holds = 120,000 per panel. Expected chance passes at
stage 1 roughly 10-40; stages 2-3 are the control. Survivor counts are reported.
