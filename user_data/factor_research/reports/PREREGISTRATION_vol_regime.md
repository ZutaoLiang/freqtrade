# Pre-registration: is market volatility a systematic conditioner of time-series factor returns?

Written 2026-09-13, before any number below was computed. It follows a **post-hoc** observation made
on 2026-09-13 (`research/regime_test.py`, recorded in `skills/fable/ts-factor-screen-research-20260912.md`
§third round): one momentum book earned -21.6 / +9.4 / +64.3 bps per day in the low / mid / high
buckets of a causal BTC realised-volatility percentile, sign-consistent across train, valid and
holdout. That observation used one factor book, one screen, 391 high-volatility days, and was found
after looking at nine regime classifiers. It is a lead, not a result. This document fixes the test.

## Two hypotheses, deliberately separated

**H1 (scientific).** Market volatility regime is a *systematic* conditioner: across the whole grid of
(screen x spec x side x hold) cells, the mean net return of trades entered in the HIGH bucket exceeds
that of trades entered in the LOW bucket, and the gap holds in valid and holdout.

H1 is one hypothesis tested on ~200,000 cells at once, so it has real power and no multiple-testing
problem. It is the question that matters: if volatility conditions the whole grid, that is a property
of the market; if only a handful of cells show it, that is the same selection effect as before.

**H2 (trading).** Restricted to HIGH-bucket entries, some cells pass the three-stage gates.

H2 carries the usual multiple-testing burden and is guarded by the same gates as rounds 1-2. H2 is
only interpreted if H1 holds; a cell that passes H2 while H1 fails is treated as selection.

## Regime definition (frozen)

Primary: on **daily** bars, `rv30 = std of BTCUSDT daily log returns over the trailing 30 days`;
`vol_pct = percentile rank of rv30 within the trailing 365 daily values` (both windows trailing and
causal, minimum 120 observations). States: **LOW** `vol_pct <= 0.33`, **MID** otherwise,
**HIGH** `vol_pct >= 0.67`. The state is read at the signal bar's close and applies to the entry;
a trade keeps the state it was opened in regardless of what happens during the hold.

Robustness grid, reported in full, never used to select: rv window {20, 30, 60} days x ranking window
{250, 365, 500} days x thresholds {0.25/0.75, 0.33/0.67, 0.40/0.60}. Also two alternative regime
series, reported alongside: cross-sectional dispersion (std across tradeable pairs of the trailing
7-day return) and mean pair volatility, each percentile-ranked the same way.

This definition is chosen because it is computable inside a freqtrade strategy from a single
informative pair, with no lookahead.

## Data, splits, universe, specs, entries

Unchanged from round 2 (`PREREGISTRATION_ts_screen_r2.md`): panels `1d_long` and `4h_long`
(2022-11-01..2026-08-16, 870 symbols); splits train 2022-11..2024-12 / valid 2025 / holdout
2026-01..08-16; 16 screens; 756 daily-window specs; entry when `ts_rank(f, W) >= 0.90` (long) or
`<= 0.10` (short), W = 90 bars at 1d and 180 at 4h; fill next open; holds 1/3/7/14/28 days at 1d and
12h/1d/3d/7d/14d at 4h; cost 5 bps per side plus carried funding.

The only new dimension is the regime bucket at entry.

## H1: statistic and decision rule

For every cell with at least 50 trades in both the HIGH and LOW buckets within a split:

    gap = mean_net5(HIGH) - mean_net5(LOW)          [bps per trade]

Primary statistic: the **median gap across cells**, reported per split. Secondary: the fraction of
cells with a positive gap (0.5 under the null), and the same two numbers computed on

    gap_adj = gap - (market return over the same holds in HIGH - in LOW)

where the market is the equal-weight book over the same screen. `gap_adj` is the one that matters:
an unadjusted gap can be produced entirely by the market being up more in high-volatility periods.

Third: the **volatility-normalised** gap, each bucket's mean divided by the realised standard
deviation of trade returns in that bucket. This separates "the same edge scaled by volatility" from
"a different edge". A pure scaling effect leaves the normalised gap at zero.

H1 holds when, on all three splits, the median `gap_adj` is positive and the positive fraction is
above 0.55, and the sign of the median `gap_adj` on valid and holdout matches train.

Nothing about H1 is a selection: every cell contributes, and the grid is the one already fixed in
round 2.

## H2: gates

Stage 1 (train, HIGH bucket only): n_trades >= 200, mean net5 >= 30 bps, mean net10 >= 0,
cluster-robust t >= 3 (trades grouped by entry date, Newey-West on the daily mean series),
>= 6 of 9 train quarters positive.
Stage 2 (valid, HIGH bucket): same sign, mean net5 > 0, t >= 1.5, n >= 40.
Stage 3 (holdout, HIGH bucket): mean net5 > 0 and t >= 1.0.
Additional requirement, fixed here: the same cell's LOW bucket must not also pass stage 1. A cell that
works in both buckets is not a volatility effect and belongs to round 2, where it already failed.

Stage 4: freqtrade port, native backtest, trade-by-trade reconciliation.

## What I will not do

- Change the regime definition, thresholds, splits, screens, specs or holds after seeing results.
- Interpret H2 survivors if H1 fails.
- Report the high-volatility bucket without the low-volatility bucket and the market control next to it.
- Treat the 2026 holdout as clean for the momentum family: the round-3 observation already looked at
  it. It is reported as a second look and labelled as such.
