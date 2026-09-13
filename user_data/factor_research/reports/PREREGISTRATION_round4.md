# Pre-registration: open-interest and positioning factors (round 4)

Written 2026-08-23, **before any round-4 factor was computed or evaluated**.
The holdout (2026-04-20..2026-08-16) has already been spent on the funding and
basis factors. It retains value for round 4 only because these factors have
never been evaluated on any split — and only if the selection is fixed in
advance rather than tuned against the result. This file is that fixture. Its
existence in git-less form is weak proof, so it is written before the sweep and
never edited afterwards; any change is a new file with a new name.

## New base series

Derived from the `metrics` dataset (open interest and long/short ratios, 5-minute
snapshots), merged into the panels as `sum_open_interest`,
`sum_open_interest_value`, `count_toptrader_long_short_ratio`,
`sum_toptrader_long_short_ratio`, `count_long_short_ratio`,
`sum_taker_long_short_vol_ratio`.

| name | definition | why |
|---|---|---|
| `log_oi` | `sign_log(sum_open_interest_value)` | position stock, level |
| `oi_change` | `sum_open_interest_value / delay(·,1) - 1` | the flow that OHLCV cannot see |
| `oi_turnover` | `quote_volume / sum_open_interest_value` | how fast positions churn |
| `oi_price_agree` | `sign(ret1) * sign(oi_change)` | +1 new money entering with the move, -1 positions closing into it |
| `ls_top_position` | `sum_toptrader_long_short_ratio` | large accounts' net tilt |
| `ls_top_account` | `count_toptrader_long_short_ratio` | large accounts, headcount-weighted |
| `ls_retail` | `count_long_short_ratio` | everyone, headcount-weighted |
| `ls_divergence` | `log(ls_top_position) - log(ls_retail)` | large accounts against the crowd |
| `taker_ls` | `sum_taker_long_short_vol_ratio` | aggressor direction, direct rather than proxied |
| `oi_per_trade` | `sum_open_interest_value / count` | position size behind each print |

## Sweep

`ROUND4 = the 10 series above x the 20 round-3 operators x windows
(4, 12, 24, 72, 168, 336, 720) hours`, minus specs whose window falls below 3
bars at a timeframe. Evaluated at 1h and 4h, forward horizon 24h.

## Selection rule, fixed now

A factor is a **candidate** only if, on **train and valid both**:

1. `breakeven_cost_bps > 5` — above the 4.5 bps Binance taker fee
2. `tail_consistent` — decile mean and median spreads agree in sign
3. `pnl_t_nw > 2` — Newey-West t on the per-bar P&L

and it passes at **both** 1h and 4h. Direction is the sign of the decile mean
spread measured on train, carried unchanged into every later split.

Candidates are then decorrelated at `|r| > 0.5` **on train data only**, in
descending order of `min(t_train, t_valid)`.

## Holdout

The surviving decorrelated list is evaluated on the holdout **once**. The result
file is `reports/HOLDOUT_RESULT_round4.csv`, and `holdout_eval.py` refuses to
overwrite an existing one. No factor is added, removed, re-parameterised or
re-oriented after that evaluation. If the result disappoints, it stands as the
answer; the response is to gather new data, not to re-cut this one.

## Renewing the holdout

The only clean reset is time. The panel ends 2026-08-16; each month of new data
is a month of genuinely unseen sample. Extend the download forward monthly and
cut the next holdout from data that postdates every decision made here.
