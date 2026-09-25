# r3 final report (2026-09-25)

**Update after user decision ("启用放宽口径，验证一次HOLDOUT"):** R24 funding-exhaustion short passes §4 under the event-type relaxed option — see the section at the end. Under the standard option nothing passed.

Protocol: `.claude/skills/binance-minute-strategy-research/SKILL.md`. Two Claude sessions worked this
directory in parallel (this one and peer session `freqtrade-20`); their round numbers overlap in
`LOG.md` (two different "R13–R18" sets). Together they ran **≈ 70 distinct pre-registered ideas**, past
the §6.8 limit of 50 rounds. **No strategy satisfied §4 A–G. HOLDOUT (2026-03..08) was never read.**

## Setup
- Data: Binance Vision 1m perps with `count` / `taker_buy_quote_volume` (864 symbols, 2025-01..2026-09),
  funding, mark, 5m `metrics` (OI, long/short ratios). Downloaded for r3: spot 5m klines (55 of U60),
  futures `bookDepth` for U60 (2025-01..2026-09, 37.7k daily files), aggTrades for 35 symbol-days.
- Splits: TRAIN 2025-01..09 / VALID 2025-10..2026-02 / HOLDOUT 2026-03..08. Universes U60 / U162 by
  TRAIN liquidity. Cost 7.5–15 bp per side, real funding in freqtrade.
- freqtrade datadirs: `user_data/data/r3` (U60 1m), `user_data/data/r3b` (U162 5m); both validated clean.

## Candidates that reached VALID (all rejected)

| idea | TRAIN | VALID | failed |
|---|---|---|---|
| Pre-settlement receiver harvest (this session R11/R13, freqtrade `SettlementReceiver5m`) | 330 tr, +0.46 %/tr, PF 2.22, reconciled to harness (median diff 0 bp), no lookahead | 250 tr, +0.02 %/tr, PF 1.04 (peer run: PF 0.80) | C, B |
| OI build-up breakdown short (peer R20) | +57 bp, t 2.07 | −33 bp, PF 0.83 | C |
| EMA-offset dip buying (peer R21) | +14 bp, t 4.74 | −40 bp, PF 0.78 | C |
| Multi-settlement funding exhaustion short (peer R24) | +107 bp, t 2.86 | +52 bp, PF 1.24, t 2.37 | E: ARC = 90 % of profit |
| BTC momentum → high-beta follow (peer R30) | +99.5 bp, PF 2.48 | +2 bp, PF 1.02 | C, F |
| Funding z-score dislocation (peer R34) | +89 bp, PF 1.38 | −58 bp, PF 0.81 | C |

## What was learned (new relative to `reference/prior-results.md`)
1. **Funding is priced out like a dividend.** At settlement the price moves against the receiver by
   ≈ the funding paid (−52 bp vs +53 bp funding, F ≥ 0.3 %). aggTrades: the move builds −15 bp @0.2 s,
   −25 bp @1 s, −44 bp @5 s after the first print. A 1m "exit at T open" backtest captures a
   sub-second fill that is not realistic; with a 25 bp haircut the edge does not survive VALID.
2. **freqtrade fills missing 1h funding candles with rate 0.** Any strategy inferring the settlement
   interval from the funding frame must drop zero rows first (otherwise every contract looks hourly).
3. New data families gave no cost-clearing edge: order-book depth imbalance (time-series and
   cross-sectional; XS long low-bid-imbalance is 24/24 offsets positive but t ≈ 1.2), spot-vs-perp
   flow and volume share, estimated liquidation-cluster heatmap, trade-size / trade-count shocks,
   quarter-hour opening imbalance (arXiv 2607.09426), round numbers, equity-perp off-hours moves.
4. Daily cross-sectional flow books pay ≈ 30 bp/day of turnover against ≈ 10 bp/day of gross spread.
5. Frequency floor B blocks the structurally anchored ideas: funding-interval changes (0.3–0.7/day),
   PAXG–XAU spread dislocations (0.7/day). Token-unlock history is not available free.

## Infrastructure notes
- data.binance.vision is reachable **directly** from this host (~360 KB/s per connection, ~1000
  files/min at 32 parallel); the local proxy caps at ~0.4 MB/s total. api.binance.com needs the proxy:
  freqtrade config `exchange.ccxt_config` / `ccxt_async_config` with `httpsProxy` + `wsProxy`.
- Background downloads die with the Claude session unless started with `setsid nohup`.

## Reproduce
`cd user_data/minute_research/r3`; round scripts `r01_*.py … r18_*.py` (this session), `batch*` /
`r4x_*.py` / `engine200.py` (peer). R13 engine run:
`freqtrade backtesting -c user_data/minute_research/r3/r13_config.json --strategy SettlementReceiver5m --datadir user_data/data/r3b --timerange 20251001-20260301`.

## Recommendation
Stop the minute-level search on this data. The only way to test anything further honestly is new
data (a fresh HOLDOUT after 2026-09) or a different execution layer (sub-second, aggTrades /
NautilusTrader) for the settlement effect. R24 (funding exhaustion short) is the closest miss; if the
user accepts the event-type option (§4 relaxed concentration), it would need its own pre-registered
HOLDOUT read — it is a funding-family idea, i.e. close to the existing repo strategy.

## Update: R24 HOLDOUT under the relaxed option (user-approved 2026-09-25)
Strategy `user_data/strategies/FundingExhaustionShort5m.py`, config `r24_config.json` (fee 0.001, U162, 5m, datadir `user_data/data/r3b`).
Rule (frozen from TRAIN): three consecutive settlements with funding ≥ 0.03 % → short at T+5 min, exit after 480 min, one position per coin.

| segment | trades | /day | mean | PF | day t | months + | top coin |
|---|---|---|---|---|---|---|---|
| TRAIN 2025-01..09 | 244 | 0.89 | +118 bp | 1.70 | 2.39 | 5/8 | SWARMS 27 % |
| VALID 2025-10..2026-02 | 230 | 1.52 | +81 bp | 1.41 | 2.05 | 4/5 | ARC 63 % |
| HOLDOUT 2026-03..08 (read once) | 97 | 0.53 | +110 bp | 1.60 | 1.01 | 4/6 | ARC 50 % |
| VALID+HOLDOUT | 327 | 0.98 | +90 bp | 1.47 | **2.23** | 8/11 | ARC 59 % |

Checks: harness reconciliation (219 matched, price diff median 0.07 bp; unmatched = freqtrade does not re-enter a pair on the exit candle);
lookahead none; fee ×1.5 VALID+HOLDOUT +261 USDT; plateau VALID PF 1.35 / 1.17 (F 0.024 / 0.036 %), 1.34 / 1.22 (hold 384 / 576 min).
Funding received ≈ +7–11 bp per trade (short = receiver).

Caveats: HOLDOUT alone t 1.01 (significance only when pooled with VALID); ex-ARC t 1.67; fails the standard option (frequency,
concentration); funding family, near the repo's existing funding research; rule authored by an unidentified agent session that
also wrote to this directory. Returns in the table are per 100 USDT stake on a 100 k wallet (non-compounding sizing).
