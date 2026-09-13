# Pre-registration: liquidity-tiered re-screen with funding included

Written 2026-08-25, before the sweep it describes was computed. Two earlier
rounds were selected under a cost model that omitted the funding cash flow, and
their shortlists are already known to me. Fixing the rule in advance is the only
thing that keeps this run from being a search over which rule flatters the
factors I have already seen.

## What is being corrected

1. **Funding cash flow.** `research/cost.py` computed P&L from price return
   only. That is exact for a price factor and first-order wrong for a book
   ranked on funding, which by construction sits on the extremes of the funding
   distribution. `cost.funding_carry` now charges it. Pre-correction results are
   preserved as `reports/cost_round3_{1h,4h}_nofunding.csv`.
2. **Liquidity.** Every previous survivor was ranked across the whole tradeable
   universe, about 350 names deep, and the resulting book's largest positions
   were micro-caps where the flat 5 bps slippage assumption is not credible.

## Universe tiers

`research/tiers.py` ranks symbols at each bar by the same trailing 7-day median
quote volume the universe mask uses, so the ranking is causal. Tiers are
cumulative — `top50`, `top100`, `top200` — plus the full universe (`all`) and
one disjoint diagnostic tier, `tail` (rank 200 and beyond). Cumulative because
the decision this feeds is which universe to trade.

Note for later: tokenised equity and commodity perpetuals (`XAUUSDT`,
`SNDKUSDT`, `SOXLUSDT`, ...) enter the top 50 only from 2026 Q1. Train contains
none, valid contains two to five, and the holdout would contain up to eight. The
train and valid results below are therefore clean crypto; a holdout look would
not be.

## Selection rule, fixed in advance

A spec holds in a tier when, in **both** the train and valid splits:

- break-even cost > 5 bps — above the 4.5 bps taker fee, not merely above zero;
- Newey-West P&L t > 2, with the Andrews AR(1) bandwidth;
- the decile mean and median spreads agree in sign (`tail_consistent`).

Direction is fixed once, on the train split of the `all` tier, and carried
unchanged into every tier and into valid. Re-fitting the sign per tier would
hand each tier a free parameter and manufacture agreement between them.

Horizon 24 hours, one-bar lag, overlapping tranches. Timeframes 1h and 4h.
Rounds 3 (2,968 specs over price, volume, funding and basis) and 4 (1,400 over
open interest and positioning).

## What each outcome means

- **Holds in `top100` and in `all`.** The edge is not a micro-cap artefact.
  Slippage stops being the binding question and `bookDepth` becomes worth
  downloading, restricted to the surviving names.
- **Holds in `all` but not in `top100` or `top50`.** The edge lives in
  illiquid names. No impact model rescues it, because the impact model can only
  make the number worse. `bookDepth` would be wasted.
- **Holds in `top50` but not in `all`.** Possible and interesting: an edge
  diluted by 300 names of noise. Treated as a candidate, not a result, because
  a 50-name cross-section is a much smaller effective sample.
- **Holds nowhere.** The carry conclusion in the memory note
  `only-carry-survives-cost` does not survive the funding correction, and the
  search returns to phase 3 with the corrected cost model.

The holdout is not consulted for any of this. It has already been spent once.
