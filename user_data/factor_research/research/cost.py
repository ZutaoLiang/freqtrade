"""Turn an IC into a break-even cost, which is the number that decides things.

A rank IC of -0.06 says a factor orders symbols. It does not say the ordering
is worth money, and the two can genuinely disagree: measured 2026-08-23 at 1h,
decay_linear(ret1,336h) has a rank IC of -0.017 at one bar yet its rank-weighted
book *earns* +0.10 bps a bar. Spearman is decided by the bulk of the
distribution; the P&L is decided by the tail. Every statistic upstream of this
module is a statement about ordering, so this is where that assumption is
tested rather than assumed.

The reported number is the **break-even cost in basis points**: the one-way
cost at which the book's net return reaches zero. Compare it against what the
venue charges and you have a decision.

Positions are built the way the factor is meant to be traded: lagged one bar,
ranked cross-sectionally inside the tradeable universe, centred to a
dollar-neutral book, scaled to unit gross, oriented by the factor's known sign,
and held for `hold` bars as overlapping tranches -- a fraction 1/hold of the
book is refreshed each bar. That last part matters. A factor selected on a
24-bar horizon rebalanced in full every bar pays 24 times the cost it needs to
for the same signal.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ic as IC  # noqa: E402

BARS_PER_YEAR = {"1m": 525600, "5m": 105120, "15m": 35040,
                 "30m": 17520, "1h": 8760, "4h": 2190, "1d": 365}


def quantile_profile(factor, close, mask=None, horizon=1, lag_bars=1, n_q=10,
                     rank=None, fwd=None):
    """Mean and median forward return per factor decile, and the D10-D1 spread.

    The decisive diagnostic for whether a rank IC is worth money. Spearman
    scores a +5% and a +500% return identically when they sit in adjacent
    ranks; a dollar-weighted book does not. Crypto's right tail is heavy enough
    that the two routinely disagree -- measured 2026-08-23 at 1h,
    decay_linear(ret1,336h) has a median spread of -69 bps and a mean spread of
    **+31 bps**: nearly every symbol in its top decile falls, a handful moon,
    and the handful decides the P&L.

    A factor whose mean and median spreads disagree in sign is ranking
    correctly for the typical symbol and wrongly for the ones that matter.
    """
    # `rank` and `fwd` let a caller that evaluates the same factor against
    # several masks hoist the shared work out. Ranking is an argsort over the
    # whole panel and forward returns do not depend on the factor at all, so
    # recomputing them per mask was most of the cost of a tiered sweep.
    rk = IC._rank_axis1(IC.apply_mask(IC.lag(factor, lag_bars), mask)) \
        if rank is None else rank
    r = IC.apply_mask(IC.forward_return(close, horizon) if fwd is None else fwd,
                      mask)
    both = np.isfinite(rk) & np.isfinite(r)
    q = np.floor(rk * n_q).clip(0, n_q - 1)
    means, medians, counts = [], [], []
    for k in range(n_q):
        v = r[both & (q == k)]
        if v.size:
            means.append(float(np.mean(v)) * 1e4)
            medians.append(float(np.median(v)) * 1e4)
        else:
            means.append(np.nan)
            medians.append(np.nan)
        counts.append(int(v.size))
    mean_spread = means[-1] - means[0]
    median_spread = medians[-1] - medians[0]
    return {
        "q_mean_bps": means, "q_median_bps": medians, "q_counts": counts,
        "mean_spread_bps": mean_spread, "median_spread_bps": median_spread,
        "tail_consistent": bool(np.isfinite(mean_spread) and np.isfinite(median_spread)
                                and np.sign(mean_spread) == np.sign(median_spread)),
    }


def target_book(factor, mask=None, lag_bars=1, sign=1.0, rank=None):
    """Dollar-neutral, unit-gross weights from a factor's cross-sectional rank."""
    r = IC._rank_axis1(IC.apply_mask(IC.lag(factor, lag_bars), mask)) \
        if rank is None else rank
    n = np.sum(np.isfinite(r), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = np.where(n > 0, np.nansum(np.where(np.isfinite(r), r, 0.0),
                                           axis=1, keepdims=True) / np.maximum(n, 1),
                          np.nan)
        w = r - centre
        gross = np.nansum(np.abs(w), axis=1, keepdims=True)
        w = np.where(gross > 0, w / gross, 0.0)
    return sign * np.where(np.isfinite(w), w, 0.0)


def held_book(target, hold):
    """Overlapping tranches: 1/hold of the book is refreshed each bar.

    Equivalent to running `hold` independent sleeves opened one bar apart, the
    standard construction for a signal with a multi-bar horizon. It is what
    makes turnover reflect the signal's speed instead of the rebalance clock.
    """
    if hold <= 1:
        return target
    # A trailing rolling mean, so a cumulative sum does it in three passes
    # instead of `hold` of them. At hold=24 on an 8,500 x 860 panel the loop
    # moved about 1.4 GB per call and the sweep was memory-bandwidth bound
    # long before it was CPU bound: 57 workers took 11 minutes for 50 specs
    # that each ran in 14 seconds alone. Weights are centred and bounded, so
    # the cumulative sum stays near zero and does not lose precision.
    cs = np.cumsum(target, axis=0)
    acc = cs.copy()
    acc[hold:] -= cs[:-hold]
    return acc / hold


def funding_carry(w, funding_rate, interval_hours, bar_hours):
    """Funding actually paid or received by holding book `w`.

    Omitting this was a real error, not a simplification: the surviving factors
    rank on funding, so the book they build sits deliberately on the extremes
    of the funding distribution and the cash flow is first-order. Measured
    2026-08-24 at 1h, it ran to -30% to -63% annualised and turned the best
    round-3 factor from +28% to -2% on train. Most of it comes from the short
    leg, which is short the *most negative* funding and therefore pays rather
    than collects.

    A long position pays a positive rate; the panel carries the last settled
    rate forward, so it is charged pro rata per bar.
    """
    iv = np.where(np.isfinite(interval_hours) & (interval_hours > 0),
                  interval_hours, 8.0)
    per_bar = np.where(np.isfinite(funding_rate), funding_rate * bar_hours / iv, 0.0)
    return -np.nansum(w * per_bar, axis=1)


def evaluate(factor, close, mask=None, lag_bars=1, hold=1, sign=None,
             bars_per_year=None, costs_bps=(0.0, 1.0, 2.0, 5.0, 10.0),
             funding_rate=None, interval_hours=None, rank=None, ret1=None):
    """Gross and net performance of the book this factor implies.

    `sign` orients the book. Pass the direction established out of sample; it
    is not a free parameter by the time this runs, and leaving it unset makes
    the module pick the in-sample sign, which flatters the result.

    Pass `funding_rate` and `interval_hours` whenever they are available. They
    are not optional for any factor built on funding or basis, and harmless for
    the rest.
    """
    if ret1 is None:
        ret1 = IC.forward_return(close, 1)
        ret1 = np.where(np.isfinite(ret1), ret1, 0.0)

    raw = target_book(factor, mask, lag_bars, 1.0, rank=rank)
    if sign is None:
        probe = held_book(raw, hold)
        sign = 1.0 if float(np.nansum(probe * ret1)) >= 0 else -1.0
        oriented_in_sample = True
    else:
        oriented_in_sample = False
    w = held_book(raw * sign, hold)

    price_pnl = np.nansum(w * ret1, axis=1)
    if funding_rate is not None and bars_per_year:
        bar_hours = 8760.0 / bars_per_year
        carry = funding_carry(w, funding_rate, interval_hours, bar_hours)
    else:
        carry = np.zeros_like(price_pnl)
    pnl = price_pnl + carry
    turn = np.concatenate([[np.nan], np.nansum(np.abs(np.diff(w, axis=0)), axis=1)])

    ok = np.isfinite(pnl) & np.isfinite(turn)
    pnl, turn = pnl[ok], turn[ok]
    if pnl.size < 100:
        return {"n_bars": int(pnl.size)}

    price_v, carry_v = price_pnl[ok], carry[ok]
    mean_turn = float(turn.mean())
    sd = float(pnl.std(ddof=1))
    ann = np.sqrt(bars_per_year) if bars_per_year else 1.0
    # A Sharpe is a point estimate; over a four-month window a headline 6.4
    # carries a t of only 3.6, and reporting the ratio without the t-stat reads
    # as far more evidence than the sample contains. The Bartlett bandwidth is
    # chosen from the P&L series itself, which for a near-static book is the
    # only thing that stops the per-bar count being mistaken for independent
    # bets.
    band = IC.andrews_band(pnl)
    _, _, t_stat, _ = IC.newey_west(pnl, band=band)
    out = {
        "n_bars": int(pnl.size), "hold": hold, "sign": float(sign),
        "sign_from_sample": oriented_in_sample,
        "gross_bps_per_bar": float(pnl.mean()) * 1e4,
        "price_bps_per_bar": float(price_v.mean()) * 1e4,
        "funding_bps_per_bar": float(carry_v.mean()) * 1e4,
        "turnover_per_bar": mean_turn,
        "vol_bps_per_bar": sd * 1e4,
        "gross_sharpe": float(pnl.mean() / sd * ann) if sd > 0 else np.nan,
        "pnl_t_nw": t_stat, "nw_band": band,
        "pnl_autocorr1": (float(np.corrcoef(pnl[1:], pnl[:-1])[0, 1])
                          if pnl.size > 2 else np.nan),
        "eff_obs": float(pnl.size / (1 + 2 * band)),
        # Cost is charged on the notional actually traded, so break-even is
        # where the per-bar edge equals the per-bar cost of the turnover.
        "breakeven_cost_bps": (float(pnl.mean()) / mean_turn * 1e4
                               if mean_turn > 0 else np.inf),
    }
    for c in costs_bps:
        net = pnl - turn * c / 1e4
        s = float(net.std(ddof=1))
        out[f"net_sharpe_{c:g}bps"] = float(net.mean() / s * ann) if s > 0 else np.nan
    return out
