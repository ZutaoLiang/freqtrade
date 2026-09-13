"""Factor specifications: base series x operator x window.

Windows are declared in **hours**, not bars, so one specification means the
same economic lookback at every timeframe -- a 168-hour window is one week
whether it lands on 168 bars at 1h or 2016 bars at 5m. Specs whose window
collapses below three bars at a given timeframe are dropped rather than
silently evaluated on a degenerate window.

`ROUND1` is the ~50 economically motivated factors that must run first: they
establish the IC distribution a later brute-force sweep is judged against, and
if none of them shows anything the pipeline itself is suspect.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import operators as ops  # noqa: E402
from base import BARS_PER_DAY  # noqa: E402

MIN_WINDOW_BARS = 3

UNARY = {
    "ts_mean": ops.ts_mean, "ts_std": ops.ts_std, "ts_sum": ops.ts_sum,
    "ts_median": ops.ts_median, "ts_zscore": ops.ts_zscore, "ts_skew": ops.ts_skew,
    "ts_kurt": ops.ts_kurt, "ts_min": ops.ts_min, "ts_max": ops.ts_max,
    "ts_range": ops.ts_range, "ts_argmin": ops.ts_argmin, "ts_argmax": ops.ts_argmax,
    "ts_rank": ops.ts_rank, "ts_count_above_mean": ops.ts_count_above_mean,
    "ts_slope": ops.ts_slope, "decay_linear": ops.decay_linear,
    "ts_delta": ops.ts_delta, "ts_pct_change": ops.ts_pct_change,
    "ts_delay": ops.ts_delay, "ts_autocorr": ops.ts_autocorr,
    "ts_var": ops.ts_var, "ts_ema": ops.ts_ema, "ts_rsi": ops.ts_rsi,
}
BINARY = {"ts_corr": ops.ts_corr, "ts_cov": ops.ts_cov, "ts_beta": ops.ts_beta}


class Spec:
    __slots__ = ("op", "base", "hours", "base2", "tag")

    def __init__(self, op, base, hours, base2=None, tag=""):
        self.op, self.base, self.hours, self.base2, self.tag = op, base, hours, base2, tag

    def bars(self, tf):
        per_hour = BARS_PER_DAY[tf] / 24.0
        return int(round(self.hours * per_hour))

    def name(self, tf=None):
        arg = f"{self.base},{self.base2}" if self.base2 else self.base
        return f"{self.op}({arg},{self.hours}h)"

    def valid(self, tf):
        return self.bars(tf) >= MIN_WINDOW_BARS

    def compute(self, tf, loader):
        w = self.bars(tf)
        if self.op in BINARY:
            return BINARY[self.op](loader(self.base), loader(self.base2), w)
        return UNARY[self.op](loader(self.base), w)

    def __repr__(self):
        return f"<Spec {self.name()} [{self.tag}]>"


def _s(op, base, hours, base2=None, tag=""):
    return [Spec(op, base, h, base2, tag) for h in hours]


DAY, WEEK, MONTH = 24, 168, 720

ROUND1 = (
    # Momentum and reversal over the whole horizon range at once: the sign of
    # the IC across windows is what separates the two, so both come from the
    # same specification family rather than being hard-coded.
    _s("ts_sum", "ret1", [4, 12, DAY, 3 * DAY, WEEK, 2 * WEEK], tag="momentum")
    + _s("ts_zscore", "close", [DAY, 3 * DAY, WEEK], tag="momentum")
    + _s("ts_rank", "close", [DAY, WEEK, MONTH], tag="momentum")
    + _s("ts_slope", "close", [DAY, WEEK], tag="momentum")
    + _s("decay_linear", "ret1", [DAY], tag="momentum")
    # How long ago the extreme was: a proxy for trend age.
    + _s("ts_argmax", "close", [WEEK], tag="momentum")
    + _s("ts_argmin", "close", [WEEK], tag="momentum")

    # Risk: level, asymmetry and tail weight of the return distribution.
    + _s("ts_std", "ret1", [DAY, 3 * DAY, WEEK], tag="volatility")
    + _s("ts_skew", "ret1", [WEEK], tag="volatility")
    + _s("ts_kurt", "ret1", [WEEK], tag="volatility")
    + _s("ts_mean", "hl_range", [DAY, WEEK], tag="volatility")
    + _s("ts_autocorr", "ret1", [WEEK], tag="volatility")

    # Participation.
    + _s("ts_mean", "vol_ratio", [DAY, 3 * DAY], tag="volume")
    + _s("ts_zscore", "log_qv", [WEEK], tag="volume")
    + _s("ts_mean", "amihud", [DAY, WEEK], tag="liquidity")

    # Volume-price divergence: price rising on shrinking turnover is the
    # classic exhaustion signature and needs a two-series operator to see.
    + _s("ts_corr", "ret1", [DAY, WEEK], base2="vol_ratio", tag="divergence")
    + _s("ts_corr", "close", [WEEK], base2="log_qv", tag="divergence")

    # Order flow: who crossed the spread.
    + _s("ts_mean", "taker_imbalance", [4, DAY, WEEK], tag="orderflow")
    + _s("ts_zscore", "taker_imbalance", [WEEK], tag="orderflow")

    # Intrabar shape.
    + _s("ts_mean", "upper_wick", [DAY], tag="microstructure")
    + _s("ts_mean", "lower_wick", [DAY], tag="microstructure")
    + _s("ts_mean", "close_loc", [DAY], tag="microstructure")
    + _s("ts_mean", "vwap_dev", [DAY], tag="microstructure")
    + _s("ts_zscore", "vwap_dev", [WEEK], tag="microstructure")

    # Perpetual-specific: crowding and the cash-and-carry spread. These have no
    # equivalent in an equity factor library and are the likeliest edge.
    + _s("ts_mean", "funding_annual", [DAY, WEEK], tag="funding")
    + _s("ts_zscore", "funding_annual", [WEEK], tag="funding")
    + _s("ts_delta", "funding_rate", [DAY], tag="funding")
    + _s("ts_mean", "basis", [DAY], tag="basis")
    + _s("ts_zscore", "basis", [WEEK], tag="basis")
    + _s("ts_delta", "basis", [DAY], tag="basis")
)


# Round 2 widens to a filtered cartesian product; round 3 opens it fully. The
# operator subsets are ordered by how much they change the shape of a series,
# so round 2 keeps the ones that produce genuinely different signals and drops
# near-duplicates (ts_mean vs ts_median vs ts_ema at the same window).
ROUND2_OPS = ("ts_mean", "ts_std", "ts_zscore", "ts_rank", "ts_sum", "ts_slope",
              "ts_delta", "decay_linear")
# Rank IC is invariant to any monotone elementwise transform, so operators that
# differ only by one produce byte-identical statistics: ts_sum is w * ts_mean,
# and ts_var is ts_std squared on a non-negative quantity. Sweeping both halves
# of each pair costs ~9% of the run and yields duplicate rows that only get
# thrown away again at the decorrelation step. Kept in UNARY so specs already
# named in an earlier report still resolve.
MONOTONE_DUPLICATES = ("ts_sum", "ts_var")
ROUND3_OPS = tuple(op for op in UNARY if op not in MONOTONE_DUPLICATES)
SWEEP_BASES = ("ret1", "gap", "body", "hl_range", "upper_wick", "lower_wick",
               "close_loc", "vwap_dev", "log_qv", "vol_ratio", "qv_ratio",
               "count_ratio", "avg_trade_qv", "taker_imbalance", "amihud",
               "close", "funding_rate", "funding_annual", "funding_z", "basis")
SWEEP_HOURS = (4, 12, DAY, 3 * DAY, WEEK, 2 * WEEK, MONTH)
# Two-series operators are not swept blindly: the pair has to mean something,
# and a full cartesian of 20 bases against themselves is 400 mostly-noise
# combinations for every operator and window.
SWEEP_PAIRS = (("ret1", "vol_ratio"), ("ret1", "taker_imbalance"),
               ("ret1", "qv_ratio"), ("close", "log_qv"),
               ("ret1", "funding_annual"), ("ret1", "basis"),
               ("hl_range", "vol_ratio"), ("taker_imbalance", "vol_ratio"))


def cartesian(bases=SWEEP_BASES, ops=ROUND2_OPS, hours=SWEEP_HOURS,
              pairs=(), pair_ops=("ts_corr",), tag="sweep"):
    out = [Spec(op, b, h, None, tag)
           for op in ops for b in bases for h in hours]
    out += [Spec(op, a, h, b, tag)
            for op in pair_ops for a, b in pairs for h in hours]
    return out


# Round 2 is deliberately a few hundred, not a few thousand: its job is to show
# that the round-1 hits are not an artefact of the 49 specs that were chosen by
# hand, while still being small enough that the t > 3 gate stays meaningful.
ROUND2_BASES = ("ret1", "body", "hl_range", "close_loc", "vwap_dev", "log_qv",
                "vol_ratio", "avg_trade_qv", "taker_imbalance", "amihud",
                "funding_annual", "basis")
ROUND2_HOURS = (12, DAY, 3 * DAY, WEEK, 2 * WEEK)


def ROUND2():
    return cartesian(bases=ROUND2_BASES, ops=ROUND2_OPS, hours=ROUND2_HOURS,
                     pairs=SWEEP_PAIRS, tag="round2")


def ROUND3():
    return cartesian(ops=ROUND3_OPS, pairs=SWEEP_PAIRS,
                     pair_ops=tuple(BINARY), tag="round3")


# Round 4: the open-interest and positioning series, swept the same way and
# under the selection rule fixed in reports/PREREGISTRATION_round4.md before any
# of it was measured. Kept a separate round rather than folded into round 3 so
# the pre-registered set stays identifiable after the fact.
ROUND4_BASES = ("log_oi", "oi_change", "oi_turnover", "oi_price_agree",
                "ls_top_position", "ls_top_account", "ls_retail",
                "ls_divergence", "taker_ls", "oi_per_trade")
ROUND4_HOURS = (4, 12, DAY, 3 * DAY, WEEK, 2 * WEEK, MONTH)


def ROUND4():
    return cartesian(bases=ROUND4_BASES, ops=ROUND3_OPS, hours=ROUND4_HOURS,
                     pairs=(), tag="round4")


DAILY_OPS = ("ts_mean", "ts_std", "ts_zscore", "ts_rank", "ts_slope", "ts_delta", "decay_linear")
DAILY_BASES = ("ret1", "gap", "body", "hl_range", "upper_wick", "lower_wick", "close_loc", "vwap_dev",
               "log_qv", "vol_ratio", "qv_ratio", "amihud", "close", "funding_rate", "funding_z")
DAILY_DAYS = (3, 7, 14, 28, 56, 91, 182)
DAILY_PAIRS = (("ret1", "vol_ratio"), ("close", "log_qv"), ("ret1", "funding_rate"))


def DAILY():
    """Round-2 of the ts x screen study: daily-scale windows, literature families. Pre-registered
    in reports/PREREGISTRATION_ts_screen_r2.md."""
    hours = tuple(24 * d for d in DAILY_DAYS)
    return cartesian(bases=DAILY_BASES, ops=DAILY_OPS, hours=hours, pairs=DAILY_PAIRS, tag="daily")


def for_timeframe(specs, tf, available):
    """Drop specs whose window degenerates or whose base series is missing."""
    out, skipped = [], []
    have = set(available)
    for s in specs:
        if s.base not in have or (s.base2 and s.base2 not in have):
            skipped.append((s.name(), "missing base series"))
        elif not s.valid(tf):
            skipped.append((s.name(), f"window {s.bars(tf)} bars < {MIN_WINDOW_BARS}"))
        else:
            out.append(s)
    return out, skipped


if __name__ == "__main__":
    import base as B

    tf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    for label, specs in (("round1", ROUND1), ("round2", ROUND2()),
                         ("round3", ROUND3()), ("round4", ROUND4())):
        keep, drop = for_timeframe(specs, tf, B.names(tf) or [])
        print(f"{tf} {label}: {len(specs)} specs -> {len(keep)} runnable, "
              f"{len(drop)} dropped")
    keep, drop = for_timeframe(ROUND1, tf, B.names(tf) or [])
    for n, why in drop:
        print(f"  drop {n:38s} {why}")
