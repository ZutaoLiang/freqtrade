"""Follow the funding-paying side of an extreme perp funding settlement.

At a settlement the crowd that is PAYING keeps being pushed its own way. Measured on every
Binance USDT perp over 2025-01..2026-08 (12178 settlements with |funding| >= 30bps), entering
one candle after the stamp on the paying side and holding under four hours: the side that
COLLECTS funding loses 12-26bps net, the side that pays makes +106bps. A placebo drawing
random entries in the same pairs and the same direction within +/-7 days returns -38bps, so
the return is in the event timing, not in being short a falling alt market.

Screening is two layers and neither is a static list. Liquidity is a floor, not a band:
per-event returns RISE with volume, and the "niche pairs only" framing this came from cost
money -- the 3M-10M slice ran negative through 2025 while every band above 10M was positive
in all seven quarters. The event itself does the picking: ~186 pairs pass the floor on a
given day, ~14 a month actually trade.

Timeframe is 5m, not 1m. Settlement stamps fall on 5m boundaries anyway, so 1m only buys a
faster stop; quarter-by-quarter the two are the same trade set (1307 trades either way,
+205.8% vs +202.8%, worst trade -27.7% both). 5m cuts live OHLCV polling roughly fivefold,
which matters when the whitelist is ~200 pairs and a missed candle is a missed event. The
hold is 225 rather than 235 minutes because the 5m entry lands at T+5m: 235 would push the
exit onto the next 4h settlement and pay a funding fee the trade is designed to avoid.

Backtest discipline for this strategy lives in skills/fable/funding-skew-momentum.md; the two
rules that reverse the result if broken are in section 6. In short: the stop must trigger on
an observed price rather than a wick (an intrabar -25% engine stop turned a +200bps book into
-46bps), and entries have to be counted against the portfolio constraint of one open trade
per pair, which silently drops the later -- and more profitable -- settlements of a run.
"""

import logging
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
from freqtrade.enums import CandleType, RunMode
from freqtrade.persistence import Trade
from freqtrade.strategy import BooleanParameter, DecimalParameter, IntParameter, IStrategy

logger = logging.getLogger(__name__)

# Trailing liquidity is slow-moving, so it is batched rather than derived in the strategy:
# a 30-day window on 5m candles needs 8640 startup candles, which exceeds what the exchange
# will serve at startup. Refresh with scripts/build_niche_dailyvol.py (see the runbook in
# skills/fable/funding-skew-momentum.md); the fast-moving funding signal is read live.
DAILY_QV = Path(__file__).resolve().parents[1] / "niche_work" / "daily_qv.parquet"


@lru_cache(maxsize=1)
def _daily_qv() -> pd.DataFrame:
    if not DAILY_QV.exists():
        logger.error("no liquidity table at %s; nothing will trade", DAILY_QV)
        return pd.DataFrame(columns=["sym", "date", "qv"])
    return pd.read_parquet(DAILY_QV)


@lru_cache(maxsize=1024)
def _pair_qv(pair_symbol: str) -> pd.Series:
    """Trailing 30d median daily quote volume, already shifted one day."""
    g = _daily_qv().query("sym == @pair_symbol")
    if not len(g):
        return pd.Series(dtype="float64")
    # the parquet round trip drops the timezone; candle dates are tz-aware UTC
    idx = pd.DatetimeIndex(g.date)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return pd.Series(g.qv.values, index=idx)


class FundingSkewMomentum5m(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count = 0

    # every exit is explicit: the hold timer, or the close-based stop in custom_exit.
    # The engine stoploss sits at the isolated-margin floor so wicks cannot reach it.
    minimal_roi = {"0": 100}
    stoploss = -0.99

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # |funding| gate in bps. 40 is the measured knee; below 30 the edge is inside the spread.
    fr_threshold = DecimalParameter(30.0, 100.0, default=40.0, decimals=0, space="buy")
    # Prior consecutive extreme settlements required. Trading the first print of a run is
    # worth -8bps in sample; the second is +82bps and it keeps climbing. The edge is a crowd
    # already stuck, not one that just arrived.
    min_streak = IntParameter(0, 3, default=1, space="buy")
    # Trailing 30d median daily quote volume floor, in USDT. No upper limit on purpose.
    min_qv = DecimalParameter(0.0, 1e8, default=1e7, decimals=0, space="buy")
    # Upper limit, only for reproducing a banded universe. Effectively off.
    max_qv = DecimalParameter(0.0, 1e12, default=1e12, decimals=0, space="buy")
    # Loss cap on an observed close. -6% cuts winners, -25% lets squeezes run.
    close_stop = DecimalParameter(0.06, 0.25, default=0.15, decimals=2, space="sell")
    # Minutes to hold. Must stay under 240 so no 4h settlement lands inside the trade.
    hold_minutes = IntParameter(60, 235, default=225, space="sell")
    # The long side has ~50 samples across the whole history and no consistent sign.
    # These pairs skew hard to crowded shorts, so leaving it off costs almost nothing.
    allow_long = BooleanParameter(default=False, space="buy")

    # Funding history is re-fetched once an hour, not once a loop. Freqtrade decides a
    # funding_rate candle is stale when its last stamp is older than one funding_fee_timeframe
    # (1h on Binance), and most of this universe settles every 4h or 8h, so with ~270 pairs in
    # informative_pairs the bot would re-request every pair's history on every 5s iteration.
    # /fapi/v1/fundingRate is capped at 500 requests per 5 minutes per IP; past that Binance
    # answers 403 and the retries keep the ban alive. Returning [] here skips the fetch and
    # leaves the cached candles readable. Settlements post within seconds of the hour, and
    # the candle stamped T is analysed at T+5m, so one fetch at hh:01 is early enough.
    fr_fetch_minute = 1
    _fr_next_fetch: datetime | None = None

    def informative_pairs(self):
        """Funding rate series for every tradable pair, refreshed hourly (see above)."""
        if self.dp is None:
            return []
        now = datetime.now(UTC)
        if self.dp.runmode in (RunMode.LIVE, RunMode.DRY_RUN):
            if self._fr_next_fetch is not None and now < self._fr_next_fetch:
                return []
            self._fr_next_fetch = now.replace(minute=0, second=0, microsecond=0) + timedelta(
                hours=1, minutes=self.fr_fetch_minute
            )
            logger.info("refreshing funding history; next at %s", self._fr_next_fetch)
        return [
            (pair, self.dp.get_funding_rate_timeframe(), CandleType.FUNDING_RATE)
            for pair in self.dp.current_whitelist()
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        return 1.0

    def _funding(self, pair: str) -> pd.Series:
        """Settled funding rates in bps, indexed by stamp. Same source live and in backtest.

        Freqtrade stores funding on an hourly grid with zeros between settlements, and the
        pairs here settle every 1h, 4h or 8h. Dropping the zeros recovers the settlements:
        a rate of exactly zero is 1.5% of all rows but never lands inside an extreme run --
        it reclassifies 0 of 7506 extreme settlements over the full history.
        """
        df = self.dp.get_pair_dataframe(pair, candle_type=CandleType.FUNDING_RATE)
        if df is None or df.empty:
            logger.warning("no funding rate data for %s; it cannot trade", pair)
            return pd.Series(dtype="float64")
        fr = pd.Series(df["open"].values * 1e4, index=pd.DatetimeIndex(df["date"]))
        return fr[fr != 0].sort_index()

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        fr = self._funding(metadata["pair"])
        # A stamp at T is settled at T, i.e. known while the candle dated T is still open,
        # so a signal on that candle fills at the next open with no lookahead.
        dataframe["fr_bps"] = dataframe["date"].map(fr).astype("float64")
        if len(fr):
            state = fr.apply(
                lambda v: (1 if v > 0 else -1) if abs(v) >= self.fr_threshold.value else 0
            )
            run = state.groupby((state != state.shift()).cumsum()).cumcount()
            dataframe["fr_streak"] = dataframe["date"].map(run).astype("float64")
        else:
            dataframe["fr_streak"] = float("nan")
        qv = _pair_qv(metadata["pair"].split("/")[0] + "_USDT_USDT")
        if len(qv):
            # The liquidity table is rebuilt daily; between UTC midnight and the rebuild the
            # newest candles would otherwise map to nothing and the gate would silently block
            # every entry. Carry the last value forward at most 2 days -- staler than that
            # means the refresh is broken and the gate SHOULD close (runbook check 3).
            days = pd.DatetimeIndex(dataframe["date"].dt.normalize().unique()).union(qv.index)
            qv = qv.reindex(days).ffill(limit=2)
        dataframe["qv"] = dataframe["date"].dt.normalize().map(qv).astype("float64")
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        fr = dataframe["fr_bps"]
        qv = dataframe["qv"]
        established = (
            (fr.abs() >= self.fr_threshold.value)
            & (dataframe["fr_streak"] >= self.min_streak.value)
            & (qv >= self.min_qv.value)
            & (qv < self.max_qv.value)
        )
        if self.allow_long.value:
            dataframe.loc[established & (fr > 0), ["enter_long", "enter_tag"]] = (1, "fund_pay_long")
        dataframe.loc[established & (fr < 0), ["enter_short", "enter_tag"]] = (1, "fund_pay_short")
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        if current_profit <= -self.close_stop.value:
            return "close_stop"
        if (current_time - trade.open_date_utc).total_seconds() / 60 >= self.hold_minutes.value:
            return "hold_elapsed"
        return None
