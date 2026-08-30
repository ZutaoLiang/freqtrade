"""Follow the funding-paying side of an extreme perp funding settlement.

Event study on the niche Binance USDT-perp band (median daily quote volume 3M-60M,
2025-11..2026-08) measured, per settlement with |funding| >= 40bps and entry at the next
1m open: +219bps mean / +131bps median over a 240 minute hold in-sample, +486/+207 out of
sample, positive in every calendar month and after dropping the five largest contributing
pairs. The edge sits on the side that PAYS funding -- the crowd's direction keeps running --
not on the side that collects it; collecting was measured at -12 to -26bps net.

Costs and fill assumptions follow the shared backtest discipline: taker entry and exit,
`--fee` carrying the real one-side fee plus a half-spread estimate. The hold stops short of
240 minutes so no funding stamp falls inside a trade under either the 4h or the 8h schedule
Binance runs on these pairs -- the settlement is the signal, never the payment.

The stop is deliberately NOT the engine stoploss. These are 1m candles on thin alts: an
intrabar -25% stop was scalped by wicks on 9% of trades that then closed the hold POSITIVE,
turning a +200bps book into a -46bps one. The stop here triggers on an observed price, not
on a wick, and the engine stoploss is left at the isolated-margin floor. Path simulation
over 2827 events, net of 20bps round trip: -15% close stop gives +62bps/trade in-sample and
+222bps out, with the worst single trade at -30% instead of -99%.

Entry needs the funding to have ALREADY been extreme at the previous settlement, same sign.
The first extreme print of a run is worth -8bps in-sample; the second is +82bps and it keeps
climbing with the run (+114bps at the fourth and beyond). The edge is a crowd that is
already stuck, not one that has just arrived. This also fixes a structural loss: one open
trade per pair means a 235 minute hold swallows the following settlements of a run, and on
the hourly-funded pairs those swallowed entries were carrying most of the return. Confirmed
out of sample, but note the streak rule was chosen after reading the in-sample split.
"""

import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
)

logger = logging.getLogger(__name__)

FUNDING_DIR = Path("/root/freqtrade/user_data/data/binance_public/funding")
DAILY_QV = Path("/root/freqtrade/user_data/niche_work/daily_qv.parquet")


@lru_cache(maxsize=512)
def _funding_series(raw_symbol: str) -> pd.Series:
    """Settled funding rate stamps for one symbol, in bps. Empty series when absent."""
    path = FUNDING_DIR / f"{raw_symbol}.parquet"
    if not path.exists():
        logger.warning("no funding data for %s", raw_symbol)
        return pd.Series(dtype="float64")
    fr = pd.read_parquet(path).set_index("date")["funding_rate"]
    return (fr[~fr.index.duplicated()] * 1e4).sort_index()


@lru_cache(maxsize=1)
def _daily_qv() -> pd.DataFrame:
    """Trailing 30-day median daily quote volume per pair, already shifted one day."""
    if not DAILY_QV.exists():
        logger.warning("no daily volume table at %s; liquidity gate disabled", DAILY_QV)
        return pd.DataFrame(columns=["sym", "date", "qv"])
    return pd.read_parquet(DAILY_QV)


@lru_cache(maxsize=512)
def _pair_qv(pair_symbol: str) -> pd.Series:
    d = _daily_qv()
    g = d[d.sym == pair_symbol]
    if not len(g):
        return pd.Series(dtype="float64")
    # the parquet round trip drops the timezone; candle dates are tz-aware UTC
    idx = pd.DatetimeIndex(g.date)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return pd.Series(g.qv.values, index=idx)


@lru_cache(maxsize=512)
def _extreme_run(raw_symbol: str, threshold: float) -> pd.Series:
    """How many consecutive PRIOR settlements were already extreme in the same direction."""
    fr = _funding_series(raw_symbol)
    if fr.empty:
        return fr
    state = np.sign(fr) * (fr.abs() >= threshold)
    return state.groupby((state != state.shift()).cumsum()).cumcount()


class FundingSkewMomentum1m(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1m"
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count = 0

    # every exit is explicit: the hold timer, or the close-based stop in custom_exit.
    # The engine stoploss is left at the isolated-margin floor so wicks cannot trigger it.
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
    # prior consecutive extreme settlements required. 0 (trade the first print) is a loser.
    min_streak = IntParameter(0, 3, default=1, space="buy")
    # The long side has 25 samples across both windows and lost in each; off until it has a
    # measured case of its own. These pairs skew hard to crowded shorts, so this costs little.
    allow_long = BooleanParameter(default=False, space="buy")
    # Liquidity floor in USDT of trailing 30d median daily quote volume, tested at entry.
    # Below 10M the edge is not there: over 2025-01..2026-08 the 3M-10M slice ran negative
    # through 2025 while every band above 10M was positive in all seven quarters. There is
    # deliberately NO upper limit -- the "niche pairs only" framing cost money.
    min_qv = DecimalParameter(0.0, 1e8, default=1e7, decimals=0, space="buy")
    # Upper liquidity limit, for reproducing a banded universe. Off by default.
    max_qv = DecimalParameter(0.0, 1e12, default=1e12, decimals=0, space="buy")
    # minutes to hold. Kept under 240 so no 4h settlement lands inside the trade.
    hold_minutes = IntParameter(60, 235, default=235, space="sell")
    # loss cap on an observed price. -6% cuts winners, -25% lets squeezes run; -15% is the knee.
    close_stop = DecimalParameter(0.06, 0.25, default=0.15, decimals=2, space="sell")

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        return 1.0

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        raw = metadata["pair"].split("/")[0] + "USDT"
        fr = _funding_series(raw)
        # A stamp at T is settled at T, i.e. known while the candle dated T is still open,
        # so the signal on that candle fills at the T+1m open with no lookahead.
        dataframe["fr_bps"] = dataframe["date"].map(fr).astype("float64")
        dataframe["fr_streak"] = dataframe["date"].map(
            _extreme_run(raw, self.fr_threshold.value)
        ).astype("float64")
        qv = _pair_qv(metadata["pair"].split("/")[0] + "_USDT_USDT")
        dataframe["qv"] = dataframe["date"].dt.normalize().map(qv).astype("float64")
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        fr = dataframe["fr_bps"]
        established = (
            (fr.abs() >= self.fr_threshold.value)
            & (dataframe["fr_streak"] >= self.min_streak.value)
            & (dataframe["qv"] >= self.min_qv.value)
            & (dataframe["qv"] < self.max_qv.value)
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
        held = (current_time - trade.open_date_utc).total_seconds() / 60
        if held >= self.hold_minutes.value:
            return "hold_elapsed"
        return None
