"""FundingExhaustionShort5m -- r3 round R24 (user_data/minute_research/r3/LOG.md, r3c/RESULTS_schemeC.md).

When a contract's funding has settled at >= min_rate for three consecutive settlements (longs crowded
and paying), short 5 minutes after the third settlement and hold for hold_minutes.

Timing on 5m candles (freqtrade: signal on candle close, fill at next candle open):
  the candle opening at settlement T carries fr(T), fr(prev), fr(prev2); it closes at T+5m -> short at T+5m open.
  custom_exit fires once the trade is hold_minutes old -> exit at market.

Funding source:
  * backtest / lookahead-analysis: the datadir's 1h funding_rate feather (real settlement rows only). The
    DataProvider copy is gap-filled by freqtrade with rate 0, which would break the three-settlement chain.
  * dry-run / live: an in-strategy cache. bot_start seeds the last settlements of every whitelisted pair;
    bot_loop_start refreshes all symbols with one /fapi/v1/fundingRate call per minute during the first
    minutes of each hour (settlements happen on the hour), so the candle opening at T sees fr(T) when it
    closes at T+5m. Letting freqtrade refresh funding_rate candles instead would re-download every pair on
    every loop between settlements (its refresh check is keyed on the last settlement time).

Stoploss -50%: in backtests it fired once in 440 trades; it is disaster insurance for leveraged use, not a
source of return (a -20% stop cut profit in every segment, see r3c/bt100_sl20_*).
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from freqtrade.enums import RunMode
from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class FundingExhaustionShort5m(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True
    minimal_roi = {"0": 100}
    stoploss = -0.50
    use_exit_signal = True          # freqtrade 2026.x only calls custom_exit when this is set
    process_only_new_candles = True
    startup_candle_count = 0
    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    min_rate = 0.0003
    hold_minutes = 480

    # dry-run / live funding cache: pair -> {settlement time (UTC, on the hour): rate}
    _funding: dict[str, dict[pd.Timestamp, float]] = {}
    _last_fetch: datetime | None = None
    _live: bool = False

    def bot_start(self, **kwargs) -> None:
        self._funding = {}
        self._last_fetch = None
        self._live = self.config["runmode"] in (RunMode.DRY_RUN, RunMode.LIVE)
        if not self._live:
            return
        api = self.dp._exchange._api
        seeded = 0
        for pair in self.dp.current_whitelist():
            try:
                hist = api.fetch_funding_rate_history(pair, limit=5)
            except Exception as e:  # a pair that fails to seed only loses its first signals
                logger.warning(f"funding seed failed for {pair}: {e}")
                continue
            for h in hist:
                self._store(pair, h["timestamp"], h["fundingRate"])
            seeded += 1
        logger.info(f"funding cache seeded for {seeded} pairs")
        self._fetch_recent(datetime.now().astimezone())

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        if not self._live or current_time.minute > 6:
            return
        if self._last_fetch and (current_time - self._last_fetch).total_seconds() < 50:
            return
        self._fetch_recent(current_time)

    def _store(self, pair: str, ts_ms: int, rate) -> None:
        t = pd.Timestamp(int(ts_ms), unit="ms", tz="UTC").round("h")   # fundingTime carries ms jitter
        self._funding.setdefault(pair, {})[t] = float(rate)

    def _fetch_recent(self, now: datetime) -> None:
        self._last_fetch = now
        ex = self.dp._exchange
        try:
            rows = ex._api.fapiPublicGetFundingRate(
                {"startTime": int((now - timedelta(hours=2)).timestamp() * 1000), "limit": 1000}
            )
        except Exception as e:
            logger.warning(f"funding refresh failed: {e}")
            return
        by_id = {
            m["id"]: symbol
            for symbol, m in ex.markets.items()
            if m.get("swap") and m.get("linear") and m.get("settle") == "USDT"
        }
        whitelist = set(self.dp.current_whitelist())
        n = 0
        for r in rows:
            pair = by_id.get(r["symbol"])
            if pair in whitelist:
                self._store(pair, r["fundingTime"], r["fundingRate"])
                n += 1
        logger.info(f"funding refresh: {n} whitelisted settlements in the last 2h")

    def _settlements(self, pair: str) -> DataFrame:
        if self._live:
            s = pd.Series(self._funding.get(pair, {}), dtype="float64").sort_index()
            f = pd.DataFrame({"date": s.index, "fr": s.to_numpy()})
        else:
            datadir = Path(self.config["datadir"])
            name = pair.replace("/", "_").replace(":", "_")
            f = pd.read_feather(datadir / "futures" / f"{name}-1h-funding_rate.feather")[["date", "open"]]
            f = f.rename(columns={"open": "fr"}).sort_values("date")
        f["fr1"] = f["fr"].shift(1)
        f["fr2"] = f["fr"].shift(2)
        f["date"] = f["date"].astype("datetime64[ns, UTC]")
        return f

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        f = self._settlements(metadata["pair"])
        d = dataframe[["date"]].copy()
        d["date"] = d["date"].astype("datetime64[ns, UTC]")
        m = d.merge(f, on="date", how="left")          # only the candle opening exactly at T gets values
        dataframe["fr"] = m["fr"].to_numpy()
        dataframe["fr1"] = m["fr1"].to_numpy()
        dataframe["fr2"] = m["fr2"].to_numpy()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        r = self.min_rate
        cond = (dataframe["fr"] >= r) & (dataframe["fr1"] >= r) & (dataframe["fr2"] >= r)
        dataframe.loc[cond, ["enter_short", "enter_tag"]] = (1, "exhaust_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        if current_time >= trade.open_date_utc + timedelta(minutes=self.hold_minutes):
            return "time"
        return None


class FundingExhaustionShort5m_F024(FundingExhaustionShort5m):
    min_rate = 0.00024


class FundingExhaustionShort5m_F036(FundingExhaustionShort5m):
    min_rate = 0.00036


class FundingExhaustionShort5m_H384(FundingExhaustionShort5m):
    hold_minutes = 384


class FundingExhaustionShort5m_H576(FundingExhaustionShort5m):
    hold_minutes = 576
