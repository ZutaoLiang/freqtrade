"""FundingExhaustionShort5m -- r3 peer round R24, HOLDOUT declaration (user_data/minute_research/r3/LOG.md).

When a contract's funding has settled at >= min_rate for three consecutive settlements (longs crowded
and paying), short 5 minutes after the third settlement and hold for hold_minutes.

Timing on 5m candles (freqtrade: signal on candle close, fill at next candle open):
  the candle opening at settlement T carries fr(T), fr(prev), fr(prev2); it closes at T+5m -> short at T+5m open.
  custom_exit fires on the candle opening at entry + hold_minutes -> exit at that open.

Backtest-only: settlements are read from the datadir's 1h funding_rate feather (real settlement rows only).
The DataProvider copy is gap-filled by freqtrade with rate 0, which would break the three-settlement chain.
"""
from datetime import timedelta
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class FundingExhaustionShort5m(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True
    minimal_roi = {"0": 100}
    stoploss = -0.99
    use_exit_signal = True          # freqtrade 2026.x only calls custom_exit when this is set
    process_only_new_candles = True
    startup_candle_count = 0

    min_rate = 0.0003
    hold_minutes = 480

    def _settlements(self, pair: str) -> DataFrame:
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
