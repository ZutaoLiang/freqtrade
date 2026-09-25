"""FundingExhaustionShort5mSeasoned -- R24 restricted to a rolling 'seasoned liquid' universe.

Identical to FundingExhaustionShort5m, plus one entry condition: on the signal day D the pair must be eligible, i.e. over the
270 calendar days before D it has >= 250 days of data and a median daily quote volume >= 10M USDT (the rolling version of
the U162 definition; see user_data/minute_research/r3c/build_seasoned_universe.py).

Why: with a fixed 2025 whitelist the backtest silently excludes coins that became hot later; on those newer coins the
short loses (r3c/r24_prod_diag.py). A rolling rule keeps the universe reproducible in production.

Eligibility source: backtest -> <datadir>/seasoned_universe.parquet; dry-run/live -> computed once per UTC day from the
exchange's daily candles (fetch_ohlcv 1d, 280 candles).
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from FundingExhaustionShort5m import FundingExhaustionShort5m


logger = logging.getLogger(__name__)


class FundingExhaustionShort5mSeasoned(FundingExhaustionShort5m):
    min_days = 250
    window_days = 270
    min_median_qv = 1e7

    _elig_table: pd.DataFrame | None = None
    _elig_live: dict[str, bool] = {}
    _elig_day: pd.Timestamp | None = None

    def bot_start(self, **kwargs) -> None:
        super().bot_start(**kwargs)
        self._elig_live, self._elig_day = {}, None
        if not self._live:
            p = Path(self.config["datadir"]) / "seasoned_universe.parquet"
            t = pd.read_parquet(p)
            t["date"] = pd.to_datetime(t["date"], utc=True)
            self._elig_table = t

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        super().bot_loop_start(current_time=current_time, **kwargs)
        if self._live:
            day = pd.Timestamp(current_time).tz_convert("UTC").floor("D")
            if self._elig_day != day:
                self._refresh_live(day)

    def _refresh_live(self, day: pd.Timestamp) -> None:
        api = self.dp._exchange._api
        out = {}
        for pair in self.dp.current_whitelist():
            try:
                c = api.fetch_ohlcv(pair, "1d", limit=self.window_days + 10)
            except Exception as e:
                logger.warning(f"daily candles failed for {pair}: {e}")
                continue
            d = pd.DataFrame(c, columns=["ts", "o", "h", "l", "c", "v"])
            d["date"] = pd.to_datetime(d.ts, unit="ms", utc=True)
            d = d[(d.date < day) & (d.date >= day - pd.Timedelta(days=self.window_days))]
            qv = d.c * d.v                                     # quote volume approximated by close x base volume
            out[pair] = bool(len(d) >= self.min_days and qv.median() >= self.min_median_qv)
        self._elig_live, self._elig_day = out, day
        logger.info(f"seasoned universe refreshed: {sum(out.values())} of {len(out)} pairs eligible")

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        pair = metadata["pair"]
        if self._live:
            ok = np.full(len(dataframe), self._elig_live.get(pair, False))
        else:
            t = self._elig_table
            e = t[t.pair == pair].set_index("date").eligible
            day = pd.to_datetime(dataframe["date"], utc=True).dt.floor("D")
            ok = day.map(e).fillna(False).to_numpy(dtype=bool)
        if "enter_short" in dataframe:
            dataframe.loc[~ok, "enter_short"] = 0
        return dataframe
