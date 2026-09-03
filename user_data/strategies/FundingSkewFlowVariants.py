"""Controlled variants of the funding-skew momentum strategy.

These classes keep the measured funding, liquidity, execution, stop, and holding
rules in ``FundingSkewMomentum5m`` unchanged.  They only vary the signal
timeframe and a causal 15-minute price-flow confirmation so the backtests have
one interpretable difference at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from FundingSkewMomentum5m import FundingSkewMomentum5m


class FundingSkewBaseline1m(FundingSkewMomentum5m):
    """The current 225-minute production logic, explicitly run on 1m candles."""

    timeframe = "1m"


class _FundingSkewMom15Base(FundingSkewBaseline1m):
    """Publish price movement in the funding-paying direction before entry."""

    startup_candle_count = 15

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        # At settlement T, close[T] is known before the T+1m fill.  Multiplying by
        # sign(funding) makes a fall positive for the overwhelmingly short signal.
        dataframe["mom15_signed"] = np.sign(dataframe["fr_bps"]) * dataframe["close"].pct_change(15)
        return dataframe


class FundingSkewMom15Filter1m(_FundingSkewMom15Base):
    """Enter only when the prior 15m move confirms the funding-paying side."""

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        rejected = (dataframe["enter_short"] == 1) & ~(dataframe["mom15_signed"] > 0)
        dataframe.loc[rejected, "enter_short"] = 0
        dataframe.loc[rejected, "enter_tag"] = None
        dataframe.loc[dataframe["enter_short"] == 1, "enter_tag"] = "fund_pay_short_mom15"
        return dataframe


class FundingSkewMom15Scaled1m(_FundingSkewMom15Base):
    """Keep every signal, but use half stake when 15m price flow disagrees."""

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        entered = dataframe["enter_short"] == 1
        confirmed = entered & (dataframe["mom15_signed"] > 0)
        dataframe.loc[confirmed, "enter_tag"] = "fund_pay_short_mom15_full"
        dataframe.loc[entered & ~confirmed, "enter_tag"] = "fund_pay_short_mom15_half"
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        if entry_tag == "fund_pay_short_mom15_half":
            return proposed_stake * 0.5
        return proposed_stake


class FundingSkewBaseline5m(FundingSkewMomentum5m):
    """The unchanged production logic, explicitly run on true 5m candles."""

    timeframe = "5m"
