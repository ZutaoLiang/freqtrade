"""Review-only variants of the 2% risk-budget shared four-slot strategy.

These classes exist to test two hypotheses raised while reviewing the
monthly-incremental Top2 research:

1. Minimum-stake scout entries consume 79% of all slot-minutes while
   contributing roughly zero profit.  ``...ScoutFreeV23`` removes them.
2. The four-slot limit is saturated 21% of the time.  ``...SixSlotV23``
   keeps every signal tier but widens the portfolio to six positions.

Nothing else changes: pools, signals, sizing formula, exits and the 40%
notional cap are inherited unmodified.
"""

from __future__ import annotations

from pathlib import Path
import sys

from freqtrade.persistence import Trade
from pandas import DataFrame


sys.path.insert(0, str(Path(__file__).resolve().parent))

from HighVolumeFourMtfV1 import (  # noqa: E402
    HighVolumeFourMtfV1,
    HighVolumeMainWaveMonthlyOffsetRiskBudget2PctV23,
)


class HighVolumeMainWaveMonthlyOffsetRiskBudget2PctScoutFreeV23(
    HighVolumeMainWaveMonthlyOffsetRiskBudget2PctV23
):
    """Trade only the risk-sized main-wave tiers."""

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        tags = dataframe["enter_tag"].fillna("")
        scouts = tags.str.contains("_scout_", regex=False)
        dataframe.loc[scouts, ["enter_long", "enter_short"]] = 0
        dataframe.loc[scouts, "enter_tag"] = None
        return dataframe


class HighVolumeMainWaveMonthlyOffsetRiskBudget2PctSixSlotV23(
    HighVolumeMainWaveMonthlyOffsetRiskBudget2PctV23
):
    """Same signals and sizing with six concurrent positions."""

    portfolio_position_slots = 6


class HighVolumeMainWaveMonthlyOffsetRiskBudget2PctLaggedTrailV23(
    HighVolumeMainWaveMonthlyOffsetRiskBudget2PctV23
):
    """Trail from the previous candle's extreme instead of the current one.

    Backtesting calls ``Trade.adjust_min_max_rates`` with the candle's high
    and low before evaluating the stop, so an ATR trail can tighten to an
    extreme that the same candle then breaches.  Live trading only knows the
    previous candle's extreme when that candle opens.  Swapping in the lagged
    extremes measures how much of the trailing-stop profit depends on the
    optimistic ordering.
    """

    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ):
        lagged_max = getattr(trade, "_lagged_max_rate", None)
        lagged_min = getattr(trade, "_lagged_min_rate", None)
        live_max, live_min = trade.max_rate, trade.min_rate
        trade.max_rate = trade.open_rate if lagged_max is None else lagged_max
        trade.min_rate = trade.open_rate if lagged_min is None else lagged_min
        try:
            return super().custom_stoploss(
                pair=pair,
                trade=trade,
                current_time=current_time,
                current_rate=current_rate,
                current_profit=current_profit,
                after_fill=after_fill,
                **kwargs,
            )
        finally:
            trade.max_rate, trade.min_rate = live_max, live_min
            trade._lagged_max_rate = live_max
            trade._lagged_min_rate = live_min


class _RiskSlotBudgetV23(HighVolumeMainWaveMonthlyOffsetRiskBudget2PctV23):
    """Cap concurrent risk-sized positions independently of the total slots.

    Three separate experiments (removing scouts, widening to six slots, and
    widening the market-cap exclusion) all degraded risk-adjusted results by
    reducing how often minimum-stake scouts blocked a slot.  The effective
    risk level is therefore set by an accident of slot contention rather than
    by a rule.  These classes make it explicit: at most
    ``risk_position_slots`` non-scout positions may be open at once, while
    the inherited four-position portfolio limit still applies.
    """

    risk_position_slots = 4

    @staticmethod
    def _is_risk_sized(entry_tag: str | None) -> bool:
        return "_scout_" not in (entry_tag or "")

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        if not super().confirm_trade_entry(
            pair=pair,
            order_type=order_type,
            amount=amount,
            rate=rate,
            time_in_force=time_in_force,
            current_time=current_time,
            entry_tag=entry_tag,
            side=side,
            **kwargs,
        ):
            return False
        if not self._is_risk_sized(entry_tag):
            return True
        open_risk = sum(
            self._is_risk_sized(trade.enter_tag) for trade in Trade.get_open_trades()
        )
        return open_risk < self.risk_position_slots


class HighVolumeMainWaveMonthlyOffsetRiskBudget2PctRiskSlots1V23(_RiskSlotBudgetV23):
    """At most one risk-sized position at a time."""

    risk_position_slots = 1


class HighVolumeMainWaveMonthlyOffsetRiskBudget2PctRiskSlots2V23(_RiskSlotBudgetV23):
    """At most two risk-sized positions at a time."""

    risk_position_slots = 2


class HighVolumeMainWaveMonthlyOffsetRiskBudget2PctRiskSlots3V23(_RiskSlotBudgetV23):
    """At most three risk-sized positions at a time."""

    risk_position_slots = 3


class HighVolumeMainWaveMonthlyOffsetRiskBudget2PctFlatRiskV23(
    HighVolumeMainWaveMonthlyOffsetRiskBudget2PctV23
):
    """Size every entry with the same risk formula, ignoring all tiering.

    The refit grid tunes stake multipliers for two "high conviction" tiers and
    routes the remaining tiers to a fixed exchange-minimum stake.  Fitting that
    grid on 2025 and on 2026 produces opposite corners of the search space, so
    this class removes the tier sizing entirely: every signal is sized by the
    inherited ATR risk budget under the same 40% notional cap.  It measures
    whether the underlying breakout signal carries an edge at all.
    """

    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return HighVolumeFourMtfV1.custom_stake_amount(
            self,
            pair=pair,
            current_time=current_time,
            current_rate=current_rate,
            proposed_stake=proposed_stake,
            min_stake=min_stake,
            max_stake=max_stake,
            leverage=leverage,
            entry_tag=entry_tag,
            side=side,
            **kwargs,
        )
