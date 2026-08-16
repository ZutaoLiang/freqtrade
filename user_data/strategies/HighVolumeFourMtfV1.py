"""Four-pair, multi-timeframe high-volume momentum strategy.

The base 15m timeframe is only used for order execution.  Each pair keeps the
signal horizon selected by the 2026 research run:

* AKE:    30m volume-confirmed Donchian breakout, long only.
* EDEN:    1h volume-confirmed Donchian breakout, long only.
* VELVET:  1h volume-confirmed Donchian breakout, long and short.
* APR:     4h volume-confirmed EMA trend, long and short.

Only standard OHLCV fields are used.  Quote turnover is approximated by
``volume * close`` and compared with the previous 20 completed signal candles.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy.strategy_helper import stoploss_from_absolute


class HighVolumeFourMtfV1(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 2200

    minimal_roi = {"0": 100.0}
    stoploss = -0.35
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    signal_timeframes = ("30m", "1h", "4h", "1d")
    risk_per_trade = 0.005
    maximum_notional_fraction = 0.25
    initial_stop_atr = 2.0
    trailing_stop_atr = 3.0
    rvol_threshold = 1.0

    pair_profiles: dict[str, dict[str, Any]] = {
        "AKE/USDT:USDT": {"signal_tf": "30m", "mode": "long", "kind": "breakout"},
        "EDEN/USDT:USDT": {"signal_tf": "1h", "mode": "long", "kind": "breakout"},
        "VELVET/USDT:USDT": {
            "signal_tf": "1h",
            "mode": "long_short",
            "kind": "breakout",
        },
        "APR/USDT:USDT": {"signal_tf": "4h", "mode": "long_short", "kind": "ema"},
    }

    def profile_for_pair(self, pair: str) -> dict[str, Any] | None:
        """Return the signal profile used by a pair.

        Fixed-universe strategies use the explicit research profiles above.
        Rotation subclasses can provide a common profile without mutating this
        mapping whenever the monthly whitelist changes.
        """
        return self.pair_profiles.get(pair)

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not self.dp:
            return []
        return [
            (pair, timeframe)
            for pair in self.dp.current_whitelist()
            for timeframe in self.signal_timeframes
        ]

    @staticmethod
    def _informative_indicators(dataframe: DataFrame) -> DataFrame:
        dataframe["ema20"] = dataframe["close"].ewm(span=20, adjust=False).mean()
        dataframe["ema50"] = dataframe["close"].ewm(span=50, adjust=False).mean()
        previous_close = dataframe["close"].shift(1)
        true_range = pd.concat(
            [
                dataframe["high"] - dataframe["low"],
                (dataframe["high"] - previous_close).abs(),
                (dataframe["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        dataframe["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

        dataframe["donchian_high20"] = dataframe["high"].shift(1).rolling(20).max()
        dataframe["donchian_low20"] = dataframe["low"].shift(1).rolling(20).min()
        dataframe["donchian_high10"] = dataframe["high"].shift(1).rolling(10).max()
        dataframe["donchian_low10"] = dataframe["low"].shift(1).rolling(10).min()

        dataframe["quote_volume_proxy"] = dataframe["volume"] * dataframe["close"]
        prior_mean = dataframe["quote_volume_proxy"].shift(1).rolling(20).mean()
        dataframe["rvol20"] = dataframe["quote_volume_proxy"] / prior_mean.replace(0, np.nan)
        recent_quote_volume6 = dataframe["quote_volume_proxy"].rolling(
            6, min_periods=6
        ).sum()
        prior_quote_volume6 = dataframe["quote_volume_proxy"].shift(6).rolling(
            6, min_periods=6
        ).sum()
        dataframe["quote_volume_growth6"] = (
            recent_quote_volume6 / prior_quote_volume6.replace(0, np.nan) - 1
        )
        dataframe["quote_volume24"] = dataframe["quote_volume_proxy"].rolling(
            24, min_periods=24
        ).sum()
        dataframe["rvol_count6"] = (
            dataframe["rvol20"].ge(1.0).rolling(6, min_periods=6).sum()
        )
        dataframe["candle_up"] = dataframe["close"] > dataframe["open"]
        dataframe["candle_down"] = dataframe["close"] < dataframe["open"]

        # Breakout-quality and higher-timeframe trend-strength fields used by
        # the main-wave experiments.  Keeping them in the informative frame
        # ensures shifts are measured in real 1h/4h candles before the values
        # are forward-filled onto the 15m execution frame.
        candle_range = (dataframe["high"] - dataframe["low"]).replace(0, np.nan)
        dataframe["close_location"] = (
            (dataframe["close"] - dataframe["low"]) / candle_range
        ).fillna(0.5)
        dataframe["body_atr14"] = (
            (dataframe["close"] - dataframe["open"]).abs()
            / dataframe["atr14"].replace(0, np.nan)
        )
        dataframe["ema20_slope3"] = (
            dataframe["ema20"] / dataframe["ema20"].shift(3) - 1
        )
        dataframe["ema_separation_atr"] = (
            (dataframe["ema20"] - dataframe["ema50"])
            / dataframe["atr14"].replace(0, np.nan)
        )
        dataframe["atr_percent"] = (
            dataframe["atr14"] / dataframe["close"].replace(0, np.nan)
        )
        dataframe["breakout_extension_atr"] = (
            (dataframe["close"] - dataframe["donchian_high20"])
            / dataframe["atr14"].replace(0, np.nan)
        )
        dataframe["breakdown_extension_atr"] = (
            (dataframe["donchian_low20"] - dataframe["close"])
            / dataframe["atr14"].replace(0, np.nan)
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            return dataframe
        for informative_timeframe in self.signal_timeframes:
            informative = self.dp.get_pair_dataframe(
                pair=metadata["pair"], timeframe=informative_timeframe
            )
            informative = self._informative_indicators(informative)
            dataframe = merge_informative_pair(
                dataframe,
                informative,
                self.timeframe,
                informative_timeframe,
                ffill=True,
            )
            date_column = f"date_{informative_timeframe}"
            dataframe[f"new_{informative_timeframe}"] = (
                dataframe[date_column].notna()
                & dataframe[date_column].ne(dataframe[date_column].shift(1))
            )
        return dataframe

    @staticmethod
    def _bullish_4h(dataframe: DataFrame) -> DataFrame:
        return (dataframe["close_4h"] > dataframe["ema50_4h"]) & (
            dataframe["ema20_4h"] > dataframe["ema50_4h"]
        )

    @staticmethod
    def _bearish_4h(dataframe: DataFrame) -> DataFrame:
        return (dataframe["close_4h"] < dataframe["ema50_4h"]) & (
            dataframe["ema20_4h"] < dataframe["ema50_4h"]
        )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        profile = self.profile_for_pair(metadata["pair"])
        if not profile:
            return dataframe

        signal_tf = profile["signal_tf"]
        new_signal = dataframe[f"new_{signal_tf}"]
        rvol = dataframe[f"rvol20_{signal_tf}"] >= self.rvol_threshold
        candle_up = dataframe[f"candle_up_{signal_tf}"]
        candle_down = dataframe[f"candle_down_{signal_tf}"]

        if profile["kind"] == "breakout":
            long_condition = (
                new_signal
                & (dataframe[f"close_{signal_tf}"] > dataframe[f"donchian_high20_{signal_tf}"])
                & rvol
                & candle_up
                & self._bullish_4h(dataframe)
                & (dataframe["volume"] > 0)
            )
            dataframe.loc[long_condition, ["enter_long", "enter_tag"]] = (
                1,
                f"{signal_tf}_volume_breakout_long",
            )

            if profile["mode"] == "long_short":
                short_condition = (
                    new_signal
                    & (
                        dataframe[f"close_{signal_tf}"]
                        < dataframe[f"donchian_low20_{signal_tf}"]
                    )
                    & rvol
                    & candle_down
                    & self._bearish_4h(dataframe)
                    & (dataframe["volume"] > 0)
                )
                dataframe.loc[short_condition, ["enter_short", "enter_tag"]] = (
                    1,
                    f"{signal_tf}_volume_breakout_short",
                )
        else:
            daily_up = (dataframe["close_1d"] > dataframe["ema20_1d"]) & (
                dataframe["ema20_1d"] > dataframe["ema20_1d"].shift(1)
            )
            daily_down = (dataframe["close_1d"] < dataframe["ema20_1d"]) & (
                dataframe["ema20_1d"] < dataframe["ema20_1d"].shift(1)
            )
            long_condition = (
                new_signal
                & (dataframe["ema20_4h"] > dataframe["ema50_4h"])
                & daily_up
                & rvol
                & candle_up
                & (dataframe["volume"] > 0)
            )
            short_condition = (
                new_signal
                & (dataframe["ema20_4h"] < dataframe["ema50_4h"])
                & daily_down
                & rvol
                & candle_down
                & (dataframe["volume"] > 0)
            )
            dataframe.loc[long_condition, ["enter_long", "enter_tag"]] = (
                1,
                "4h_volume_ema_long",
            )
            dataframe.loc[short_condition, ["enter_short", "enter_tag"]] = (
                1,
                "4h_volume_ema_short",
            )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None

        profile = self.profile_for_pair(metadata["pair"])
        if not profile:
            return dataframe
        signal_tf = profile["signal_tf"]
        new_signal = dataframe[f"new_{signal_tf}"]

        if profile["kind"] == "breakout":
            long_exit = new_signal & (
                dataframe[f"close_{signal_tf}"] < dataframe[f"donchian_low10_{signal_tf}"]
            )
            short_exit = new_signal & (
                dataframe[f"close_{signal_tf}"] > dataframe[f"donchian_high10_{signal_tf}"]
            )
            dataframe.loc[long_exit, ["exit_long", "exit_tag"]] = (
                1,
                f"{signal_tf}_donchian_long_exit",
            )
            dataframe.loc[short_exit, ["exit_short", "exit_tag"]] = (
                1,
                f"{signal_tf}_donchian_short_exit",
            )
        else:
            long_exit = new_signal & (dataframe["ema20_4h"] < dataframe["ema50_4h"])
            short_exit = new_signal & (dataframe["ema20_4h"] > dataframe["ema50_4h"])
            dataframe.loc[long_exit, ["exit_long", "exit_tag"]] = (1, "4h_ema_long_exit")
            dataframe.loc[short_exit, ["exit_short", "exit_tag"]] = (1, "4h_ema_short_exit")
        return dataframe

    def _latest_atr(self, pair: str) -> float | None:
        if not self.dp:
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None
        profile = self.profile_for_pair(pair)
        if profile is None:
            return None
        signal_tf = profile["signal_tf"]
        value = dataframe.iloc[-1].get(f"atr14_{signal_tf}")
        return float(value) if value is not None and np.isfinite(value) and value > 0 else None

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        atr = self._latest_atr(pair)
        if atr is None or current_rate <= 0:
            return 0.0
        total_equity = float(self.wallets.get_total_stake_amount())
        stop_fraction = self.initial_stop_atr * atr / current_rate
        if stop_fraction <= 0:
            return 0.0
        risk_sized_notional = total_equity * self.risk_per_trade / stop_fraction
        capped_notional = min(
            risk_sized_notional,
            total_equity * self.maximum_notional_fraction,
        )
        stake = min(capped_notional / max(leverage, 1.0), max_stake)
        if min_stake is not None and stake < min_stake:
            return 0.0
        return stake

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return 1.0

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        atr = self._latest_atr(pair)
        if atr is None:
            return None

        if trade.is_short:
            initial_stop = trade.open_rate + self.initial_stop_atr * atr
            lowest_rate = min(trade.open_rate, current_rate)
            if trade.min_rate is not None:
                lowest_rate = min(lowest_rate, trade.min_rate)
            trailing_stop = lowest_rate + self.trailing_stop_atr * atr
            stop_rate = min(initial_stop, trailing_stop)
        else:
            initial_stop = trade.open_rate - self.initial_stop_atr * atr
            highest_rate = max(trade.open_rate, current_rate)
            if trade.max_rate is not None:
                highest_rate = max(highest_rate, trade.max_rate)
            trailing_stop = highest_rate - self.trailing_stop_atr * atr
            stop_rate = max(initial_stop, trailing_stop)
        return stoploss_from_absolute(
            stop_rate,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )


class HighVolumeThreeMtfV2(HighVolumeFourMtfV1):
    """Higher-return profile selected by controlled V2 backtests.

    APR is intentionally excluded after producing persistent negative results in
    V1.  Entry rules for AKE, EDEN, and VELVET are unchanged.  V2 doubles the
    per-trade risk budget, raises the notional cap, and lets large trends develop
    before enabling a wider ATR trail.
    """

    risk_per_trade = 0.01
    maximum_notional_fraction = 0.40
    trailing_activation_atr = 3.0
    trailing_stop_atr = 4.5

    pair_profiles = {
        pair: profile.copy()
        for pair, profile in HighVolumeFourMtfV1.pair_profiles.items()
        if pair != "APR/USDT:USDT"
    }

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        atr = self._latest_atr(pair)
        if atr is None:
            return None

        if trade.is_short:
            initial_stop = trade.open_rate + self.initial_stop_atr * atr
            lowest_rate = min(trade.open_rate, current_rate)
            if trade.min_rate is not None:
                lowest_rate = min(lowest_rate, trade.min_rate)
            favorable_excursion = trade.open_rate - lowest_rate
            stop_rate = initial_stop
            if favorable_excursion >= self.trailing_activation_atr * atr:
                stop_rate = min(
                    initial_stop,
                    lowest_rate + self.trailing_stop_atr * atr,
                )
        else:
            initial_stop = trade.open_rate - self.initial_stop_atr * atr
            highest_rate = max(trade.open_rate, current_rate)
            if trade.max_rate is not None:
                highest_rate = max(highest_rate, trade.max_rate)
            favorable_excursion = highest_rate - trade.open_rate
            stop_rate = initial_stop
            if favorable_excursion >= self.trailing_activation_atr * atr:
                stop_rate = max(
                    initial_stop,
                    highest_rate - self.trailing_stop_atr * atr,
                )

        return stoploss_from_absolute(
            stop_rate,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )


class HighVolumeRotationV2(HighVolumeThreeMtfV2):
    """Point-in-time volume-rotation version of V2.

    V2's risk sizing and ATR exit logic are unchanged.  Since a rotating pool
    cannot use the original coin-specific AKE/EDEN/VELVET assignments, every
    whitelisted coin receives V2's existing generic 1h, long/short,
    volume-confirmed Donchian breakout profile.
    """

    rotation_profile = {
        "signal_tf": "1h",
        "mode": "long_short",
        "kind": "breakout",
    }
    # The generic profile only consumes its 1h signal and 4h trend filter.
    # Avoid loading unused 30m/1d informative frames for a much larger pool.
    signal_timeframes = ("1h", "4h")

    informative_columns = {
        "1h": (
            "date", "close", "atr14", "donchian_high20", "donchian_low20",
            "donchian_high10", "donchian_low10", "rvol20", "candle_up",
            "rvol_count6", "quote_volume_growth6", "quote_volume24", "candle_down",
            "close_location", "body_atr14", "atr_percent",
            "breakout_extension_atr", "breakdown_extension_atr",
        ),
        "4h": (
            "date", "close", "ema20", "ema50", "ema20_slope3",
            "ema_separation_atr",
        ),
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Merge only columns consumed by the rotation strategies.

        With roughly fifty pairs, merging every intermediate indicator from
        both informative frames materially increases peak memory.  This keeps
        baseline signals identical while preserving the host's safety margin.
        """
        if not self.dp:
            return dataframe
        for informative_timeframe in self.signal_timeframes:
            informative = self.dp.get_pair_dataframe(
                pair=metadata["pair"], timeframe=informative_timeframe
            )
            informative = self._informative_indicators(informative)
            informative = informative.loc[
                :, self.informative_columns[informative_timeframe]
            ]
            dataframe = merge_informative_pair(
                dataframe,
                informative,
                self.timeframe,
                informative_timeframe,
                ffill=True,
            )
            date_column = f"date_{informative_timeframe}"
            dataframe[f"new_{informative_timeframe}"] = (
                dataframe[date_column].notna()
                & dataframe[date_column].ne(dataframe[date_column].shift(1))
            )
        return dataframe

    def profile_for_pair(self, pair: str) -> dict[str, Any] | None:
        return self.rotation_profile


class HighVolumeIncrementalRotationV2(HighVolumeRotationV2):
    """Continuous May-July walk-forward with an append-only monthly pool.

    The full cumulative whitelist is loaded once so trades can remain open
    across month boundaries.  Entry signals are suppressed until the first
    month in which each pair passed the point-in-time volume screen.
    """

    activation_dates = {
        # May pool.
        "SKYAI/USDT:USDT": "2026-05-01",
        "BSB/USDT:USDT": "2026-05-01",
        "BIO/USDT:USDT": "2026-05-01",
        "MEGA/USDT:USDT": "2026-05-01",
        "1000PEPE/USDT:USDT": "2026-05-01",
        "BR/USDT:USDT": "2026-05-01",
        "ORCA/USDT:USDT": "2026-05-01",
        "NAORIS/USDT:USDT": "2026-05-01",
        "AIOT/USDT:USDT": "2026-05-01",
        "AIGENSYN/USDT:USDT": "2026-05-01",
        "RIVER/USDT:USDT": "2026-05-01",
        "PENGU/USDT:USDT": "2026-05-01",
        # New June pairs.
        "LAB/USDT:USDT": "2026-06-01",
        "PORTAL/USDT:USDT": "2026-06-01",
        "STG/USDT:USDT": "2026-06-01",
        "ALLO/USDT:USDT": "2026-06-01",
        "WLD/USDT:USDT": "2026-06-01",
        "ASTER/USDT:USDT": "2026-06-01",
        "PLAY/USDT:USDT": "2026-06-01",
        "币安人生/USDT:USDT": "2026-06-01",
        "H/USDT:USDT": "2026-06-01",
        # New July pairs (pairs already present retain their earlier date).
        "IN/USDT:USDT": "2026-07-01",
        "SYN/USDT:USDT": "2026-07-01",
        "TAC/USDT:USDT": "2026-07-01",
        "RE/USDT:USDT": "2026-07-01",
        "SLX/USDT:USDT": "2026-07-01",
        "UB/USDT:USDT": "2026-07-01",
        "VELVET/USDT:USDT": "2026-07-01",
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        activation = self.activation_dates.get(metadata["pair"])
        if activation is None:
            dataframe[["enter_long", "enter_short"]] = 0
            dataframe["enter_tag"] = None
            return dataframe

        inactive = dataframe["date"] < pd.Timestamp(activation, tz="UTC")
        dataframe.loc[inactive, ["enter_long", "enter_short"]] = 0
        dataframe.loc[inactive, "enter_tag"] = None
        return dataframe


class HighVolumeRollingEntryRotationV2(HighVolumeRotationV2):
    """Monthly entry whitelist with natural exits for carried positions.

    A pair can open a new position only while it belongs to that month's
    point-in-time screen.  Removing a pair suppresses entries only: open
    trades keep receiving all normal stoploss, trailing and trend exits.
    """

    monthly_entry_pools = (
        (
            "2026-05-01",
            "2026-06-01",
            frozenset(
                {
                    "SKYAI/USDT:USDT", "BSB/USDT:USDT", "BIO/USDT:USDT",
                    "MEGA/USDT:USDT", "1000PEPE/USDT:USDT", "BR/USDT:USDT",
                    "ORCA/USDT:USDT", "NAORIS/USDT:USDT", "AIOT/USDT:USDT",
                    "AIGENSYN/USDT:USDT", "RIVER/USDT:USDT", "PENGU/USDT:USDT",
                }
            ),
        ),
        (
            "2026-06-01",
            "2026-07-01",
            frozenset(
                {
                    "LAB/USDT:USDT", "PORTAL/USDT:USDT", "STG/USDT:USDT",
                    "ALLO/USDT:USDT", "WLD/USDT:USDT", "ASTER/USDT:USDT",
                    "PLAY/USDT:USDT", "币安人生/USDT:USDT", "H/USDT:USDT",
                }
            ),
        ),
        (
            "2026-07-01",
            "2026-08-01",
            frozenset(
                {
                    "IN/USDT:USDT", "SYN/USDT:USDT", "LAB/USDT:USDT",
                    "AIGENSYN/USDT:USDT", "TAC/USDT:USDT", "RE/USDT:USDT",
                    "WLD/USDT:USDT", "SLX/USDT:USDT", "UB/USDT:USDT",
                    "VELVET/USDT:USDT", "1000PEPE/USDT:USDT",
                }
            ),
        ),
    )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        pair = metadata["pair"]
        active = pd.Series(False, index=dataframe.index)
        for start, end, pool in self.monthly_entry_pools:
            if pair in pool:
                active |= (dataframe["date"] >= pd.Timestamp(start, tz="UTC")) & (
                    dataframe["date"] < pd.Timestamp(end, tz="UTC")
                )

        dataframe.loc[~active, ["enter_long", "enter_short"]] = 0
        dataframe.loc[~active, "enter_tag"] = None
        return dataframe


class HighVolumeExtendedRollingEntryRotationV2(HighVolumeRollingEntryRotationV2):
    """Point-in-time monthly entry pools from February through August 2026."""

    monthly_entry_pools = (
        (
            "2026-02-01", "2026-03-01",
            frozenset({
                "BULLA/USDT:USDT", "1000PEPE/USDT:USDT", "SENT/USDT:USDT",
                "RIVER/USDT:USDT", "ENSO/USDT:USDT", "SYN/USDT:USDT",
            }),
        ),
        (
            "2026-03-01", "2026-04-01",
            frozenset({
                "POWER/USDT:USDT", "1000PEPE/USDT:USDT", "SAHARA/USDT:USDT",
                "PIPPIN/USDT:USDT", "RIVER/USDT:USDT", "GWEI/USDT:USDT",
                "DOT/USDT:USDT",
            }),
        ),
        (
            "2026-04-01", "2026-05-01",
            frozenset({
                "SIREN/USDT:USDT", "KERNEL/USDT:USDT", "RIVER/USDT:USDT",
                "ONT/USDT:USDT", "1000PEPE/USDT:USDT", "STO/USDT:USDT",
                "PIPPIN/USDT:USDT", "NOM/USDT:USDT", "NIGHT/USDT:USDT",
            }),
        ),
        HighVolumeRollingEntryRotationV2.monthly_entry_pools[0],
        HighVolumeRollingEntryRotationV2.monthly_entry_pools[1],
        HighVolumeRollingEntryRotationV2.monthly_entry_pools[2],
        (
            "2026-08-01", "2026-09-01",
            frozenset({
                "KOMA/USDT:USDT", "BANK/USDT:USDT", "MMT/USDT:USDT",
                "GIGGLE/USDT:USDT", "COTI/USDT:USDT", "1000RATS/USDT:USDT",
                "UNI/USDT:USDT", "AKE/USDT:USDT", "1000PEPE/USDT:USDT",
            }),
        ),
    )


class HighVolumeExtendedRollingEntryRotation30mBaseline(
    HighVolumeExtendedRollingEntryRotationV2
):
    """Low-memory 30m execution baseline with unchanged 1h/4h signals."""

    timeframe = "30m"
    # Preserve the original roughly 23-day 15m warmup in wall-clock time.
    startup_candle_count = 1100


class HighVolumeMainWaveExperimentBase(
    HighVolumeExtendedRollingEntryRotation30mBaseline
):
    """Controlled variants aimed at capturing persistent directional waves.

    Defaults are neutral and reproduce the V2 entry/exit logic.  Subclasses
    tighten signal-candle quality and 4h trend strength and/or slow the
    Donchian/trailing exits.  Long and short entries remain enabled.
    """

    minimum_close_location = 0.0
    minimum_body_atr = 0.0
    minimum_ema_separation_atr = 0.0
    require_ema20_slope = False
    exit_donchian_period = 10

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]

        long_quality = pd.Series(True, index=dataframe.index)
        short_quality = pd.Series(True, index=dataframe.index)

        if self.minimum_close_location > 0:
            long_quality &= (
                dataframe[f"close_location_{signal_tf}"]
                >= self.minimum_close_location
            )
            short_quality &= (
                dataframe[f"close_location_{signal_tf}"]
                <= 1 - self.minimum_close_location
            )
        if self.minimum_body_atr > 0:
            body_quality = (
                dataframe[f"body_atr14_{signal_tf}"] >= self.minimum_body_atr
            )
            long_quality &= body_quality
            short_quality &= body_quality
        if self.minimum_ema_separation_atr > 0:
            separation = dataframe["ema_separation_atr_4h"]
            long_quality &= separation >= self.minimum_ema_separation_atr
            short_quality &= separation <= -self.minimum_ema_separation_atr
        if self.require_ema20_slope:
            slope = dataframe["ema20_slope3_4h"]
            long_quality &= slope > 0
            short_quality &= slope < 0

        rejected_long = dataframe["enter_long"].eq(1) & ~long_quality.fillna(False)
        rejected_short = dataframe["enter_short"].eq(1) & ~short_quality.fillna(False)
        dataframe.loc[rejected_long, "enter_long"] = 0
        dataframe.loc[rejected_short, "enter_short"] = 0
        dataframe.loc[rejected_long | rejected_short, "enter_tag"] = None
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.exit_donchian_period == 10:
            return super().populate_exit_trend(dataframe, metadata)

        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        signal_tf = self.rotation_profile["signal_tf"]
        new_signal = dataframe[f"new_{signal_tf}"]
        period = self.exit_donchian_period
        long_exit = new_signal & (
            dataframe[f"close_{signal_tf}"]
            < dataframe[f"donchian_low{period}_{signal_tf}"]
        )
        short_exit = new_signal & (
            dataframe[f"close_{signal_tf}"]
            > dataframe[f"donchian_high{period}_{signal_tf}"]
        )
        dataframe.loc[long_exit, ["exit_long", "exit_tag"]] = (
            1,
            f"{signal_tf}_donchian{period}_long_exit",
        )
        dataframe.loc[short_exit, ["exit_short", "exit_tag"]] = (
            1,
            f"{signal_tf}_donchian{period}_short_exit",
        )
        return dataframe


class HighVolumeMainWaveHoldV1(HighVolumeMainWaveExperimentBase):
    """Baseline entries with slower exits so established waves can develop."""

    exit_donchian_period = 20
    trailing_activation_atr = 4.0
    trailing_stop_atr = 6.0


class HighVolumeMainWaveQualityV1(HighVolumeMainWaveExperimentBase):
    """Baseline exits with stricter volume, candle and 4h trend quality."""

    rvol_threshold = 1.25
    minimum_close_location = 0.65
    minimum_body_atr = 0.25
    minimum_ema_separation_atr = 0.20
    require_ema20_slope = True


class HighVolumeMainWaveBalancedV1(HighVolumeMainWaveExperimentBase):
    """Moderate quality filters combined with slower main-wave exits."""

    rvol_threshold = 1.15
    minimum_close_location = 0.60
    minimum_body_atr = 0.15
    minimum_ema_separation_atr = 0.10
    require_ema20_slope = True
    exit_donchian_period = 20
    trailing_activation_atr = 4.0
    trailing_stop_atr = 6.0


class HighVolumeMainWaveAsymmetricV2(
    HighVolumeExtendedRollingEntryRotation30mBaseline
):
    """Treat upside and downside breakouts differently.

    The first experiment showed that quality filters and slower exits improved
    longs but materially damaged shorts.  V2 therefore applies those filters
    only to longs.  Shorts retain the original breakout and faster exit/trail.
    """

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]
        long_quality = (
            (dataframe[f"rvol20_{signal_tf}"] >= 1.15)
            & (dataframe[f"close_location_{signal_tf}"] >= 0.60)
            & (dataframe[f"body_atr14_{signal_tf}"] >= 0.15)
            & (dataframe["ema_separation_atr_4h"] >= 0.10)
            & (dataframe["ema20_slope3_4h"] > 0)
        ).fillna(False)
        rejected_long = dataframe["enter_long"].eq(1) & ~long_quality
        dataframe.loc[rejected_long, "enter_long"] = 0
        dataframe.loc[rejected_long, "enter_tag"] = None
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        signal_tf = self.rotation_profile["signal_tf"]
        new_signal = dataframe[f"new_{signal_tf}"]
        long_exit = new_signal & (
            dataframe[f"close_{signal_tf}"]
            < dataframe[f"donchian_low20_{signal_tf}"]
        )
        short_exit = new_signal & (
            dataframe[f"close_{signal_tf}"]
            > dataframe[f"donchian_high10_{signal_tf}"]
        )
        dataframe.loc[long_exit, ["exit_long", "exit_tag"]] = (
            1,
            f"{signal_tf}_donchian20_long_exit",
        )
        dataframe.loc[short_exit, ["exit_short", "exit_tag"]] = (
            1,
            f"{signal_tf}_donchian10_short_exit",
        )
        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        atr = self._latest_atr(pair)
        if atr is None:
            return None

        if trade.is_short:
            initial_stop = trade.open_rate + self.initial_stop_atr * atr
            lowest_rate = min(trade.open_rate, current_rate)
            if trade.min_rate is not None:
                lowest_rate = min(lowest_rate, trade.min_rate)
            favorable_excursion = trade.open_rate - lowest_rate
            stop_rate = initial_stop
            if favorable_excursion >= 3.0 * atr:
                stop_rate = min(initial_stop, lowest_rate + 4.5 * atr)
        else:
            initial_stop = trade.open_rate - self.initial_stop_atr * atr
            highest_rate = max(trade.open_rate, current_rate)
            if trade.max_rate is not None:
                highest_rate = max(highest_rate, trade.max_rate)
            favorable_excursion = highest_rate - trade.open_rate
            stop_rate = initial_stop
            if favorable_excursion >= 4.0 * atr:
                stop_rate = max(initial_stop, highest_rate - 6.0 * atr)

        return stoploss_from_absolute(
            stop_rate,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )


class HighVolumeMainWaveAsymmetricAntiCapV2(HighVolumeMainWaveAsymmetricV2):
    """Avoid opening shorts on likely high-volume capitulation candles."""

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]
        short_quality = (
            (dataframe[f"rvol20_{signal_tf}"] <= 2.5)
            & (dataframe[f"body_atr14_{signal_tf}"] <= 1.25)
            & (dataframe[f"close_location_{signal_tf}"] >= 0.08)
        ).fillna(False)
        rejected_short = dataframe["enter_short"].eq(1) & ~short_quality
        dataframe.loc[rejected_short, "enter_short"] = 0
        dataframe.loc[rejected_short, "enter_tag"] = None
        return dataframe


class HighVolumeMainWaveVolatilityV3(
    HighVolumeExtendedRollingEntryRotation30mBaseline
):
    """Trade breakouts only after the pair has entered a volatile regime."""

    minimum_atr_percent = 0.03

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]
        volatile = (
            dataframe[f"atr_percent_{signal_tf}"] >= self.minimum_atr_percent
        ).fillna(False)
        rejected = (
            dataframe["enter_long"].eq(1) | dataframe["enter_short"].eq(1)
        ) & ~volatile
        dataframe.loc[rejected, ["enter_long", "enter_short"]] = 0
        dataframe.loc[rejected, "enter_tag"] = None
        return dataframe


class HighVolumeMainWaveSelectiveV3(HighVolumeMainWaveVolatilityV3):
    """Volatile-regime breakouts without chasing an already extended candle."""

    minimum_long_extension_atr = 0.10
    maximum_long_extension_atr = 0.50
    maximum_short_extension_atr = 0.25

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]
        long_extension = dataframe[f"breakout_extension_atr_{signal_tf}"]
        short_extension = dataframe[f"breakdown_extension_atr_{signal_tf}"]
        long_quality = long_extension.between(
            self.minimum_long_extension_atr,
            self.maximum_long_extension_atr,
            inclusive="both",
        ).fillna(False)
        short_quality = short_extension.between(
            0.0,
            self.maximum_short_extension_atr,
            inclusive="both",
        ).fillna(False)
        rejected_long = dataframe["enter_long"].eq(1) & ~long_quality
        rejected_short = dataframe["enter_short"].eq(1) & ~short_quality
        dataframe.loc[rejected_long, "enter_long"] = 0
        dataframe.loc[rejected_short, "enter_short"] = 0
        dataframe.loc[rejected_long | rejected_short, "enter_tag"] = None
        return dataframe


class HighVolumeMainWaveScoutV4(
    HighVolumeExtendedRollingEntryRotation30mBaseline
):
    """Full breakout coverage with confidence-tiered position sizing.

    Selective V3 signals receive the normal risk budget.  Other valid V2
    breakouts remain tradable as quarter-sized scout positions so an early or
    atypical main wave is not discarded entirely.
    """

    scout_stake_fraction = 0.25
    scout_stake_fraction_long = 0.25
    scout_stake_fraction_short = 0.25
    scout_minimum_stake_floor = False
    minimum_atr_percent = 0.03
    minimum_long_extension_atr = 0.10
    maximum_long_extension_atr = 0.50
    maximum_short_extension_atr = 0.25

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]
        volatile = (
            dataframe[f"atr_percent_{signal_tf}"] >= self.minimum_atr_percent
        )
        long_extension = dataframe[f"breakout_extension_atr_{signal_tf}"]
        short_extension = dataframe[f"breakdown_extension_atr_{signal_tf}"]
        mainwave_long = (
            volatile
            & long_extension.between(
                self.minimum_long_extension_atr,
                self.maximum_long_extension_atr,
                inclusive="both",
            )
        ).fillna(False)
        mainwave_short = (
            volatile
            & short_extension.between(
                0.0,
                self.maximum_short_extension_atr,
                inclusive="both",
            )
        ).fillna(False)

        long_entry = dataframe["enter_long"].eq(1)
        short_entry = dataframe["enter_short"].eq(1)
        dataframe.loc[long_entry & mainwave_long, "enter_tag"] = (
            f"{signal_tf}_mainwave_breakout_long"
        )
        dataframe.loc[short_entry & mainwave_short, "enter_tag"] = (
            f"{signal_tf}_mainwave_breakout_short"
        )
        dataframe.loc[long_entry & ~mainwave_long, "enter_tag"] = (
            f"{signal_tf}_scout_breakout_long"
        )
        dataframe.loc[short_entry & ~mainwave_short, "enter_tag"] = (
            f"{signal_tf}_scout_breakout_short"
        )
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        stake = super().custom_stake_amount(
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
        if entry_tag and "_scout_" in entry_tag:
            fraction = (
                self.scout_stake_fraction_short
                if side == "short"
                else self.scout_stake_fraction_long
            )
            stake *= fraction
            if min_stake is not None and stake < min_stake:
                if self.scout_minimum_stake_floor and min_stake <= max_stake:
                    return min_stake
                return 0.0
        return stake


class HighVolumeMainWaveScoutLooseV4(HighVolumeMainWaveScoutV4):
    """Nearby looser confidence boundary for robustness testing."""

    minimum_atr_percent = 0.025
    minimum_long_extension_atr = 0.05
    maximum_long_extension_atr = 0.65
    maximum_short_extension_atr = 0.35


class HighVolumeMainWaveScoutTightV4(HighVolumeMainWaveScoutV4):
    """Nearby tighter confidence boundary for robustness testing."""

    minimum_atr_percent = 0.035
    minimum_long_extension_atr = 0.15
    maximum_long_extension_atr = 0.40
    maximum_short_extension_atr = 0.20


class HighVolumeMainWaveScoutFastFailV5(HighVolumeMainWaveScoutV4):
    """Release scout slots when a breakout never develops follow-through.

    Main-wave positions retain V4's deliberately wide ATR stop and normal
    Donchian/trailing exits.  A quarter-sized scout is closed only after it has
    had six hours to prove itself, is currently losing, and never travelled one
    current ATR in the favorable direction.  The rule uses only live trade
    extrema and the latest completed 1h ATR, so it is safe from lookahead.
    """

    scout_confirmation_hours = 6
    scout_confirmation_hours_long = 6
    scout_confirmation_hours_short = 6
    scout_minimum_favorable_atr = 1.0

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        if not trade.enter_tag or "_scout_" not in trade.enter_tag:
            return None
        confirmation_hours = (
            self.scout_confirmation_hours_short
            if trade.is_short
            else self.scout_confirmation_hours_long
        )
        if current_time < trade.open_date_utc + timedelta(hours=confirmation_hours):
            return None
        if current_profit > 0:
            return None

        atr = self._latest_atr(pair)
        if atr is None:
            return None
        if trade.is_short:
            best_rate = min(trade.open_rate, current_rate)
            if trade.min_rate is not None:
                best_rate = min(best_rate, trade.min_rate)
            favorable_excursion = trade.open_rate - best_rate
        else:
            best_rate = max(trade.open_rate, current_rate)
            if trade.max_rate is not None:
                best_rate = max(best_rate, trade.max_rate)
            favorable_excursion = best_rate - trade.open_rate

        if favorable_excursion < self.scout_minimum_favorable_atr * atr:
            return "scout_failed_followthrough"
        return None


class HighVolumeMainWaveScoutFastFailEarlyV5(
    HighVolumeMainWaveScoutFastFailV5
):
    """Nearby timing check: require scout follow-through within three hours."""

    scout_confirmation_hours = 3
    scout_confirmation_hours_long = 3
    scout_confirmation_hours_short = 3


class HighVolumeMainWaveScoutFastFailLateV5(
    HighVolumeMainWaveScoutFastFailV5
):
    """Nearby timing check: give scouts nine hours to confirm."""

    scout_confirmation_hours = 9
    scout_confirmation_hours_long = 9
    scout_confirmation_hours_short = 9


class HighVolumeMainWaveScoutDirectionalTimingV5(
    HighVolumeMainWaveScoutFastFailV5
):
    """Factor-isolation check: nine-hour longs and six-hour shorts."""

    scout_confirmation_hours_long = 9
    scout_confirmation_hours_short = 6


class HighVolumeMainWaveScoutDirectionalV6(
    HighVolumeMainWaveScoutDirectionalTimingV5
):
    """Direction-aware scout timing and risk based on stable cohort behavior.

    Scout longs have not shown a durable edge, especially in the later months,
    so they retain coverage with a smaller one-tenth risk tier and receive nine
    hours to confirm.  Scout shorts retain the quarter-risk tier and the more
    responsive six-hour failure check.  Main-wave sizing stays unchanged.
    """

    scout_stake_fraction_long = 0.10
    scout_stake_fraction_short = 0.25


class HighVolumeMainWaveScoutDirectionalLightV6(
    HighVolumeMainWaveScoutDirectionalV6
):
    """Nearby risk check with five-percent scout-long sizing."""

    scout_stake_fraction_long = 0.05


class HighVolumeMainWaveScoutDirectionalHeavyV6(
    HighVolumeMainWaveScoutDirectionalV6
):
    """Nearby risk check with fifteen-percent scout-long sizing."""

    scout_stake_fraction_long = 0.15


class HighVolumeMainWaveScoutDirectionalMidLowV6(
    HighVolumeMainWaveScoutDirectionalV6
):
    """Local robustness check below the fifteen-percent candidate."""

    scout_stake_fraction_long = 0.125


class HighVolumeMainWaveScoutDirectionalMidHighV6(
    HighVolumeMainWaveScoutDirectionalV6
):
    """Local robustness check above the fifteen-percent candidate."""

    scout_stake_fraction_long = 0.175


class HighVolumeMainWaveScoutMinFloorV7(
    HighVolumeMainWaveScoutDirectionalHeavyV6
):
    """Make small-account scout coverage independent of min-notional cliffs.

    When a fractionally sized scout is below Binance's minimum stake, use the
    minimum valid stake instead of silently dropping the signal.  This keeps
    the scout universe stable as the wallet changes and prevents tiny sizing
    differences from changing which later main-wave trades obtain a slot.
    """

    scout_minimum_stake_floor = True


class HighVolumeMainWaveScoutMinFloorLightV7(
    HighVolumeMainWaveScoutMinFloorV7
):
    """Minimum-notional robustness check with ten-percent scout longs."""

    scout_stake_fraction_long = 0.10


class HighVolumeMainWaveScoutMinFloorExtraLightV7(
    HighVolumeMainWaveScoutMinFloorV7
):
    """Minimum-notional robustness check with five-percent scout longs."""

    scout_stake_fraction_long = 0.05


class HighVolumeMainWaveScoutMinOnlyLongV7(
    HighVolumeMainWaveScoutMinFloorV7
):
    """Keep weak-edge scout longs at the exchange minimum valid stake."""

    scout_stake_fraction_long = 0.0


class HighVolumeMainWaveScoutMinOnlyFastV7(
    HighVolumeMainWaveScoutMinOnlyLongV7
):
    """Timing robustness check with six-hour confirmation on both sides."""

    scout_confirmation_hours_long = 6
    scout_confirmation_hours_short = 6


class HighVolumeMainWaveScoutMinOnlyLateV7(
    HighVolumeMainWaveScoutMinOnlyLongV7
):
    """Timing robustness check with nine-hour confirmation on both sides."""

    scout_confirmation_hours_long = 9
    scout_confirmation_hours_short = 9


class HighVolumeMainWaveV7(HighVolumeMainWaveScoutMinOnlyLateV7):
    """Current robust candidate for the monthly high-volume rotation system."""


class HighVolumeMainWavePersistentVolumeV9(HighVolumeMainWaveV7):
    """Risk-tier scout shorts by six-hour relative-volume persistence.

    A one-hour downside breakout can be caused by a single liquidation spike.
    Scout shorts retain V7's quarter-risk stake only when at least four of the
    latest six completed 1h candles have relative volume of one or more.  Thin
    volume breakouts remain tradable at Binance's minimum stake so main-wave
    coverage is not deliberately discarded.
    """

    persistent_volume_hours = 4

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]
        thin_scout_short = (
            dataframe["enter_short"].eq(1)
            & dataframe["enter_tag"].eq(f"{signal_tf}_scout_breakout_short")
            & (
                dataframe[f"rvol_count6_{signal_tf}"]
                < self.persistent_volume_hours
            )
        ).fillna(False)
        dataframe.loc[thin_scout_short, "enter_tag"] = (
            f"{signal_tf}_thinvolume_scout_breakout_short"
        )
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        if entry_tag and "_thinvolume_scout_" in entry_tag:
            if min_stake is None or min_stake > max_stake:
                return 0.0
            return min_stake
        return super().custom_stake_amount(
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


class HighVolumeMainWavePersistentVolumeLooseV9(
    HighVolumeMainWavePersistentVolumeV9
):
    """Nearby persistence check requiring three of six volume hours."""

    persistent_volume_hours = 3


class HighVolumeMainWavePersistentVolumeTightV9(
    HighVolumeMainWavePersistentVolumeV9
):
    """Nearby persistence check requiring five of six volume hours."""

    persistent_volume_hours = 5


class HighVolumeMainWavePersistentStrengthV10(
    HighVolumeMainWavePersistentVolumeV9
):
    """Modestly amplify persistent-volume main-wave longs.

    The qualifying long signals were strongly profitable in both the early and
    late chronological cohorts.  Their risk-sized stake is multiplied by 1.25
    while retaining the existing forty-percent equity notional cap.  Signals,
    exits, scout sizing, short sizing, and maximum open trades are unchanged.
    """

    persistent_mainwave_long_multiplier = 1.25

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]
        persistent_mainwave_long = (
            dataframe["enter_long"].eq(1)
            & dataframe["enter_tag"].eq(f"{signal_tf}_mainwave_breakout_long")
            & (
                dataframe[f"rvol_count6_{signal_tf}"]
                >= self.persistent_volume_hours
            )
        ).fillna(False)
        dataframe.loc[persistent_mainwave_long, "enter_tag"] = (
            f"{signal_tf}_persistent_mainwave_breakout_long"
        )
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        stake = super().custom_stake_amount(
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
        if entry_tag and "_persistent_mainwave_breakout_long" in entry_tag:
            total_equity = float(self.wallets.get_total_stake_amount())
            notional_cap_stake = (
                total_equity
                * self.maximum_notional_fraction
                / max(leverage, 1.0)
            )
            stake = min(
                stake * self.persistent_mainwave_long_multiplier,
                notional_cap_stake,
                max_stake,
            )
        return stake


class HighVolumeMainWavePersistentStrengthConservativeV10(
    HighVolumeMainWavePersistentStrengthV10
):
    """Nearby strength-sizing check using a 1.15 multiplier."""

    persistent_mainwave_long_multiplier = 1.15


class HighVolumeMainWavePersistentStrengthAggressiveV10(
    HighVolumeMainWavePersistentStrengthV10
):
    """Nearby strength-sizing check using a 1.50 multiplier."""

    persistent_mainwave_long_multiplier = 1.50


class HighVolumeMainWavePersistentStrengthExtendedV10(
    HighVolumeMainWavePersistentStrengthV10
):
    """Upper-bound check using a 1.75 persistent-long multiplier."""

    persistent_mainwave_long_multiplier = 1.75


class HighVolumeMainWavePersistentStrengthDoubleV10(
    HighVolumeMainWavePersistentStrengthV10
):
    """Selected V10 candidate using a 2.00 persistent-long multiplier.

    This represents about two percent planned risk on the strongest main-wave
    longs.  The inherited forty-percent equity notional cap remains active.
    """

    persistent_mainwave_long_multiplier = 2.00


class HighVolumeMainWavePersistentStrengthDoubleLooseV10(
    HighVolumeMainWavePersistentStrengthDoubleV10
):
    """Combined robustness check using three of six persistent-volume hours."""

    persistent_volume_hours = 3


class HighVolumeMainWavePersistentStrengthDoubleTightV10(
    HighVolumeMainWavePersistentStrengthDoubleV10
):
    """Combined robustness check using five of six persistent-volume hours."""

    persistent_volume_hours = 5


class HighVolumeMainWaveFollowthroughExitV11(
    HighVolumeMainWavePersistentStrengthDoubleV10
):
    """Release a main-wave slot when the breakout never proves follow-through.

    Entry coverage and confidence sizing stay identical to V10.  Once a
    main-wave trade has had six hours to develop, it is closed only while it is
    losing and its best excursion has never reached one current 1h ATR.  A
    trade that has demonstrated one ATR of favorable movement is never subject
    to this failure exit, even after a later pullback.
    """

    mainwave_confirmation_hours = 6
    mainwave_minimum_favorable_atr = 1.0

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        inherited_exit = super().custom_exit(
            pair=pair,
            trade=trade,
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            **kwargs,
        )
        if inherited_exit is not None:
            return inherited_exit
        if not trade.enter_tag or "_mainwave_" not in trade.enter_tag:
            return None
        if current_time < trade.open_date_utc + timedelta(
            hours=self.mainwave_confirmation_hours
        ):
            return None
        if current_profit > 0:
            return None

        atr = self._latest_atr(pair)
        if atr is None:
            return None
        if trade.is_short:
            best_rate = min(trade.open_rate, current_rate)
            if trade.min_rate is not None:
                best_rate = min(best_rate, trade.min_rate)
            favorable_excursion = trade.open_rate - best_rate
        else:
            best_rate = max(trade.open_rate, current_rate)
            if trade.max_rate is not None:
                best_rate = max(best_rate, trade.max_rate)
            favorable_excursion = best_rate - trade.open_rate

        if favorable_excursion < self.mainwave_minimum_favorable_atr * atr:
            return "mainwave_failed_followthrough"
        return None


class HighVolumeMainWaveFollowthroughExitEarlyV11(
    HighVolumeMainWaveFollowthroughExitV11
):
    """Nearby timing check requiring proof within three hours."""

    mainwave_confirmation_hours = 3


class HighVolumeMainWaveFollowthroughExitLateV11(
    HighVolumeMainWaveFollowthroughExitV11
):
    """Nearby timing check allowing nine hours for proof."""

    mainwave_confirmation_hours = 9


class HighVolumeMainWaveFollowthroughExitExtraLateV11(
    HighVolumeMainWaveFollowthroughExitV11
):
    """Nearby timing check allowing twelve hours for proof."""

    mainwave_confirmation_hours = 12


class HighVolumeMainWaveTrendAlignedStrengthV11(
    HighVolumeMainWavePersistentStrengthDoubleV10
):
    """Grant the V10 double-risk tier only while the 4h EMA20 is rising.

    All valid breakouts still enter, so signal coverage and slot occupancy are
    unchanged.  Persistent-volume main-wave longs whose three-bar 4h EMA20
    slope is below the threshold are simply restored to the ordinary one-
    percent risk tier rather than receiving V10's two-percent tier.
    """

    minimum_persistent_long_slope = 0.0

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]
        unaligned = (
            dataframe["enter_long"].eq(1)
            & dataframe["enter_tag"].eq(
                f"{signal_tf}_persistent_mainwave_breakout_long"
            )
            & (
                dataframe["ema20_slope3_4h"]
                < self.minimum_persistent_long_slope
            )
        ).fillna(False)
        dataframe.loc[unaligned, "enter_tag"] = (
            f"{signal_tf}_mainwave_breakout_long"
        )
        return dataframe


class HighVolumeMainWaveTrendAlignedStrengthLooseV11(
    HighVolumeMainWaveTrendAlignedStrengthV11
):
    """Nearby alignment check allowing a mildly falling 4h EMA20."""

    minimum_persistent_long_slope = -0.005


class HighVolumeMainWaveTrendAlignedStrengthVeryLooseV11(
    HighVolumeMainWaveTrendAlignedStrengthV11
):
    """Broader alignment check allowing a one-percent 4h EMA20 decline."""

    minimum_persistent_long_slope = -0.01


class HighVolumeMainWaveTrendAlignedStrengthTightV11(
    HighVolumeMainWaveTrendAlignedStrengthV11
):
    """Nearby alignment check requiring a modestly rising 4h EMA20."""

    minimum_persistent_long_slope = 0.005


class HighVolumeMainWaveProfitLockV12(
    HighVolumeMainWaveTrendAlignedStrengthLooseV11
):
    """Tighten the trail only after a persistent long proves a large wave.

    V11 entries, sizing, shorts, scouts, and ordinary exits remain unchanged.
    Persistent-volume main-wave longs retain the inherited 4.5 ATR trail until
    their favorable excursion reaches six current 1h ATR.  Only then is the
    trail tightened to 3.5 ATR to reduce late-stage profit giveback.
    """

    persistent_lock_activation_atr = 6.0
    persistent_lock_trailing_atr = 3.5

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        if (
            trade.is_short
            or not trade.enter_tag
            or "_persistent_mainwave_breakout_long" not in trade.enter_tag
        ):
            return super().custom_stoploss(
                pair=pair,
                trade=trade,
                current_time=current_time,
                current_rate=current_rate,
                current_profit=current_profit,
                after_fill=after_fill,
                **kwargs,
            )

        atr = self._latest_atr(pair)
        if atr is None:
            return None
        highest_rate = max(trade.open_rate, current_rate)
        if trade.max_rate is not None:
            highest_rate = max(highest_rate, trade.max_rate)
        favorable_excursion = highest_rate - trade.open_rate
        if favorable_excursion < self.persistent_lock_activation_atr * atr:
            return super().custom_stoploss(
                pair=pair,
                trade=trade,
                current_time=current_time,
                current_rate=current_rate,
                current_profit=current_profit,
                after_fill=after_fill,
                **kwargs,
            )

        stop_rate = max(
            trade.open_rate - self.initial_stop_atr * atr,
            highest_rate - self.persistent_lock_trailing_atr * atr,
        )
        return stoploss_from_absolute(
            stop_rate,
            current_rate=current_rate,
            is_short=False,
            leverage=trade.leverage,
        )


class HighVolumeMainWaveProfitLockEarlyV12(HighVolumeMainWaveProfitLockV12):
    """Nearby activation check tightening after five favorable ATR."""

    persistent_lock_activation_atr = 5.0


class HighVolumeMainWaveProfitLockLateV12(HighVolumeMainWaveProfitLockV12):
    """Nearby activation check tightening after seven favorable ATR."""

    persistent_lock_activation_atr = 7.0


class HighVolumeMainWaveProfitLockTightV12(HighVolumeMainWaveProfitLockV12):
    """Nearby distance check using a three-ATR profit-locking trail."""

    persistent_lock_trailing_atr = 3.0


class HighVolumeMainWaveProfitLockWideV12(HighVolumeMainWaveProfitLockV12):
    """Nearby distance check using a four-ATR profit-locking trail."""

    persistent_lock_trailing_atr = 4.0


class HighVolumeMainWaveEarlyTrendScoutV13(
    HighVolumeMainWaveTrendAlignedStrengthLooseV11
):
    """Probe persistent breakouts before the slow 4h EMA cross completes.

    V11 signals remain untouched.  An additional minimum-stake scout is
    permitted when a completed 1h candle breaks the twenty-hour boundary with
    persistent volume in a volatile regime, while 4h price and EMA20 slope
    already agree with the direction but EMA20 has not yet crossed EMA50.
    Monthly point-in-time entry pools are reapplied explicitly so the new
    signal cannot trade a pair outside its selected month.
    """

    earlytrend_persistent_hours = 4
    earlytrend_minimum_atr_percent = 0.03
    earlytrend_short_stake_fraction = 0.0
    earlytrend_long_enabled = True
    earlytrend_short_enabled = True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]
        pair = metadata["pair"]

        active = pd.Series(False, index=dataframe.index)
        for start, end, pool in self.monthly_entry_pools:
            if pair in pool:
                active |= (dataframe["date"] >= pd.Timestamp(start, tz="UTC")) & (
                    dataframe["date"] < pd.Timestamp(end, tz="UTC")
                )

        common = (
            active
            & dataframe[f"new_{signal_tf}"]
            & (
                dataframe[f"rvol20_{signal_tf}"]
                >= self.rvol_threshold
            )
            & (
                dataframe[f"rvol_count6_{signal_tf}"]
                >= self.earlytrend_persistent_hours
            )
            & (
                dataframe[f"atr_percent_{signal_tf}"]
                >= self.earlytrend_minimum_atr_percent
            )
            & (dataframe["volume"] > 0)
        )
        early_long = (
            self.earlytrend_long_enabled
            & common
            & dataframe["enter_long"].eq(0)
            & (
                dataframe[f"close_{signal_tf}"]
                > dataframe[f"donchian_high20_{signal_tf}"]
            )
            & dataframe[f"candle_up_{signal_tf}"]
            & (dataframe["close_4h"] > dataframe["ema20_4h"])
            & (dataframe["ema20_slope3_4h"] > 0)
            & ~self._bullish_4h(dataframe)
        ).fillna(False)
        early_short = (
            self.earlytrend_short_enabled
            & common
            & dataframe["enter_short"].eq(0)
            & (
                dataframe[f"close_{signal_tf}"]
                < dataframe[f"donchian_low20_{signal_tf}"]
            )
            & dataframe[f"candle_down_{signal_tf}"]
            & (dataframe["close_4h"] < dataframe["ema20_4h"])
            & (dataframe["ema20_slope3_4h"] < 0)
            & ~self._bearish_4h(dataframe)
        ).fillna(False)
        dataframe.loc[early_long, ["enter_long", "enter_tag"]] = (
            1,
            f"{signal_tf}_earlytrend_scout_breakout_long",
        )
        dataframe.loc[early_short, ["enter_short", "enter_tag"]] = (
            1,
            f"{signal_tf}_earlytrend_scout_breakout_short",
        )
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        if entry_tag and "_earlytrend_scout_" in entry_tag:
            if side == "short" and self.earlytrend_short_stake_fraction > 0:
                full_stake = super().custom_stake_amount(
                    pair=pair,
                    current_time=current_time,
                    current_rate=current_rate,
                    proposed_stake=proposed_stake,
                    min_stake=min_stake,
                    max_stake=max_stake,
                    leverage=leverage,
                    entry_tag=entry_tag.replace("_scout_", "_"),
                    side=side,
                    **kwargs,
                )
                stake = full_stake * self.earlytrend_short_stake_fraction
                if min_stake is not None and stake < min_stake:
                    return min_stake if min_stake <= max_stake else 0.0
                return min(stake, max_stake)
            if min_stake is None or min_stake > max_stake:
                return 0.0
            return min_stake
        return super().custom_stake_amount(
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


class HighVolumeMainWaveEarlyTrendScoutLooseV13(
    HighVolumeMainWaveEarlyTrendScoutV13
):
    """Nearby persistence check requiring three of six volume hours."""

    earlytrend_persistent_hours = 3


class HighVolumeMainWaveEarlyTrendScoutTightV13(
    HighVolumeMainWaveEarlyTrendScoutV13
):
    """Nearby persistence check requiring five of six volume hours."""

    earlytrend_persistent_hours = 5


class HighVolumeMainWaveEarlyTrendScoutLongOnlyV13(
    HighVolumeMainWaveEarlyTrendScoutV13
):
    """Factor-isolation check retaining only early-trend long scouts."""

    earlytrend_short_enabled = False


class HighVolumeMainWaveEarlyTrendScoutShortOnlyV13(
    HighVolumeMainWaveEarlyTrendScoutV13
):
    """Selected V13 candidate: add only early-trend short scouts.

    Ordinary and persistent V11 long entries remain fully enabled.  Only the
    additional pre-EMA-cross scout layer is short-only because its long side
    reduced portfolio quality through slot displacement in factor tests.
    """

    earlytrend_long_enabled = False


class HighVolumeMainWaveEarlyTrendScoutShortLooseV13(
    HighVolumeMainWaveEarlyTrendScoutShortOnlyV13
):
    """Short-only robustness check requiring three of six volume hours."""

    earlytrend_persistent_hours = 3


class HighVolumeMainWaveEarlyTrendScoutShortTightV13(
    HighVolumeMainWaveEarlyTrendScoutShortOnlyV13
):
    """Short-only robustness check requiring five of six volume hours."""

    earlytrend_persistent_hours = 5


class HighVolumeMainWaveAdaptiveRiskV16(
    HighVolumeMainWaveEarlyTrendScoutShortOnlyV13
):
    """Reduce late-candle long risk and reject capitulation short scouts.

    V13's main-wave and early-trend coverage is retained.  Ordinary main-wave
    longs that close too far beyond the completed 1h Donchian boundary remain
    eligible, but receive half of the ordinary risk-sized stake because the
    entry is more likely to be chasing the end of the breakout candle.

    Thin-volume scout shorts already trade at Binance's minimum stake.  The
    subset whose completed 1h candle both closes in its bottom ten percent and
    spans at least 1.5 ATR is suppressed: this combination behaved like a
    liquidation-exhaustion candle rather than a durable downside break.
    """

    extended_long_minimum_atr = 0.38
    extended_long_stake_fraction = 0.50
    filter_thin_capitulation = True
    capitulation_maximum_close_location = 0.10
    capitulation_minimum_body_atr = 1.50

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]

        extended_long = (
            dataframe["enter_long"].eq(1)
            & dataframe["enter_tag"].eq(f"{signal_tf}_mainwave_breakout_long")
            & (
                dataframe[f"breakout_extension_atr_{signal_tf}"]
                >= self.extended_long_minimum_atr
            )
        ).fillna(False)
        dataframe.loc[extended_long, "enter_tag"] = (
            f"{signal_tf}_extended_mainwave_breakout_long"
        )

        if self.filter_thin_capitulation:
            thin_capitulation = (
                dataframe["enter_short"].eq(1)
                & dataframe["enter_tag"].eq(
                    f"{signal_tf}_thinvolume_scout_breakout_short"
                )
                & (
                    dataframe[f"close_location_{signal_tf}"]
                    <= self.capitulation_maximum_close_location
                )
                & (
                    dataframe[f"body_atr14_{signal_tf}"]
                    >= self.capitulation_minimum_body_atr
                )
            ).fillna(False)
            dataframe.loc[thin_capitulation, "enter_short"] = 0
            dataframe.loc[thin_capitulation, "enter_tag"] = None
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        stake = super().custom_stake_amount(
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
        if entry_tag and "_extended_mainwave_breakout_long" in entry_tag:
            stake *= self.extended_long_stake_fraction
            if min_stake is not None and stake < min_stake:
                return min_stake if min_stake <= max_stake else 0.0
        return min(stake, max_stake)


class HighVolumeMainWaveExtendedLongRiskOnlyV16(
    HighVolumeMainWaveAdaptiveRiskV16
):
    """Factor-isolation check retaining only the extended-long risk tier."""

    filter_thin_capitulation = False


class HighVolumeMainWaveExtendedLongRiskLooseV16(
    HighVolumeMainWaveExtendedLongRiskOnlyV16
):
    """Nearby extended-long check starting at 0.35 ATR."""

    extended_long_minimum_atr = 0.35


class HighVolumeMainWaveExtendedLongRiskTightV16(
    HighVolumeMainWaveExtendedLongRiskOnlyV16
):
    """Nearby extended-long check starting at 0.40 ATR."""

    extended_long_minimum_atr = 0.40


class HighVolumeMainWaveExtendedLongRiskQuarterV16(
    HighVolumeMainWaveExtendedLongRiskOnlyV16
):
    """Sizing sensitivity check using one quarter of ordinary long risk."""

    extended_long_stake_fraction = 0.25


class HighVolumeMainWaveExtendedLongRiskThreeQuarterV16(
    HighVolumeMainWaveExtendedLongRiskOnlyV16
):
    """Sizing sensitivity check using three quarters of ordinary long risk."""

    extended_long_stake_fraction = 0.75


class HighVolumeMainWaveV16(HighVolumeMainWaveExtendedLongRiskOnlyV16):
    """Selected V16 candidate with coverage-preserving long risk ranking."""


class HighVolumeMainWaveMinScoutFastFailV17(HighVolumeMainWaveV16):
    """Release only the weakest minimum-stake scouts after six hours.

    V16's profitable normal scout shorts and early-trend scout shorts retain
    their nine-hour confirmation window.  Minimum-stake scout longs and thin-
    volume scout shorts can exit three hours earlier when they are still
    losing and have never travelled one current ATR in their favor.  Main-wave
    entries, sizing, and exits are unchanged.
    """

    minimum_scout_confirmation_hours = 6
    minimum_scout_fastfail_favorable_atr = 1.0
    fast_fail_minimum_scout_longs = True
    fast_fail_earlytrend_scout_longs = False
    fast_fail_pretrend_scout_shorts = False
    fast_fail_thinvolume_scout_shorts = True

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        tag = trade.enter_tag or ""
        selected = (
            self.fast_fail_minimum_scout_longs
            and tag == "1h_scout_breakout_long"
        ) or (
            self.fast_fail_earlytrend_scout_longs
            and tag == "1h_earlytrend_scout_breakout_long"
        ) or (
            self.fast_fail_pretrend_scout_shorts
            and tag == "1h_pretrend_scout_breakout_short"
        ) or (
            self.fast_fail_thinvolume_scout_shorts
            and tag == "1h_thinvolume_scout_breakout_short"
        )
        if (
            selected
            and current_time
            >= trade.open_date_utc
            + timedelta(hours=self.minimum_scout_confirmation_hours)
            and current_profit <= 0
        ):
            atr = self._latest_atr(pair)
            if atr is not None:
                if trade.is_short:
                    best_rate = min(trade.open_rate, current_rate)
                    if trade.min_rate is not None:
                        best_rate = min(best_rate, trade.min_rate)
                    favorable_excursion = trade.open_rate - best_rate
                else:
                    best_rate = max(trade.open_rate, current_rate)
                    if trade.max_rate is not None:
                        best_rate = max(best_rate, trade.max_rate)
                    favorable_excursion = best_rate - trade.open_rate
                if (
                    favorable_excursion
                    < self.minimum_scout_fastfail_favorable_atr * atr
                ):
                    return "minimum_scout_failed_followthrough"

        return super().custom_exit(
            pair=pair,
            trade=trade,
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            **kwargs,
        )


class HighVolumeMainWaveMinScoutLongFastFailOnlyV17(
    HighVolumeMainWaveMinScoutFastFailV17
):
    """Factor-isolation check applying the six-hour rule only to scout longs."""

    fast_fail_thinvolume_scout_shorts = False


class HighVolumeMainWaveThinScoutShortFastFailOnlyV17(
    HighVolumeMainWaveMinScoutFastFailV17
):
    """Factor-isolation check applying the six-hour rule only to thin shorts."""

    fast_fail_minimum_scout_longs = False


class HighVolumeMainWaveMinScoutLongFastFailEarlyV17(
    HighVolumeMainWaveMinScoutLongFastFailOnlyV17
):
    """Nearby scout-long timing check using a 4.5-hour window."""

    minimum_scout_confirmation_hours = 4.5


class HighVolumeMainWaveMinScoutLongFastFailLateV17(
    HighVolumeMainWaveMinScoutLongFastFailOnlyV17
):
    """Nearby scout-long timing check using a 7.5-hour window."""

    minimum_scout_confirmation_hours = 7.5


class HighVolumeMainWaveMinScoutLongFastFailNarrowV17(
    HighVolumeMainWaveMinScoutLongFastFailOnlyV17
):
    """Require less favorable excursion before the six-hour failure exit."""

    minimum_scout_fastfail_favorable_atr = 0.75


class HighVolumeMainWaveMinScoutLongFastFailWideV17(
    HighVolumeMainWaveMinScoutLongFastFailOnlyV17
):
    """Require more favorable excursion before the six-hour failure exit."""

    minimum_scout_fastfail_favorable_atr = 1.25


class HighVolumeMainWaveV17(HighVolumeMainWaveMinScoutLongFastFailOnlyV17):
    """Selected V17: let weak long scouts yield to the real main wave."""


class HighVolumeMainWaveStrongShortRiskV18(HighVolumeMainWaveV17):
    """Increase risk only on consistently high-volume main-wave shorts.

    Main-wave short entries whose completed 1h relative volume is at least
    1.5 receive a 1.5x stake multiplier, subject to the inherited forty-
    percent equity notional cap.  Lower-volume short breakouts remain fully
    tradable at ordinary risk; signal coverage, exits, and slot usage do not
    change.
    """

    strong_short_minimum_rvol = 1.5
    strong_short_stake_multiplier = 1.5

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]
        strong_short = (
            dataframe["enter_short"].eq(1)
            & dataframe["enter_tag"].eq(f"{signal_tf}_mainwave_breakout_short")
            & (
                dataframe[f"rvol20_{signal_tf}"]
                >= self.strong_short_minimum_rvol
            )
        ).fillna(False)
        dataframe.loc[strong_short, "enter_tag"] = (
            f"{signal_tf}_highvolume_mainwave_breakout_short"
        )
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        stake = super().custom_stake_amount(
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
        if entry_tag and "_highvolume_mainwave_breakout_short" in entry_tag:
            total_equity = float(self.wallets.get_total_stake_amount())
            notional_cap_stake = (
                total_equity
                * self.maximum_notional_fraction
                / max(leverage, 1.0)
            )
            stake = min(
                stake * self.strong_short_stake_multiplier,
                notional_cap_stake,
                max_stake,
            )
        return stake


class HighVolumeMainWaveStrongShortRiskConservativeV18(
    HighVolumeMainWaveStrongShortRiskV18
):
    """Sizing sensitivity check using a 1.25x strong-short multiplier."""

    strong_short_stake_multiplier = 1.25


class HighVolumeMainWaveStrongShortRiskExtendedV18(
    HighVolumeMainWaveStrongShortRiskV18
):
    """Sizing sensitivity check using a 1.75x strong-short multiplier."""

    strong_short_stake_multiplier = 1.75


class HighVolumeMainWaveStrongShortRiskDoubleV18(
    HighVolumeMainWaveStrongShortRiskV18
):
    """Sizing sensitivity check using a 2.00x strong-short multiplier."""

    strong_short_stake_multiplier = 2.00


class HighVolumeMainWaveStrongShortRiskLooseV18(
    HighVolumeMainWaveStrongShortRiskV18
):
    """Relative-volume boundary check starting at 1.25."""

    strong_short_minimum_rvol = 1.25


class HighVolumeMainWaveStrongShortRiskTightV18(
    HighVolumeMainWaveStrongShortRiskV18
):
    """Relative-volume boundary check starting at 1.75."""

    strong_short_minimum_rvol = 1.75


class HighVolumeMainWaveStrongShortRiskVeryTightV18(
    HighVolumeMainWaveStrongShortRiskV18
):
    """Relative-volume boundary check starting at 2.00."""

    strong_short_minimum_rvol = 2.00


class HighVolumeMainWaveV18(HighVolumeMainWaveStrongShortRiskV18):
    """Selected V18: 1.5x risk for the high-volume main-wave short tier."""


class HighVolumeMainWaveSelectiveEarlyLongV19(HighVolumeMainWaveV18):
    """Probe only gently rising, volatile early-trend long breakouts.

    The broad V13 early-long layer displaced stronger trades.  V19 re-enables
    it only while 4h EMA20 has started rising but its three-bar slope is no
    more than one percent, and 1h ATR is at least four percent of price.  This
    targets the start of a volatile reversal rather than an already-accelerated
    candle.  Qualifying entries retain the exchange-minimum scout stake.
    """

    earlytrend_long_enabled = True
    selective_early_long_maximum_slope = 0.01
    selective_early_long_minimum_atr_percent = 0.04

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]
        early_long = (
            dataframe["enter_long"].eq(1)
            & dataframe["enter_tag"].eq(
                f"{signal_tf}_earlytrend_scout_breakout_long"
            )
        )
        qualified = (
            (
                dataframe["ema20_slope3_4h"]
                <= self.selective_early_long_maximum_slope
            )
            & (
                dataframe[f"atr_percent_{signal_tf}"]
                >= self.selective_early_long_minimum_atr_percent
            )
        ).fillna(False)
        rejected = early_long & ~qualified
        dataframe.loc[rejected, "enter_long"] = 0
        dataframe.loc[rejected, "enter_tag"] = None
        return dataframe


class HighVolumeMainWaveSelectiveEarlyLongSlopeTightV19(
    HighVolumeMainWaveSelectiveEarlyLongV19
):
    """Nearby early-long check capping the 4h slope at 0.75%."""

    selective_early_long_maximum_slope = 0.0075


class HighVolumeMainWaveSelectiveEarlyLongSlopeLooseV19(
    HighVolumeMainWaveSelectiveEarlyLongV19
):
    """Nearby early-long check capping the 4h slope at 1.25%."""

    selective_early_long_maximum_slope = 0.0125


class HighVolumeMainWaveSelectiveEarlyLongVolatilityLooseV19(
    HighVolumeMainWaveSelectiveEarlyLongV19
):
    """Nearby early-long check requiring at least 3.5% 1h ATR."""

    selective_early_long_minimum_atr_percent = 0.035


class HighVolumeMainWaveSelectiveEarlyLongVolatilityTightV19(
    HighVolumeMainWaveSelectiveEarlyLongV19
):
    """Nearby early-long check requiring at least 4.5% 1h ATR."""

    selective_early_long_minimum_atr_percent = 0.045


class HighVolumeMainWaveSelectiveEarlyLongFastFailV19(
    HighVolumeMainWaveSelectiveEarlyLongV19
):
    """Apply V17's six-hour failure check to the selective early longs."""

    fast_fail_earlytrend_scout_longs = True


class HighVolumeMainWavePreTrendShortV20(HighVolumeMainWaveV18):
    """Use a minimum-stake 1h downside probe before any 4h confirmation.

    The probe requires a completed 1h Donchian-20 breakdown, relative volume
    of at least 1.25, and a volatile three-percent ATR regime.  It is added
    only when V18 has no ordinary or early-trend short on the candle.  No 4h
    EMA condition is required because the purpose is to observe the reversal
    before that deliberately slow filter confirms.  Existing nine-hour scout
    failure handling applies and stake is fixed at the exchange minimum.
    """

    pretrend_short_donchian_hours = 20
    pretrend_short_minimum_rvol = 1.25
    pretrend_short_minimum_atr_percent = 0.03

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]
        pair = metadata["pair"]

        active = pd.Series(False, index=dataframe.index)
        for start, end, pool in self.monthly_entry_pools:
            if pair in pool:
                active |= (dataframe["date"] >= pd.Timestamp(start, tz="UTC")) & (
                    dataframe["date"] < pd.Timestamp(end, tz="UTC")
                )

        pretrend_short = (
            active
            & dataframe[f"new_{signal_tf}"]
            & dataframe["enter_short"].eq(0)
            & (
                dataframe[f"close_{signal_tf}"]
                < dataframe[
                    f"donchian_low{self.pretrend_short_donchian_hours}_{signal_tf}"
                ]
            )
            & (
                dataframe[f"rvol20_{signal_tf}"]
                >= self.pretrend_short_minimum_rvol
            )
            & dataframe[f"candle_down_{signal_tf}"]
            & (
                dataframe[f"atr_percent_{signal_tf}"]
                >= self.pretrend_short_minimum_atr_percent
            )
            & (dataframe["volume"] > 0)
        ).fillna(False)
        dataframe.loc[pretrend_short, ["enter_short", "enter_tag"]] = (
            1,
            f"{signal_tf}_pretrend_scout_breakout_short",
        )
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        if entry_tag and "_pretrend_scout_" in entry_tag:
            if min_stake is None or min_stake > max_stake:
                return 0.0
            return min_stake
        return super().custom_stake_amount(
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


class HighVolumeMainWavePreTrendShortRvolLooseV20(
    HighVolumeMainWavePreTrendShortV20
):
    """Nearby pre-trend probe accepting relative volume from 1.0."""

    pretrend_short_minimum_rvol = 1.0


class HighVolumeMainWavePreTrendShortRvolTightV20(
    HighVolumeMainWavePreTrendShortV20
):
    """Nearby pre-trend probe requiring relative volume of at least 1.5."""

    pretrend_short_minimum_rvol = 1.5


class HighVolumeMainWavePreTrendShortDonchian10V20(
    HighVolumeMainWavePreTrendShortV20
):
    """Earlier boundary check using the prior ten-hour low."""

    pretrend_short_donchian_hours = 10


class HighVolumeMainWavePreTrendShortRvolTightFastFailV20(
    HighVolumeMainWavePreTrendShortRvolTightV20
):
    """Give the 1.5-rvol pre-trend probes six hours to prove follow-through."""

    fast_fail_pretrend_scout_shorts = True


class HighVolumeMainWaveWeeklyRotationJulyV20(
    HighVolumeMainWavePreTrendShortRvolTightFastFailV20
):
    """July 2026 V20 experiment with seven-day entry-pool refreshes.

    Only permission to open a new position changes at a weekly boundary.
    Positions carried from a removed pair retain every inherited V20 exit.
    """

    monthly_entry_pools = (
        (
            "2026-07-01", "2026-07-08",
            frozenset({
                "IN/USDT:USDT", "SYN/USDT:USDT", "LAB/USDT:USDT",
                "AIGENSYN/USDT:USDT", "TAC/USDT:USDT", "RE/USDT:USDT",
                "WLD/USDT:USDT", "SLX/USDT:USDT", "UB/USDT:USDT",
                "VELVET/USDT:USDT", "1000PEPE/USDT:USDT",
            }),
        ),
        (
            "2026-07-08", "2026-07-15",
            frozenset({
                "LAB/USDT:USDT", "EVAA/USDT:USDT", "VANRY/USDT:USDT",
                "TAC/USDT:USDT", "BLUR/USDT:USDT", "WLD/USDT:USDT",
                "EPIC/USDT:USDT", "NEAR/USDT:USDT", "YFI/USDT:USDT",
                "EDGE/USDT:USDT", "1000PEPE/USDT:USDT",
            }),
        ),
        (
            "2026-07-15", "2026-07-22",
            frozenset({
                "LAB/USDT:USDT", "EVAA/USDT:USDT", "BILL/USDT:USDT",
                "SXT/USDT:USDT", "WLD/USDT:USDT", "1000PEPE/USDT:USDT",
                "VELVET/USDT:USDT", "BSB/USDT:USDT", "TRIA/USDT:USDT",
            }),
        ),
        (
            "2026-07-22", "2026-07-29",
            frozenset({
                "BANK/USDT:USDT", "DEXE/USDT:USDT", "ERA/USDT:USDT",
                "AKE/USDT:USDT", "ONDO/USDT:USDT", "NEAR/USDT:USDT",
                "ESPORTS/USDT:USDT", "1000PEPE/USDT:USDT",
                "ONE/USDT:USDT", "LA/USDT:USDT",
            }),
        ),
        (
            "2026-07-29", "2026-08-01",
            frozenset({
                "BANK/USDT:USDT", "COTI/USDT:USDT", "ON/USDT:USDT",
                "AKE/USDT:USDT", "DEXE/USDT:USDT", "1000PEPE/USDT:USDT",
                "KAITO/USDT:USDT", "PUMP/USDT:USDT", "BEAT/USDT:USDT",
                "SOON/USDT:USDT",
            }),
        ),
    )


def _load_weekly_rotation_pools(
    filename: str = "weekly-volume-rotation-2026-02-08.json",
) -> tuple[tuple[str, str, frozenset[str]], ...]:
    """Load the point-in-time weekly screen without duplicating 31 pool literals."""
    selection_path = (
        Path(__file__).resolve().parents[1]
        / "analysis"
        / filename
    )
    selection = json.loads(selection_path.read_text())
    return tuple(
        (
            pd.Timestamp(row["start"]).strftime("%Y-%m-%d %H:%M:%S"),
            pd.Timestamp(row["end_exclusive"]).strftime("%Y-%m-%d %H:%M:%S"),
            frozenset(
                f"{symbol.removesuffix('USDT')}/USDT:USDT"
                for symbol in row["selected_symbols"]
            ),
        )
        for row in selection["periods"]
    )


def _load_weekly_rotation_volume_ranks(
    filename: str = "weekly-volume-rotation-2026-02-08.json",
) -> tuple[tuple[pd.Timestamp, pd.Timestamp, dict[str, int]], ...]:
    """Load the frozen raw-volume rank attached to each weekly selection."""
    selection_path = Path(__file__).resolve().parents[1] / "analysis" / filename
    selection = json.loads(selection_path.read_text())
    return tuple(
        (
            pd.Timestamp(row["start"]),
            pd.Timestamp(row["end_exclusive"]),
            {
                f"{candidate['raw_symbol'].removesuffix('USDT')}/USDT:USDT": int(
                    candidate["volume_rank"]
                )
                for candidate in row["top_volume"]
            },
        )
        for row in selection["periods"]
    )


def _load_incremental_rotation_roles(
    filename: str,
) -> tuple[
    tuple[pd.Timestamp, pd.Timestamp, frozenset[str], frozenset[str]], ...
]:
    """Load core/incremental membership at each point-in-time pool segment."""
    selection_path = Path(__file__).resolve().parents[1] / "analysis" / filename
    selection = json.loads(selection_path.read_text())
    roles = []
    for row in selection["periods"]:
        if "core_selected_symbols" not in row or "incremental_selected_symbols" not in row:
            raise ValueError(f"Incremental role metadata is absent from {selection_path}")
        roles.append(
            (
                pd.Timestamp(row["start"]),
                pd.Timestamp(row["end_exclusive"]),
                frozenset(
                    f"{symbol.removesuffix('USDT')}/USDT:USDT"
                    for symbol in row["core_selected_symbols"]
                ),
                frozenset(
                    f"{symbol.removesuffix('USDT')}/USDT:USDT"
                    for symbol in row["incremental_selected_symbols"]
                ),
            )
        )
    return tuple(roles)


def _load_monthly_rotation_pools(
    filename: str,
) -> tuple[tuple[str, str, frozenset[str]], ...]:
    """Load a monthly point-in-time screen for the low-memory backtester."""
    selection_path = (
        Path(__file__).resolve().parents[1]
        / "analysis"
        / filename
    )
    selection = json.loads(selection_path.read_text())
    pools = []
    for month, row in selection["months"].items():
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        pools.append(
            (
                start.strftime("%Y-%m-%d %H:%M:%S"),
                end.strftime("%Y-%m-%d %H:%M:%S"),
                frozenset(
                    f"{symbol.removesuffix('USDT')}/USDT:USDT"
                    for symbol in row["selected_symbols"]
                ),
            )
        )
    return tuple(pools)


class HighVolumeMainWaveWeeklyRotationFullV20(
    HighVolumeMainWavePreTrendShortRvolTightFastFailV20
):
    """February-August V20 with weekly entry-pool refreshes.

    The sliced research datadir can contain disconnected per-pair windows.
    Informative indicators therefore restart after a real data gap, using the
    warmup stored ahead of each active window.  The two memory hooks release
    source/informative frames as each pair is analyzed and retain only fields
    required by V20's callbacks once signals have been calculated.

    Run this class by itself: destructive source-frame release intentionally
    optimizes the 142-pair, low-memory research backtest rather than a
    multi-strategy invocation.
    """

    monthly_entry_pools = _load_weekly_rotation_pools()
    # V20 callbacks read only the current ATR.  Backtesting can therefore
    # retain a two-column callback cache instead of another full OHLC/signal
    # frame for every pair.
    backtest_callback_cache_columns = ("date", "atr14_1h")

    def _export_training_signal_features(
        self, dataframe: DataFrame, metadata: dict
    ) -> None:
        """Persist pre-entry features for an explicitly enabled research run.

        Export happens before the low-memory dataframe compaction and includes
        every eligible strategy signal, including signals later rejected by a
        full portfolio.  Future outcomes are deliberately not calculated here.
        """
        output_value = self.config.get("v20_signal_feature_export_dir")
        if not output_value:
            return
        signal_mask = dataframe["enter_long"].eq(1) | dataframe["enter_short"].eq(1)
        if not signal_mask.any():
            return
        frame = dataframe.loc[
            signal_mask,
            [
                "date", "open", "high", "low", "close", "enter_long",
                "enter_short", "enter_tag", "close_1h", "atr14_1h",
                "rvol20_1h", "rvol_count6_1h", "close_location_1h",
                "body_atr14_1h", "atr_percent_1h",
                "breakout_extension_atr_1h", "breakdown_extension_atr_1h",
                "close_4h", "ema20_4h", "ema50_4h", "ema20_slope3_4h",
                "ema_separation_atr_4h",
            ],
        ].copy()
        frame["pair"] = metadata["pair"]
        frame["side"] = np.where(frame["enter_short"].eq(1), "short", "long")
        direction = np.where(frame["side"].eq("long"), 1.0, -1.0)
        frame["directional_close_location"] = np.where(
            frame["side"].eq("long"),
            frame["close_location_1h"],
            1 - frame["close_location_1h"],
        )
        frame["directional_extension_atr"] = np.where(
            frame["side"].eq("long"),
            frame["breakout_extension_atr_1h"],
            frame["breakdown_extension_atr_1h"],
        )
        frame["directional_ema_separation_atr_4h"] = (
            frame["ema_separation_atr_4h"] * direction
        )
        frame["directional_ema20_slope3_4h"] = (
            frame["ema20_slope3_4h"] * direction
        )
        frame["directional_price_vs_ema20_4h"] = (
            (frame["close_4h"] / frame["ema20_4h"] - 1) * direction
        )
        frame["directional_price_vs_ema50_4h"] = (
            (frame["close_4h"] / frame["ema50_4h"] - 1) * direction
        )
        frame["execution_date"] = frame["date"] + pd.Timedelta(minutes=30)
        output = Path(output_value)
        if not output.is_absolute():
            output = Path.cwd() / output
        output.mkdir(parents=True, exist_ok=True)
        safe_pair = metadata["pair"].replace("/", "_").replace(":", "_")
        frame.reset_index(drop=True).to_feather(output / f"{safe_pair}.feather")

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        """Reject a previous-candle signal if its pair left at this boundary."""
        for start, end, pool in self.monthly_entry_pools:
            if pair in pool and (
                pd.Timestamp(start, tz="UTC")
                <= current_time
                < pd.Timestamp(end, tz="UTC")
            ):
                return True
        return False

    def _segmented_informative_indicators(
        self, dataframe: DataFrame, informative_timeframe: str
    ) -> DataFrame:
        if dataframe.empty:
            return self._informative_indicators(dataframe)
        expected = pd.Timedelta(informative_timeframe)
        segment = dataframe["date"].diff().gt(expected).cumsum()
        if not segment.any():
            return self._informative_indicators(dataframe)
        return pd.concat(
            [
                self._informative_indicators(part.copy())
                for _, part in dataframe.groupby(segment, sort=False)
            ]
        ).sort_index()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            return dataframe
        for informative_timeframe in self.signal_timeframes:
            informative = self.dp.get_pair_dataframe(
                pair=metadata["pair"], timeframe=informative_timeframe
            )
            informative = self._segmented_informative_indicators(
                informative, informative_timeframe
            )
            informative = informative.loc[
                :, self.informative_columns[informative_timeframe]
            ]
            dataframe = merge_informative_pair(
                dataframe,
                informative,
                self.timeframe,
                informative_timeframe,
                ffill=True,
            )
            date_column = f"date_{informative_timeframe}"
            dataframe[f"new_{informative_timeframe}"] = (
                dataframe[date_column].notna()
                & dataframe[date_column].ne(dataframe[date_column].shift(1))
            )
        return dataframe

    def advise_all_indicators(
        self, data: dict[str, DataFrame]
    ) -> dict[str, DataFrame]:
        processed: dict[str, DataFrame] = {}
        for pair in list(data):
            pair_data = data[pair]
            indicators = super().advise_all_indicators({pair: pair_data})[pair]
            # Signals are pair-local and can be calculated immediately.  Keep
            # only callback/loop fields instead of accumulating every merged
            # informative column for all 142 pairs.
            processed[pair] = self.ft_advise_signals(indicators, {"pair": pair})
            # Backtesting still consumes date/close after the strategy run to
            # calculate its equal-weight market benchmark.
            data[pair] = pair_data.loc[:, ["date", "close"]].copy()
            if self.dp:
                cache = getattr(
                    self.dp, "_DataProvider__cached_pairs_backtesting", None
                )
                if cache is not None:
                    for key in [key for key in cache if key[0] == pair]:
                        cache.pop(key, None)
        return processed

    def ft_advise_signals(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The low-memory advise_all_indicators hook has already populated
        # signals before Backtesting reaches its normal signal pass.
        if "donchian_high20_1h" not in dataframe.columns:
            return dataframe
        dataframe = super().ft_advise_signals(dataframe, metadata)
        self._export_training_signal_features(dataframe, metadata)
        for signal_column in (
            "enter_long", "exit_long", "enter_short", "exit_short"
        ):
            dataframe[signal_column] = (
                dataframe[signal_column].fillna(0).astype("int8")
            )
        callback_columns = [
            "date", "open", "high", "low", "close",
            "enter_long", "exit_long", "enter_short", "exit_short",
            "enter_tag", "exit_tag", "atr14_1h",
        ]
        return dataframe.loc[:, callback_columns].copy()


class HighVolumeMainWaveWeeklyRetentionOneMonthV20(
    HighVolumeMainWaveWeeklyRotationFullV20
):
    """Keep every weekly selection eligible for one rolling calendar month."""

    monthly_entry_pools = _load_weekly_rotation_pools(
        "weekly-volume-rotation-retention-1m-2026-02-08.json"
    )


class _HighVolumeMainWaveRefitParameters:
    """Apply a small, explicit research grid without changing V20's logic.

    These classes are used only by the chronological 2026 refit.  Keeping the
    values in the config makes every candidate artifact self-describing and
    lets the monthly and weekly pool mechanisms use the same search space.
    """

    refit_parameter_names = frozenset(
        {
            "pretrend_short_minimum_rvol",
            "persistent_mainwave_long_multiplier",
            "strong_short_stake_multiplier",
        }
    )

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        parameters = config.get("v20_refit_parameters", {})
        unknown = set(parameters).difference(self.refit_parameter_names)
        if unknown:
            raise ValueError(f"Unsupported V20 refit parameters: {sorted(unknown)}")
        for name, value in parameters.items():
            setattr(self, name, float(value))


class HighVolumeMainWaveMonthlyRefitV20(
    _HighVolumeMainWaveRefitParameters,
    HighVolumeMainWavePreTrendShortRvolTightFastFailV20,
):
    """Configurable V20 used for the monthly-pool chronological refit."""


class HighVolumeMainWaveWeeklyRefitV20(
    _HighVolumeMainWaveRefitParameters,
    HighVolumeMainWaveWeeklyRotationFullV20,
):
    """Configurable low-memory V20 used for the weekly-pool refit."""


class HighVolumeMainWaveMonthlyExpandedRefitV20(
    _HighVolumeMainWaveRefitParameters,
    HighVolumeMainWaveWeeklyRotationFullV20,
):
    """Monthly top-40-volume pool with V20's frozen monthly parameters."""

    monthly_entry_pools = _load_monthly_rotation_pools(
        "monthly-volume-rotation-top40-2026-02-08.json"
    )


class HighVolumeMainWaveMonthlyLowMemoryRefitV20(
    _HighVolumeMainWaveRefitParameters,
    HighVolumeMainWaveWeeklyRotationFullV20,
):
    """Original monthly top-20 pool through the same low-memory code path."""

    monthly_entry_pools = _load_monthly_rotation_pools(
        "monthly-volume-rotation-2026-02-08.json"
    )


class HighVolumeMainWaveMonthlyOffsetRefitV20(
    _HighVolumeMainWaveRefitParameters,
    HighVolumeMainWaveWeeklyRotationFullV20,
):
    """Low-memory monthly V20 with a config-selected refresh-day artifact."""

    def __init__(self, config: dict[str, Any]) -> None:
        selection = config.get("v20_monthly_offset_selection")
        if not isinstance(selection, str) or not selection:
            raise ValueError("v20_monthly_offset_selection must name an artifact")
        relative = Path(selection)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("v20_monthly_offset_selection must stay under analysis")
        self.monthly_entry_pools = _load_weekly_rotation_pools(selection)
        position_limit = config.get("v20_incremental_position_limit")
        if position_limit is not None:
            if (
                isinstance(position_limit, bool)
                or not isinstance(position_limit, int)
                or position_limit <= 0
            ):
                raise ValueError("v20_incremental_position_limit must be a positive integer")
            self.incremental_position_limit: int | None = position_limit
            self.incremental_role_periods = _load_incremental_rotation_roles(selection)
        else:
            self.incremental_position_limit = None
            self.incremental_role_periods = ()
        super().__init__(config)

    def _is_incremental_at(self, pair: str, moment: datetime) -> bool:
        timestamp = pd.Timestamp(moment)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return any(
            start <= timestamp < end and pair in incremental
            for start, end, _core, incremental in self.incremental_role_periods
        )

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
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
        if self.incremental_position_limit is None or not self._is_incremental_at(
            pair, current_time
        ):
            return True
        incremental_open = sum(
            self._is_incremental_at(trade.pair, trade.open_date_utc)
            for trade in Trade.get_open_trades()
        )
        return incremental_open < self.incremental_position_limit


class HighVolumeMainWaveMonthlyOffsetRiskDoubleV23(
    HighVolumeMainWaveMonthlyOffsetRefitV20
):
    """Double every position's notional risk for the shared four-slot test.

    Two-times leverage while evaluating every inherited signal tier at its
    one-times baseline stake preserves the original margin requirement and
    doubles notional exposure, including fixed minimum-stake scouts.  The hard
    strategy stop is also scaled so its maximum price distance is unchanged.
    """

    position_risk_multiplier = 2.0
    stoploss = -0.70
    portfolio_position_slots = 4

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(self.position_risk_multiplier, max_leverage)

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        stake = super().custom_stake_amount(
            pair=pair,
            current_time=current_time,
            current_rate=current_rate,
            proposed_stake=proposed_stake,
            min_stake=min_stake,
            max_stake=max_stake,
            # Compute the exact frozen 1x margin, then apply 2x leverage to it.
            # This avoids turning exchange-minimum scouts into 4x positions.
            leverage=1.0,
            entry_tag=entry_tag,
            side=side,
            **kwargs,
        )
        # Freqtrade divides the exchange minimum stake by leverage before the
        # callback.  Fixed-minimum scouts therefore need their margin restored
        # to the 1x minimum as well, otherwise their notional would not double.
        if min_stake is not None and 0 < stake <= min_stake * (1 + 1e-9):
            stake *= self.position_risk_multiplier
        return min(stake, max_stake)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
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
        return len(Trade.get_open_trades()) < self.portfolio_position_slots


class HighVolumeMainWaveMonthlyOffsetRiskBudget2PctV23(
    HighVolumeMainWaveMonthlyOffsetRefitV20
):
    """Use a two-percent base risk budget while retaining one-times leverage.

    The inherited forty-percent notional cap, fixed minimum-stake scouts and
    all V20 signal-tier multipliers remain unchanged.  A strategy-side four-
    position gate lets the research runner use a small provisional stake for
    Binance leverage-tier lookup without altering actual slot competition.
    """

    risk_per_trade = 0.02
    portfolio_position_slots = 4

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
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
        return len(Trade.get_open_trades()) < self.portfolio_position_slots


class HighVolumeMainWaveMonthlyReducedRefitV20(
    _HighVolumeMainWaveRefitParameters,
    HighVolumeMainWaveWeeklyRotationFullV20,
):
    """Monthly top-10-volume pool with V20's frozen monthly parameters."""

    monthly_entry_pools = _load_monthly_rotation_pools(
        "monthly-volume-rotation-top10-2026-02-08.json"
    )


class HighVolumeMainWaveMonthlyPostFilterTop10RefitV20(
    _HighVolumeMainWaveRefitParameters,
    HighVolumeMainWaveWeeklyRotationFullV20,
):
    """Monthly top 10 selected after market-cap and theme exclusions."""

    monthly_entry_pools = _load_monthly_rotation_pools(
        "monthly-volume-rotation-filtered-top10-2026-02-08.json"
    )


class HighVolumeMainWaveMonthlyPostFilterTop5RefitV20(
    _HighVolumeMainWaveRefitParameters,
    HighVolumeMainWaveWeeklyRotationFullV20,
):
    """Monthly top 5 selected after market-cap and theme exclusions."""

    monthly_entry_pools = _load_monthly_rotation_pools(
        "monthly-volume-rotation-filtered-top5-2026-02-08.json"
    )


class HighVolumeMainWaveMonthlyPostFilterTop15RefitV20(
    _HighVolumeMainWaveRefitParameters,
    HighVolumeMainWaveWeeklyRotationFullV20,
):
    """Monthly top 15 selected after market-cap and theme exclusions."""

    monthly_entry_pools = _load_monthly_rotation_pools(
        "monthly-volume-rotation-filtered-top15-2026-02-08.json"
    )


class HighVolumeMainWaveWeeklyFilteredV21(HighVolumeMainWaveWeeklyRefitV20):
    """Training-only experiment that filters weak V20 entries by 4h context.

    The filter uses only completed 4h indicators already present when V20
    emits the signal.  Configuration deliberately separates the target scope
    from thresholds so a small number of interpretable variants can be tested
    without duplicating strategy classes.
    """

    supported_filter_scopes = frozenset(
        {"all", "all_scouts", "weak_scouts", "long_thin_early"}
    )

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        parameters = config.get("v21_entry_filter", {})
        self.entry_filter_scope = str(parameters.get("scope", "weak_scouts"))
        if self.entry_filter_scope not in self.supported_filter_scopes:
            raise ValueError(
                f"Unsupported V21 entry-filter scope: {self.entry_filter_scope}"
            )
        self.entry_filter_minimum_directional_slope = float(
            parameters.get("minimum_directional_ema20_slope3_4h", 0.02)
        )
        maximum_distance = parameters.get(
            "maximum_directional_price_vs_ema20_4h"
        )
        self.entry_filter_maximum_directional_distance = (
            None if maximum_distance is None else float(maximum_distance)
        )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        long_entry = dataframe["enter_long"].eq(1)
        short_entry = dataframe["enter_short"].eq(1)
        entry = long_entry | short_entry
        tags = dataframe["enter_tag"].fillna("")
        scouts = tags.str.contains("_scout_", regex=False)
        pretrend = tags.str.contains("_pretrend_", regex=False)

        if self.entry_filter_scope == "all":
            target = entry
        elif self.entry_filter_scope == "all_scouts":
            target = entry & scouts
        elif self.entry_filter_scope == "weak_scouts":
            # Preserve V20's intentionally early pre-trend short probe.  It
            # cannot logically be required to have mature 4h confirmation.
            target = entry & scouts & ~pretrend
        else:
            target = entry & (
                (long_entry & scouts)
                | tags.str.contains("_thinvolume_scout_", regex=False)
                | tags.str.contains("_earlytrend_scout_", regex=False)
            )

        direction = pd.Series(
            np.where(long_entry, 1.0, -1.0), index=dataframe.index
        )
        directional_slope = dataframe["ema20_slope3_4h"] * direction
        directional_distance = (
            dataframe["close_4h"] / dataframe["ema20_4h"] - 1
        ) * direction
        quality = (
            directional_slope
            >= self.entry_filter_minimum_directional_slope
        )
        if self.entry_filter_maximum_directional_distance is not None:
            quality &= (
                directional_distance
                <= self.entry_filter_maximum_directional_distance
            )
        rejected = target & ~quality.fillna(False)
        dataframe.loc[rejected, ["enter_long", "enter_short"]] = 0
        dataframe.loc[rejected, "enter_tag"] = None
        return dataframe


class HighVolumeMainWaveWeeklyRankedV21(HighVolumeMainWaveWeeklyRefitV20):
    """Rank simultaneous weekly-pool entries when free slots are scarce.

    Unlike the hard-filter experiment, every V20 signal remains eligible when
    capacity is available.  Signal features are collected during preprocessing
    and ``confirm_trade_entry`` admits only the highest-scoring candidates at
    a timestamp when their count exceeds the currently free slots.
    """

    ranking_modes = frozenset(
        {
            "confidence", "trend", "balanced", "volume_growth",
            "volume_composite",
        }
    )
    slot_ranking_volume_rank_periods = _load_weekly_rotation_volume_ranks()

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        parameters = config.get("v21_slot_ranking", {})
        self.slot_ranking_mode = str(parameters.get("mode", "balanced"))
        if self.slot_ranking_mode not in self.ranking_modes:
            raise ValueError(
                f"Unsupported V21 slot-ranking mode: {self.slot_ranking_mode}"
            )
        self.slot_ranking_confidence_weight = float(
            parameters.get("confidence_weight", 2.0)
        )
        self.slot_ranking_reverse_pair_ties = bool(
            parameters.get("reverse_pair_ties", False)
        )
        self._slot_candidates: dict[int, list[dict[str, Any]]] = {}

    @staticmethod
    def _confidence_score(tags: pd.Series) -> pd.Series:
        tags = tags.fillna("")
        score = pd.Series(2.0, index=tags.index)
        score.loc[tags.str.contains("_thinvolume_scout_", regex=False)] = 1.0
        score.loc[tags.str.contains("_earlytrend_scout_", regex=False)] = 2.0
        score.loc[tags.str.contains("_pretrend_scout_", regex=False)] = 3.0
        score.loc[tags.str.contains("_mainwave_", regex=False)] = 4.0
        score.loc[
            tags.str.contains("_persistent_mainwave_", regex=False)
            | tags.str.contains("_highvolume_mainwave_", regex=False)
        ] = 5.0
        return score

    @staticmethod
    def _trend_score(frame: DataFrame, direction: np.ndarray) -> pd.Series:
        directional_slope = frame["ema20_slope3_4h"] * direction
        directional_separation = frame["ema_separation_atr_4h"] * direction
        directional_distance = (
            frame["close_4h"] / frame["ema20_4h"] - 1
        ) * direction
        # Fixed, dimensionless scaling from the discovery-period descriptive
        # statistics.  Clipping prevents a single extreme coin from dominating
        # every other entry candidate.
        return (
            (directional_slope / 0.02).clip(-3.0, 3.0)
            + 0.5 * directional_separation.clip(-3.0, 3.0)
            - 0.5 * (directional_distance / 0.20).clip(lower=0.0, upper=3.0)
            + 0.25 * (frame["rvol_count6_1h"] / 6.0).clip(0.0, 1.0)
        ).fillna(-10.0)

    def _selection_volume_rank(
        self, pair: str, execution_date: pd.Timestamp
    ) -> int | None:
        for start, end, ranks in self.slot_ranking_volume_rank_periods:
            if start <= execution_date < end:
                return ranks.get(pair)
        return None

    @staticmethod
    def _volume_composite_scores(
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Equal-weight cross-sectional percentiles for three volume fields."""
        growth = pd.Series(
            [candidate.get("volume_growth") for candidate in candidates],
            dtype=float,
        ).replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
        absolute = pd.Series(
            [candidate.get("absolute_quote_volume") for candidate in candidates],
            dtype=float,
        ).replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
        frozen_rank = pd.Series(
            [
                -float(candidate["selection_volume_rank"])
                if candidate.get("selection_volume_rank") is not None
                else -np.inf
                for candidate in candidates
            ],
            dtype=float,
        )
        scores = (
            growth.rank(method="average", pct=True)
            + absolute.rank(method="average", pct=True)
            + frozen_rank.rank(method="average", pct=True)
        ) / 3.0
        return [
            {**candidate, "score": float(score)}
            for candidate, score in zip(candidates, scores, strict=True)
        ]

    def _export_training_signal_features(
        self, dataframe: DataFrame, metadata: dict
    ) -> None:
        # This hook runs after entries have been populated but before the
        # weekly strategy compacts away informative fields.
        super()._export_training_signal_features(dataframe, metadata)
        signal_mask = dataframe["enter_long"].eq(1) | dataframe["enter_short"].eq(1)
        if not signal_mask.any():
            return
        frame = dataframe.loc[signal_mask].copy()
        execution_dates = frame["date"] + pd.Timedelta(minutes=30)
        sides = np.where(frame["enter_long"].eq(1), "long", "short")
        if self.slot_ranking_mode == "volume_composite":
            growth = frame["quote_volume_growth6_1h"].replace(
                [np.inf, -np.inf], np.nan
            )
            absolute = frame["quote_volume24_1h"].replace(
                [np.inf, -np.inf], np.nan
            )
            for execution_date, side, growth_value, absolute_value in zip(
                execution_dates, sides, growth, absolute, strict=True
            ):
                execution_timestamp = pd.Timestamp(execution_date)
                key = execution_timestamp.value
                self._slot_candidates.setdefault(key, []).append(
                    {
                        "pair": metadata["pair"],
                        "side": side,
                        "score": 0.0,
                        "volume_growth": float(growth_value),
                        "absolute_quote_volume": float(absolute_value),
                        "selection_volume_rank": self._selection_volume_rank(
                            metadata["pair"], execution_timestamp
                        ),
                    }
                )
            return

        if self.slot_ranking_mode == "volume_growth":
            # Pure acceleration rank: completed recent 6h quote volume versus
            # the preceding 6h.  No signal-confidence or trend term is mixed
            # into this score.  Missing history ranks last.
            score = frame["quote_volume_growth6_1h"].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(-1e12)
        else:
            direction = np.where(frame["enter_long"].eq(1), 1.0, -1.0)
            confidence = self._confidence_score(frame["enter_tag"])
            trend = self._trend_score(frame, direction)
            if self.slot_ranking_mode == "confidence":
                score = confidence
            elif self.slot_ranking_mode == "trend":
                score = trend
            else:
                score = self.slot_ranking_confidence_weight * confidence + trend

        for execution_date, side, value in zip(
            execution_dates, sides, score, strict=True
        ):
            key = pd.Timestamp(execution_date).value
            self._slot_candidates.setdefault(key, []).append(
                {
                    "pair": metadata["pair"],
                    "side": side,
                    "score": float(value),
                }
            )

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
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

        candidates = self._slot_candidates.get(pd.Timestamp(current_time).value, [])
        if len(candidates) <= 1:
            return True
        open_trades = Trade.get_open_trades()
        open_pairs = {trade.pair for trade in open_trades}
        maximum = int(self.config.get("max_open_trades", self.max_open_trades))
        free_slots = maximum - len(open_trades)
        if maximum <= 0 or free_slots >= len(candidates):
            return True
        if free_slots <= 0:
            return False
        ranked_candidates = (
            self._volume_composite_scores(candidates)
            if self.slot_ranking_mode == "volume_composite"
            else candidates
        )
        eligible = [
            candidate
            for candidate in ranked_candidates
            if candidate["pair"] not in open_pairs
        ]
        if self.slot_ranking_reverse_pair_ties:
            selected = sorted(
                eligible,
                key=lambda candidate: (candidate["score"], candidate["pair"]),
                reverse=True,
            )[:free_slots]
        else:
            selected = sorted(
                eligible,
                key=lambda candidate: (-candidate["score"], candidate["pair"]),
            )[:free_slots]
        return any(
            candidate["pair"] == pair and candidate["side"] == side
            for candidate in selected
        )


class HighVolumeMainWaveMonthlyIncrementalReservedRankedV22(
    HighVolumeMainWaveMonthlyOffsetRefitV20
):
    """Keep four monthly-core slots plus one ranked incremental slot.

    The fifth slot is exclusively reserved for the weekly incremental pool.
    Core and incremental positions are counted from their role at entry time,
    so a pool refresh never force-closes or reclassifies an existing trade.
    When several incremental signals compete at the same candle, the V21
    signal-confidence score selects the single eligible candidate.
    """

    core_position_slots = 4
    incremental_position_slots = 1

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        expected = self.core_position_slots + self.incremental_position_slots
        maximum = int(config.get("max_open_trades", expected))
        if maximum != expected:
            raise ValueError(
                f"V22 reserved-slot strategy requires max_open_trades={expected}"
            )
        selection = config["v20_monthly_offset_selection"]
        self.incremental_role_periods = _load_incremental_rotation_roles(selection)
        self._incremental_slot_candidates: dict[int, list[dict[str, Any]]] = {}

    def _export_training_signal_features(
        self, dataframe: DataFrame, metadata: dict
    ) -> None:
        # Collect candidates before the low-memory strategy compacts the frame.
        super()._export_training_signal_features(dataframe, metadata)
        signal_mask = dataframe["enter_long"].eq(1) | dataframe["enter_short"].eq(1)
        if not signal_mask.any():
            return
        frame = dataframe.loc[signal_mask]
        execution_dates = frame["date"] + pd.Timedelta(minutes=30)
        sides = np.where(frame["enter_long"].eq(1), "long", "short")
        confidence = HighVolumeMainWaveWeeklyRankedV21._confidence_score(
            frame["enter_tag"]
        )
        for execution_date, side, score in zip(
            execution_dates, sides, confidence, strict=True
        ):
            key = pd.Timestamp(execution_date).value
            self._incremental_slot_candidates.setdefault(key, []).append(
                {
                    "pair": metadata["pair"],
                    "side": side,
                    "score": float(score),
                }
            )

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
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

        open_trades = Trade.get_open_trades()
        incremental_open = sum(
            self._is_incremental_at(trade.pair, trade.open_date_utc)
            for trade in open_trades
        )
        is_incremental = self._is_incremental_at(pair, current_time)
        if not is_incremental:
            core_open = len(open_trades) - incremental_open
            return core_open < self.core_position_slots
        if incremental_open >= self.incremental_position_slots:
            return False

        open_pairs = {trade.pair for trade in open_trades}
        candidates = [
            candidate
            for candidate in self._incremental_slot_candidates.get(
                pd.Timestamp(current_time).value, []
            )
            if candidate["pair"] not in open_pairs
            and self._is_incremental_at(candidate["pair"], current_time)
        ]
        if not candidates:
            return True
        selected = min(
            candidates,
            key=lambda candidate: (-candidate["score"], candidate["pair"]),
        )
        return selected["pair"] == pair and selected["side"] == side


class HighVolumeMainWaveMonthlyRankedV21(HighVolumeMainWaveWeeklyRankedV21):
    """Confidence-ranked original monthly top-20 rotation pool."""

    monthly_entry_pools = _load_monthly_rotation_pools(
        "monthly-volume-rotation-2026-02-08.json"
    )


class HighVolumeMainWaveWeeklyTop10RankedV21(HighVolumeMainWaveWeeklyRankedV21):
    """Confidence-ranked weekly pool from raw-volume top 10 before exclusions."""

    monthly_entry_pools = _load_weekly_rotation_pools(
        "weekly-volume-rotation-top10-2026-02-08.json"
    )


class HighVolumeMainWaveWeeklyExpandedRankedV21(
    HighVolumeMainWaveWeeklyRankedV21
):
    """Confidence-ranked weekly top-40-volume pool with frozen V21 logic."""

    monthly_entry_pools = _load_weekly_rotation_pools(
        "weekly-volume-rotation-top40-2026-02-08.json"
    )


class HighVolumeMainWaveThinCapitulationOnlyV16(
    HighVolumeMainWaveAdaptiveRiskV16
):
    """Factor-isolation check retaining only the thin-short exhaustion filter."""

    extended_long_stake_fraction = 1.0


class HighVolumeMainWaveAdaptiveRiskLooseV16(
    HighVolumeMainWaveAdaptiveRiskV16
):
    """Combined robustness check using a 0.35 ATR long-risk boundary."""

    extended_long_minimum_atr = 0.35


class HighVolumeMainWaveAdaptiveRiskTightV16(
    HighVolumeMainWaveAdaptiveRiskV16
):
    """Combined robustness check using a 0.40 ATR long-risk boundary."""

    extended_long_minimum_atr = 0.40


class HighVolumeMainWaveEarlyTrendReservedV15(
    HighVolumeMainWaveEarlyTrendScoutV13
):
    """Keep four core slots and reserve one extra slot for an early scout.

    The runtime/backtest configuration must set ``max_open_trades`` to five.
    Entry confirmation still limits ordinary V11 trades to four concurrent
    positions and early-trend scouts to one, so the fifth slot cannot expand
    normal portfolio risk.
    """

    core_position_slots = 4
    earlytrend_position_slots = 1

    @staticmethod
    def _is_earlytrend_trade(trade: Trade) -> bool:
        return bool(
            trade.enter_tag and "_earlytrend_scout_" in trade.enter_tag
        )

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        open_trades = Trade.get_open_trades()
        earlytrend_count = sum(
            self._is_earlytrend_trade(trade) for trade in open_trades
        )
        core_count = len(open_trades) - earlytrend_count
        if entry_tag and "_earlytrend_scout_" in entry_tag:
            return earlytrend_count < self.earlytrend_position_slots
        return core_count < self.core_position_slots


class HighVolumeMainWaveEarlyTrendDirectionalV14(
    HighVolumeMainWaveEarlyTrendScoutLooseV13
):
    """Use normal scout risk for the stable early-trend short cohort."""

    earlytrend_short_stake_fraction = 0.25


class HighVolumeMainWaveEarlyTrendDirectionalLightV14(
    HighVolumeMainWaveEarlyTrendDirectionalV14
):
    """Nearby early-short sizing check using fifteen-percent scout risk."""

    earlytrend_short_stake_fraction = 0.15


class HighVolumeMainWaveEarlyTrendDirectionalHeavyV14(
    HighVolumeMainWaveEarlyTrendDirectionalV14
):
    """Nearby early-short sizing check using thirty-five-percent scout risk."""

    earlytrend_short_stake_fraction = 0.35


class HighVolumeMainWavePreBreakoutV8(HighVolumeMainWaveV7):
    """Probe a pressure setup before breakout, then upgrade on confirmation.

    The pre-breakout position is always the exchange minimum stake.  It is
    allowed only when a completed 1h candle is within 0.1 ATR of the prior
    Donchian boundary, relative volume is elevated, and the 4h trend agrees.
    If V7 later emits its normal breakout while the probe is open, one position
    adjustment brings the trade up to the stake V7 would have opened at that
    confirmation candle.  An unconfirmed probe is released after twelve hours.
    """

    position_adjustment_enable = True
    max_entry_position_adjustment = 1
    prebreakout_distance_atr = 0.10
    prebreakout_rvol_threshold = 1.25
    prebreakout_minimum_atr_percent = 0.02
    prebreakout_timeout_hours = 12
    prebreakout_long_enabled = True
    prebreakout_short_enabled = True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        signal_tf = self.rotation_profile["signal_tf"]

        active = pd.Series(False, index=dataframe.index)
        pair = metadata["pair"]
        for start, end, pool in self.monthly_entry_pools:
            if pair in pool:
                active |= (dataframe["date"] >= pd.Timestamp(start, tz="UTC")) & (
                    dataframe["date"] < pd.Timestamp(end, tz="UTC")
                )

        atr = dataframe[f"atr14_{signal_tf}"].replace(0, np.nan)
        long_distance = (
            dataframe[f"donchian_high20_{signal_tf}"]
            - dataframe[f"close_{signal_tf}"]
        ) / atr
        short_distance = (
            dataframe[f"close_{signal_tf}"]
            - dataframe[f"donchian_low20_{signal_tf}"]
        ) / atr
        common = (
            active
            & dataframe[f"new_{signal_tf}"]
            & (
                dataframe[f"rvol20_{signal_tf}"]
                >= self.prebreakout_rvol_threshold
            )
            & (
                dataframe[f"atr_percent_{signal_tf}"]
                >= self.prebreakout_minimum_atr_percent
            )
            & (dataframe["volume"] > 0)
        )
        prebreakout_long = (
            self.prebreakout_long_enabled
            & common
            & dataframe["enter_long"].eq(0)
            & long_distance.between(
                0.0, self.prebreakout_distance_atr, inclusive="both"
            )
            & dataframe[f"candle_up_{signal_tf}"]
            & self._bullish_4h(dataframe)
        ).fillna(False)
        prebreakout_short = (
            self.prebreakout_short_enabled
            & common
            & dataframe["enter_short"].eq(0)
            & short_distance.between(
                0.0, self.prebreakout_distance_atr, inclusive="both"
            )
            & dataframe[f"candle_down_{signal_tf}"]
            & self._bearish_4h(dataframe)
        ).fillna(False)
        dataframe.loc[prebreakout_long, ["enter_long", "enter_tag"]] = (
            1,
            f"{signal_tf}_prebreakout_long",
        )
        dataframe.loc[prebreakout_short, ["enter_short", "enter_tag"]] = (
            1,
            f"{signal_tf}_prebreakout_short",
        )
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        if entry_tag and "_prebreakout_" in entry_tag:
            if min_stake is None or min_stake > max_stake:
                return 0.0
            return min_stake
        return super().custom_stake_amount(
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

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        if (
            not trade.enter_tag
            or "_prebreakout_" not in trade.enter_tag
            or trade.nr_of_successful_entries != 1
            or not self.dp
        ):
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None
        latest = dataframe.iloc[-1]
        side = "short" if trade.is_short else "long"
        signal_column = "enter_short" if trade.is_short else "enter_long"
        entry_tag = latest.get("enter_tag")
        if (
            latest.get(signal_column, 0) != 1
            or not isinstance(entry_tag, str)
            or "_breakout_" not in entry_tag
            or not entry_tag.endswith(side)
        ):
            return None

        target_stake = self.custom_stake_amount(
            pair=trade.pair,
            current_time=current_time,
            current_rate=current_rate,
            proposed_stake=max_stake,
            min_stake=min_stake,
            max_stake=max_stake,
            leverage=trade.leverage,
            entry_tag=entry_tag,
            side=side,
        )
        additional_stake = min(target_stake - trade.stake_amount, max_stake)
        if additional_stake <= 0:
            return None
        if min_stake is not None and additional_stake < min_stake:
            return None
        return additional_stake, "prebreakout_confirmed"

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        if trade.enter_tag and "_prebreakout_" in trade.enter_tag:
            if trade.nr_of_successful_entries > 1:
                return None
            if current_time >= trade.open_date_utc + timedelta(
                hours=self.prebreakout_timeout_hours
            ):
                return "prebreakout_timeout"
            return None
        return super().custom_exit(
            pair=pair,
            trade=trade,
            current_time=current_time,
            current_rate=current_rate,
            current_profit=current_profit,
            **kwargs,
        )


class HighVolumeMainWavePreBreakoutWideV8(HighVolumeMainWavePreBreakoutV8):
    """Nearby setup check using a 0.2 ATR distance from the boundary."""

    prebreakout_distance_atr = 0.20


class HighVolumeMainWavePreBreakoutStrictV8(HighVolumeMainWavePreBreakoutV8):
    """Nearby setup check requiring 1.5 relative volume."""

    prebreakout_rvol_threshold = 1.50


class HighVolumeMainWavePreBreakoutShortV8(HighVolumeMainWavePreBreakoutV8):
    """Direction-isolation check retaining only the less harmful short probe."""

    prebreakout_long_enabled = False


class HighVolumeMainWaveScoutBreakEvenV8(HighVolumeMainWaveV7):
    """Protect a confirmed scout from turning into a full failed breakout.

    V2's wide trail is retained for normal main-wave positions.  A scout that
    has travelled 1.5 current ATR in the favorable direction has done enough
    to confirm momentum, so its stop advances just beyond entry to cover the
    approximate round-trip fee.  This targets profit give-back without
    tightening the initial room available to an emerging wave.
    """

    scout_break_even_activation_atr = 1.5
    scout_break_even_buffer = 0.002

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        if not trade.enter_tag or "_scout_" not in trade.enter_tag:
            return super().custom_stoploss(
                pair=pair,
                trade=trade,
                current_time=current_time,
                current_rate=current_rate,
                current_profit=current_profit,
                after_fill=after_fill,
                **kwargs,
            )

        atr = self._latest_atr(pair)
        if atr is None:
            return None

        if trade.is_short:
            best_rate = min(trade.open_rate, current_rate)
            if trade.min_rate is not None:
                best_rate = min(best_rate, trade.min_rate)
            favorable_excursion = trade.open_rate - best_rate
            stop_rate = trade.open_rate + self.initial_stop_atr * atr
            if favorable_excursion >= self.trailing_activation_atr * atr:
                stop_rate = min(
                    stop_rate,
                    best_rate + self.trailing_stop_atr * atr,
                )
            if favorable_excursion >= self.scout_break_even_activation_atr * atr:
                stop_rate = min(
                    stop_rate,
                    trade.open_rate * (1 - self.scout_break_even_buffer),
                )
        else:
            best_rate = max(trade.open_rate, current_rate)
            if trade.max_rate is not None:
                best_rate = max(best_rate, trade.max_rate)
            favorable_excursion = best_rate - trade.open_rate
            stop_rate = trade.open_rate - self.initial_stop_atr * atr
            if favorable_excursion >= self.trailing_activation_atr * atr:
                stop_rate = max(
                    stop_rate,
                    best_rate - self.trailing_stop_atr * atr,
                )
            if favorable_excursion >= self.scout_break_even_activation_atr * atr:
                stop_rate = max(
                    stop_rate,
                    trade.open_rate * (1 + self.scout_break_even_buffer),
                )

        return stoploss_from_absolute(
            stop_rate,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )


class HighVolumeMainWaveScoutBreakEvenEarlyV8(
    HighVolumeMainWaveScoutBreakEvenV8
):
    """Nearby protection check at one ATR of favorable excursion."""

    scout_break_even_activation_atr = 1.0


class HighVolumeMainWaveScoutBreakEvenLateV8(
    HighVolumeMainWaveScoutBreakEvenV8
):
    """Nearby protection check at two ATR of favorable excursion."""

    scout_break_even_activation_atr = 2.0


class HighVolumeMainWaveScoutMinFloorHeavyV7(
    HighVolumeMainWaveScoutMinFloorV7
):
    """Minimum-notional robustness check with twenty-percent scout longs."""

    scout_stake_fraction_long = 0.20
