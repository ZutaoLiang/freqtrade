"""Cross-sectional multi-horizon momentum ensemble, long/short, daily rebalance.

Origin
------
Event-study first (scripts/analyze_mom_ensemble_quarters.py): the equal-weight
rank blend of 3d/14d/30d momentum, long top-5 / short bottom-5 inside a
point-in-time liquidity-filtered universe, was net-of-fee positive in both the
2025 rotating-pool universe and the 2026 all-perps universe, with 6 of 7
native quarters positive.  Single-horizon momentum and funding carry both
failed the same split test and must not be revived (see the
high-volume-trend-research skill for the falsified list).

Mechanics
---------
* Universe eligibility per rebalance: trailing 7d mean of 24h quote volume
  above ``min_quote_volume`` (default 30M USDT), computed from the panel.
* Score: mean of the cross-sectional percentile ranks of 3d, 14d and 30d
  returns among eligible pairs.  Long the top ``basket_size``, short the
  bottom ``basket_size``.
* Rebalance every ``rebalance_hours`` (default 24) on a fixed UTC grid; the
  book is held between boundaries.  No regime filter, no per-trade tiering.
* Equal stakes per slot from compounding equity; disaster stop only.
* Position sizes are trimmed/topped back to the equal-weight target at every
  boundary (adjust_trade_position).  Without this, a runaway winner compounds
  inside its slot and the year degenerates into a single-trade lottery -- the
  exact concentration failure the research skill documents; first run of this
  strategy without trimming reproduced it (one PIPPIN trade > total profit).

The panel/caching plumbing follows TrendRotationV1.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)


class XsMomEnsembleV1(IStrategy):

    timeframe = "1h"
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    minimal_roi = {"0": 100}
    position_adjustment_enable = True
    max_entry_position_adjustment = -1

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.mom_hours = [int(x) for x in
                          str(self.get_config("mom_hours", "72,336,720")).split(",")]
        self.basket_size = int(self.get_config("basket_size", 5))
        self.min_universe = int(self.get_config("min_universe", 2 * self.basket_size + 2))
        self.rebalance_hours = float(self.get_config("rebalance_hours", 24))
        self.min_quote_volume = float(self.get_config("min_quote_volume", 30e6))
        self.qvol_hours = int(self.get_config("qvol_hours", 168))

        self.total_stake_ratio = float(self.get_config("total_stake_ratio", 0.95))
        # Trim/top-up back to target only when the position drifted further than this.
        self.rebalance_tolerance = float(self.get_config("rebalance_tolerance", 0.15))
        self.trade_leverage = float(self.get_config("trade_leverage", 1))
        self.min_notional = float(self.get_config("min_notional", 5.5))

        # Disaster stop only: the event study rode positions through candle
        # wicks and cut losers via the daily equal-weight trim instead.  A
        # tight hard stop realises wick extremes and was the dominant loss
        # source in the first runs (2026: stops -2879 vs rotations +2907).
        self.stoploss = -abs(float(self.get_config("base_stop_loss", 0.90)))
        self.trailing_stop = False
        self.use_custom_stoploss = False

        self.startup_candle_count = max(self.mom_hours) + self.qvol_hours + 30

        self._panel_cache_key: tuple | None = None
        self._panel_cache: dict[str, DataFrame] = {}

    def get_config(self, key: str, default):
        return self.config.get(key, default)

    def get_pairs(self) -> list[str]:
        return self.config["exchange"]["pair_whitelist"]

    def informative_pairs(self):
        return [(pair, self.timeframe) for pair in self.get_pairs()]

    # ------------------------------------------------------------------ panel

    def _panels(self) -> tuple[DataFrame, DataFrame]:
        """Close and quote-volume panels of the whole universe on one index."""
        closes: dict[str, pd.Series] = {}
        qvols: dict[str, pd.Series] = {}
        for pair in self.get_pairs():
            frame = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            if frame is None or frame.empty:
                continue
            index = pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
            close = pd.Series(frame["close"].to_numpy(dtype=float), index=index)
            qvol = pd.Series(
                (frame["close"] * frame["volume"]).to_numpy(dtype=float), index=index
            )
            keep = ~close.index.duplicated(keep="last")
            closes[pair] = close.loc[keep]
            qvols[pair] = qvol.loc[keep]
        if not closes:
            return DataFrame(), DataFrame()
        return (
            pd.concat(closes, axis=1).sort_index(),
            pd.concat(qvols, axis=1).sort_index(),
        )

    def _build_panel_signals(self) -> dict[str, DataFrame]:
        close, qvol = self._panels()
        if close.empty:
            return {}

        # Trailing 7d mean of 24h quote volume; eligibility is point-in-time.
        qvol24 = qvol.rolling(24, min_periods=12).sum()
        qvol_trail = qvol24.rolling(self.qvol_hours, min_periods=self.qvol_hours // 2).mean()
        eligible = qvol_trail > self.min_quote_volume

        score = None
        for hours in self.mom_hours:
            mom = close / close.shift(hours) - 1.0
            r = mom.where(eligible).rank(axis=1, pct=True)
            score = r if score is None else score + r
        score = score / len(self.mom_hours)

        nn = score.notna().sum(axis=1)
        ranks_low = score.rank(axis=1, method="first", ascending=True)
        ranks_high = score.rank(axis=1, method="first", ascending=False)
        k = self.basket_size
        want_short = ranks_low.le(k) & score.notna()
        want_long = ranks_high.le(k) & score.notna()
        enough = nn >= self.min_universe
        want_short = want_short.where(enough, other=False)
        want_long = want_long.where(enough, other=False)

        # Equal weights per slot.  An inverse-vol (equal-risk) weighting was
        # tried and flipped BOTH years negative (2025 -12%, 2026 -5%): the
        # momentum profit lives precisely in the highest-vol names, so
        # downweighting them removes the edge and leaves fees.  Do not revive.
        weight = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for side_mask in (want_long, want_short):
            cnt = side_mask.sum(axis=1).replace(0, np.nan)
            weight = weight + 0.5 * side_mask.astype(float).div(cnt, axis=0).fillna(0.0)

        # Freeze at the rebalance grid, hold in between.
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        epoch_minutes = (close.index - epoch) // pd.Timedelta(minutes=1)
        step = int(round(self.rebalance_hours * 60))
        boundary = pd.Series((epoch_minutes % step) == 0, index=close.index)
        boundary.iloc[0] = True

        want_long = want_long.where(boundary, other=np.nan).ffill().fillna(False).astype(bool)
        want_short = want_short.where(boundary, other=np.nan).ffill().fillna(False).astype(bool)
        weight = weight.where(boundary, other=np.nan).ffill().fillna(0.0)

        signals: dict[str, DataFrame] = {}
        for pair in close.columns:
            signals[pair] = DataFrame(
                {
                    "want_long": want_long[pair],
                    "want_short": want_short[pair],
                    "weight": weight[pair],
                },
                index=close.index,
            )
        return signals

    def _weight_at(self, pair: str, when: datetime) -> float:
        frame = self._panel_cache.get(pair)
        if frame is None or frame.empty:
            return 1.0 / (2 * self.basket_size)
        ts = pd.Timestamp(when)
        pos = frame.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            return 1.0 / (2 * self.basket_size)
        w = float(frame["weight"].iloc[pos])
        return w if w > 0 else 1.0 / (2 * self.basket_size)

    def _signals_for(self, pair: str, dataframe: DataFrame) -> DataFrame | None:
        if dataframe.empty:
            return None
        cache_key = (pd.Timestamp(dataframe["date"].iloc[-1]).value,)
        if cache_key != self._panel_cache_key:
            self._panel_cache = self._build_panel_signals()
            self._panel_cache_key = cache_key
        return self._panel_cache.get(pair)

    # ------------------------------------------------------------- freqtrade

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["want_long"] = False
        dataframe["want_short"] = False
        try:
            signals = self._signals_for(metadata["pair"], dataframe)
            if signals is None or signals.empty:
                return dataframe
            dates = pd.DatetimeIndex(pd.to_datetime(dataframe["date"], utc=True))
            aligned = signals.reindex(dates)
            dataframe["want_long"] = aligned["want_long"].fillna(False).to_numpy(dtype=bool)
            dataframe["want_short"] = aligned["want_short"].fillna(False).to_numpy(dtype=bool)
        except Exception as exc:
            logger.error("Error in %s::populate_indicators: %s", self.__class__.__name__, exc)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        if dataframe.empty:
            return dataframe
        dataframe.loc[dataframe["want_long"], ["enter_long", "enter_tag"]] = (1, "xs_long")
        dataframe.loc[dataframe["want_short"], ["enter_short", "enter_tag"]] = (1, "xs_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        if dataframe.empty:
            return dataframe
        dataframe.loc[~dataframe["want_long"], "exit_long"] = 1
        dataframe.loc[~dataframe["want_short"], "exit_short"] = 1
        dataframe.loc[
            (dataframe["exit_long"] == 1) | (dataframe["exit_short"] == 1), "exit_tag"
        ] = "rotate_out"
        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return min(self.trade_leverage, max_leverage)

    def _target_stake(self, pair: str, when: datetime) -> float:
        try:
            equity = float(self.wallets.get_total_stake_amount())
        except Exception:
            return 0.0
        if equity <= 0:
            return 0.0
        return equity * self.total_stake_ratio * self._weight_at(pair, when)

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None:
        # Only reshape on the rebalance grid, mirroring the signal boundaries.
        step_h = self.rebalance_hours
        if (current_time.hour % max(int(step_h), 1)) != 0 or current_time.minute != 0:
            return None
        target = self._target_stake(trade.pair, current_time)
        if target <= 0:
            return None
        current_value = float(trade.amount) * current_rate / max(float(trade.leverage), 1.0)
        drift = current_value / target - 1.0
        if abs(drift) < self.rebalance_tolerance:
            return None
        delta = target - current_value
        if abs(delta) < self.min_notional:
            return None
        if delta < 0:
            # Backtesting converts a negative return into an exit amount via
            # amount * |delta| / stake_amount, i.e. it must be expressed as a
            # share of the ENTRY margin, not of current value.  A value-based
            # delta larger than the position is silently rejected, which is
            # exactly how runaway winners escaped trimming in the first run.
            frac = min(-delta / current_value, 0.9)
            return -frac * float(trade.stake_amount)
        if min_stake is not None and delta < min_stake:
            return None
        return min(delta, max_stake)

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side,
                            **kwargs) -> float:
        stake = self._target_stake(pair, current_time)
        if stake <= 0:
            stake = proposed_stake
        if stake * leverage < self.min_notional:
            stake = self.min_notional / leverage
        if min_stake is not None:
            stake = max(stake, min_stake)
        return min(stake, max_stake)
