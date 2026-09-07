"""Donchian breakout on the 3d-volatility movers, for engine reconciliation.

This is the research configuration written for the native engine so the two can
be compared trade by trade:

  screen   top 150 perpetuals by 30d dollar volume, keep the 30 with the
           highest 3d realised volatility, refreshed every 3 days
  gate     only bars where |3d mean funding| sits above the cross-sectional
           80th percentile
  entry    close above the 96h Donchian high (long) or below the 96h low
  exit     close through the 12h channel in the other direction, or a
           2 x ATR(24, Wilder) chandelier stop measured on closes

The screen and the gate are panel-wide computations that a per-pair strategy
cannot reproduce, so they are precomputed by
``scripts/ma_harness/export_for_freqtrade.py`` into ``gate.parquet`` and read
here. Everything else -- the channels, the trail, the state machine -- is
recomputed from the candles, which is what the reconciliation is testing.

The state machine runs in ``populate_indicators`` so the signal timing matches
the vectorised research exactly; the remaining differences against it are the
engine's own: fills at the next candle's open rather than at the signal close,
per-order fees, and funding charged on the settlement schedule.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, stoploss_from_absolute

GATE = Path("/root/freqtrade/user_data/research/ma_harness/gate.parquet")


@lru_cache(maxsize=1)
def _gate() -> DataFrame:
    return pd.read_parquet(GATE)


class VolScreenDonchian(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # Exits are signals by default. VSD_HARD_STOP adds a static disaster stop
    # expressed as a price move; freqtrade divides stoploss by leverage to get
    # the price distance, so the class value is the price move times leverage.
    # VSD_ATR_STOP instead hands the ATR trail to the engine as a real stop,
    # which then triggers intrabar rather than on the close.
    minimal_roi = {"0": 100.0}
    _hard = float(os.environ.get("VSD_HARD_STOP", 0.0))
    _lev_env = float(os.environ.get("VSD_LEVERAGE", 1.0))
    stoploss = -(_hard * _lev_env) if _hard > 0 else -0.99
    use_custom_stoploss = bool(int(os.environ.get("VSD_ATR_STOP", 0)))
    trailing_stop = False

    startup_candle_count = 250

    entry_channel = 96
    exit_channel = 12
    atr_window = 24
    atr_mult = 2.0

    # Sizing, read from the environment so a sweep needs no code edit.
    # risk_pct   fraction of equity risked to the initial 2 x ATR stop
    # lev        isolated leverage asked of the exchange
    # max_frac   ceiling on one position's notional as a fraction of equity
    risk_pct = float(os.environ.get("VSD_RISK_PCT", 0.0))
    lev = float(os.environ.get("VSD_LEVERAGE", 1.0))
    max_frac = float(os.environ.get("VSD_MAX_FRAC", 0.5))
    flat_frac = float(os.environ.get("VSD_FLAT_FRAC", 0.0))
    # 1: leaving the basket closes the position. 0: the basket gates entries
    # only and an open position runs to its own exit.
    gate_exit = bool(int(os.environ.get("VSD_GATE_EXIT", 1)))

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return min(self.lev, max_leverage)

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs) -> float | None:
        """Hand the ATR trail to the engine so it can fire inside a candle.

        Without this the trail is only checked on closes, which is what let
        four positions run all the way to liquidation.
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None
        stop = df["trail_stop"].iloc[-1]
        if not np.isfinite(stop):
            return None
        return stoploss_from_absolute(float(stop), current_rate,
                                      is_short=trade.is_short,
                                      leverage=trade.leverage)

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side,
                            **kwargs) -> float:
        """Size by the distance to the initial stop, not by a fixed slice.

        The trail sits 2 x ATR away at entry, so risking a fixed fraction of
        equity against that distance equalises risk across coins whose
        volatility differs by an order of magnitude. Returned value is margin;
        notional is margin x leverage.
        """
        if self.risk_pct <= 0 and self.flat_frac <= 0:
            return proposed_stake        # fixed-notional mode, no equity feedback
        equity = self.wallets.get_total_stake_amount()
        if self.flat_frac > 0:
            notional = self.flat_frac * equity * leverage
        else:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or df.empty:
                return proposed_stake
            atr = float(df["atr"].iloc[-1])
            if not (atr > 0) or not (current_rate > 0):
                return proposed_stake
            stop_distance = self.atr_mult * atr / current_rate
            notional = (self.risk_pct * equity) / stop_distance
        notional = min(notional, self.max_frac * equity * leverage)
        stake = notional / leverage
        if min_stake is not None:
            stake = max(stake, min_stake)
        return float(min(stake, max_stake))

    def _atr(self, df: DataFrame) -> np.ndarray:
        prev = df["close"].shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                        (df["low"] - prev).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / self.atr_window, adjust=False).mean().to_numpy()

    def _state(self, close, high, low, atr) -> np.ndarray:
        """The same stop-and-stay state machine the research runs.

        The machine itself never sees the gate: the research builds the raw
        breakout state over the full history and then multiplies it by the
        screen-and-funding mask, so a position masked off comes back when the
        mask reopens rather than waiting for a fresh breakout. Applying the
        mask afterwards here reproduces that exactly.
        """
        hh = pd.Series(high).rolling(self.entry_channel).max().shift(1).to_numpy()
        ll = pd.Series(low).rolling(self.entry_channel).min().shift(1).to_numpy()
        xl = pd.Series(low).rolling(self.exit_channel).min().shift(1).to_numpy()
        xh = pd.Series(high).rolling(self.exit_channel).max().shift(1).to_numpy()
        out = np.zeros(close.size)
        stops = np.full(close.size, np.nan)
        pos, stop = 0.0, np.nan
        for i in range(close.size):
            c = close[i]
            if pos > 0:
                if np.isfinite(atr[i]):
                    stop = max(stop, c - self.atr_mult * atr[i])
                if c < stop:
                    pos = 0.0
            elif pos < 0:
                if np.isfinite(atr[i]):
                    stop = min(stop, c + self.atr_mult * atr[i])
                if c > stop:
                    pos = 0.0
            if pos > 0 and np.isfinite(xl[i]) and c < xl[i]:
                pos = 0.0
            elif pos < 0 and np.isfinite(xh[i]) and c > xh[i]:
                pos = 0.0
            if pos == 0.0:
                if np.isfinite(hh[i]) and c > hh[i]:
                    pos, stop = 1.0, c - self.atr_mult * atr[i]
                elif np.isfinite(ll[i]) and c < ll[i]:
                    pos, stop = -1.0, c + self.atr_mult * atr[i]
            out[i] = pos
            stops[i] = stop if pos != 0 else np.nan
        return out, stops

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        g = _gate()
        pair = metadata["pair"]
        if pair in g.columns:
            s = g[pair].reindex(dataframe["date"]).to_numpy(
                dtype=bool, na_value=False)
        else:
            s = np.zeros(len(dataframe), dtype=bool)
        dataframe["gate"] = s
        atr = self._atr(dataframe)
        dataframe["atr"] = atr
        raw, stops = self._state(dataframe["close"].to_numpy(),
                                 dataframe["high"].to_numpy(),
                                 dataframe["low"].to_numpy(), atr)
        if self.gate_exit:
            # a coin leaving the basket closes the position at the next open
            dataframe["pos"] = np.where(s, raw, 0.0)
        else:
            # the basket only decides where new positions may be opened; an
            # existing one runs until its own channel or trail says to leave
            entered = np.zeros(len(raw), dtype=bool)
            held = 0.0
            out = np.zeros(len(raw))
            for i in range(len(raw)):
                if held == 0.0:
                    if raw[i] != 0.0 and s[i] and (i == 0 or raw[i - 1] != raw[i]):
                        held = raw[i]
                elif raw[i] == 0.0 or raw[i] != held:
                    held = raw[i] if (raw[i] != 0.0 and s[i]) else 0.0
                out[i] = held
            dataframe["pos"] = out
        dataframe["trail_stop"] = stops
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pos = dataframe["pos"]
        prev = pos.shift(1).fillna(0.0)
        dataframe.loc[(pos > 0) & (prev <= 0), "enter_long"] = 1
        dataframe.loc[(pos < 0) & (prev >= 0), "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pos = dataframe["pos"]
        prev = pos.shift(1).fillna(0.0)
        dataframe.loc[(prev > 0) & (pos <= 0), "exit_long"] = 1
        dataframe.loc[(prev < 0) & (pos >= 0), "exit_short"] = 1
        return dataframe
