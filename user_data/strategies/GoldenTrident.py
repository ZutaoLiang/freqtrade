"""Freqtrade port of "Golden Trident | Swing-Anchored VWAP Trend System".

Reference: TradingView script Eq4kPrrJ (Pine v6), a long-only daily XAUUSD
trend system. Ported from the published source.

Logic
-----
* Swing structure: a bar whose high is the highest of the last ``swing_len``
  bars marks a swing high, likewise for lows. ``dir`` is 1 when the last swing
  high is more recent than the last swing low, else -1.
* Entry on the rising edge of (dir == 1 AND close > EMA200 AND chop filter OK).
* Exit as soon as ``dir`` turns -1 ("structure flip").
* Backstop stop loss at 8 x ATR(14) from the entry bar - catastrophe only.
* Shorts are disabled by default, exactly as in the Pine script.

Notes on the original
---------------------
* The anchored VWAP that names the script is computed and plotted but never
  read by the entry or exit logic. It is reproduced here as an indicator
  column for parity, and likewise does not affect trading.
* The chop filter compares a 20-bar high/low range against 0.8 x ATR(20). A
  20-bar range is almost always wider than 0.8 ATR, so it blocks very little.
* Pine sizes the backstop off the signal bar's close; freqtrade enters at the
  next bar's open, so the stop distance differs by one bar of drift.

Pine defaults: swing 30, EMA 200 (filter on), chop filter on with ATR 20 and
0.8x range multiplier, backstop ATR 14 at 8.0x, shorts off, 20% of equity per
trade, 0.02% commission.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib.abstract as ta

from freqtrade.strategy import (
    BooleanParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    stoploss_from_open,
)


def swing_direction(dataframe: pd.DataFrame, swing_len: int) -> pd.Series:
    """Pine: dir = (bar of last swing high) > (bar of last swing low) ? 1 : -1."""
    idx = np.arange(len(dataframe), dtype=float)
    is_ph = dataframe["high"] >= dataframe["high"].rolling(swing_len).max()
    is_pl = dataframe["low"] <= dataframe["low"].rolling(swing_len).min()

    ph_bar = pd.Series(np.where(is_ph, idx, np.nan), index=dataframe.index).ffill().fillna(0.0)
    pl_bar = pd.Series(np.where(is_pl, idx, np.nan), index=dataframe.index).ffill().fillna(0.0)
    return pd.Series(np.where(ph_bar > pl_bar, 1, -1), index=dataframe.index)


def anchored_vwap(dataframe: pd.DataFrame, direction: pd.Series) -> pd.Series:
    """VWAP re-anchored at every change of ``direction`` (plot only in Pine)."""
    hlc3 = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3.0
    pv = hlc3 * dataframe["volume"]
    segment = (direction != direction.shift(1)).cumsum()
    cum_pv = pv.groupby(segment).cumsum()
    cum_v = dataframe["volume"].groupby(segment).cumsum()
    return (cum_pv / cum_v).where(cum_v > 0, hlc3)


class GoldenTrident(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1d"
    can_short = True  # Gated by the allow_short parameter, off by default.

    minimal_roi = {"0": 100.0}
    stoploss = -0.99  # Replaced by the ATR backstop in custom_stoploss.
    trailing_stop = False
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    process_only_new_candles = True

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    startup_candle_count = 250

    swing_len = IntParameter(5, 60, default=30, space="buy", optimize=True)
    ema_len = IntParameter(50, 400, default=200, space="buy", optimize=True)
    use_ema = BooleanParameter(default=True, space="buy", optimize=False)

    use_chop = BooleanParameter(default=True, space="buy", optimize=False)
    atr_chop_len = IntParameter(10, 40, default=20, space="buy", optimize=True)
    range_mult = DecimalParameter(0.1, 3.0, default=0.8, decimals=1, space="buy", optimize=True)

    allow_short = BooleanParameter(default=False, space="buy", optimize=False)

    atr_stop_len = IntParameter(7, 30, default=14, space="sell", optimize=False)
    atr_stop_mult = DecimalParameter(1.0, 12.0, default=8.0, decimals=1, space="sell", optimize=True)

    leverage_value = 1.0

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        direction = swing_direction(dataframe, self.swing_len.value)
        dataframe["dir"] = direction
        dataframe["avwap"] = anchored_vwap(dataframe, direction)

        dataframe["ema"] = ta.EMA(dataframe, timeperiod=self.ema_len.value)
        dataframe["atr_stop"] = ta.ATR(dataframe, timeperiod=self.atr_stop_len.value)
        atr_chop = ta.ATR(dataframe, timeperiod=self.atr_chop_len.value)

        span = (
            dataframe["high"].rolling(self.atr_chop_len.value).max()
            - dataframe["low"].rolling(self.atr_chop_len.value).min()
        )
        dataframe["range_ok"] = (
            (span > atr_chop * self.range_mult.value) if self.use_chop.value else True
        )

        # Backstop distance as a ratio, read back per trade in custom_stoploss.
        dataframe["backstop_ratio"] = (
            dataframe["atr_stop"] * self.atr_stop_mult.value / dataframe["close"]
        )
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        trend_long_ok = (
            (dataframe["close"] > dataframe["ema"]) if self.use_ema.value else True
        )
        trend_short_ok = (
            (dataframe["close"] < dataframe["ema"]) if self.use_ema.value else True
        )

        long_cond = (dataframe["dir"] == 1) & trend_long_ok & dataframe["range_ok"]
        short_cond = (dataframe["dir"] == -1) & trend_short_ok & dataframe["range_ok"]

        # Pine fires on the rising edge of the condition, not while it holds.
        dataframe.loc[
            long_cond & ~long_cond.shift(1).fillna(False),
            ["enter_long", "enter_tag"],
        ] = (1, "structure_long")

        if self.allow_short.value:
            dataframe.loc[
                short_cond & ~short_cond.shift(1).fillna(False),
                ["enter_short", "enter_tag"],
            ] = (1, "structure_short")
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[dataframe["dir"] == -1, ["exit_long", "exit_tag"]] = (1, "structure_flip")
        dataframe.loc[dataframe["dir"] == 1, ["exit_short", "exit_tag"]] = (1, "structure_flip")
        return dataframe

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill, **kwargs):
        """Fixed 8 x ATR backstop, measured from the entry bar."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        # The Pine backstop is set from the signal bar, one bar before the fill.
        entry_rows = dataframe.loc[dataframe["date"] < trade.open_date_utc, "backstop_ratio"]
        if entry_rows.empty or not np.isfinite(entry_rows.iloc[-1]):
            return None
        return stoploss_from_open(
            -float(entry_rows.iloc[-1]),
            current_profit,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return self.leverage_value
