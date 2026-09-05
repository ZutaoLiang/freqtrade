"""Freqtrade port of the "Bitcoin SuperFlip" Supertrend + EMA trend-following strategy.

Reference: TradingView script NgSIfKQx by blitz_locked ("BTCUSD Supertrend +
EMA Trend Filter (1H)"), Pine v6, MPL-2.0. Ported from the published source.

Logic
-----
* Supertrend (Wilder ATR based, Pine ``ta.supertrend`` semantics) gives the
  directional state. A change of that state is a "flip".
* A long EMA (200 by default) is the directional bias filter: longs only above
  the EMA, shorts only below it.
* An optional ADX filter (default threshold 20) blocks entries in low-trend
  regimes.
* Entry happens on a Supertrend flip that agrees with the EMA bias, exit
  happens on the opposite flip. Percentage stop loss / take profit are
  available but disabled by default, matching the original script.

Pine defaults carried over verbatim: Supertrend ATR 10 / factor 1.8, EMA 200,
ADX filter OFF (length 14, smoothing 14, threshold 20), stop loss OFF (4%),
take profit OFF (8%). Note the Pine ``strategy.exit`` call is only issued when
``enableSL`` is true, so the take profit is inert unless the stop loss is also
enabled; that coupling is reproduced here.
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


def supertrend(dataframe: pd.DataFrame, period: int, multiplier: float) -> pd.DataFrame:
    """Replicate Pine ``ta.supertrend`` (band persistence + trend latch).

    Returns a frame with ``st`` (the plotted line) and ``dir`` where 1 means
    uptrend and -1 means downtrend.
    """
    high = dataframe["high"].to_numpy(dtype=float)
    low = dataframe["low"].to_numpy(dtype=float)
    close = dataframe["close"].to_numpy(dtype=float)

    atr = ta.ATR(dataframe, timeperiod=period).to_numpy(dtype=float)
    hl2 = (high + low) / 2.0
    upper_raw = hl2 - multiplier * atr  # lower band, used while in an uptrend
    lower_raw = hl2 + multiplier * atr  # upper band, used while in a downtrend

    n = len(close)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    direction = np.zeros(n, dtype=float)
    line = np.full(n, np.nan)

    prev_up = np.nan
    prev_dn = np.nan
    prev_dir = 1.0
    for i in range(n):
        u = upper_raw[i]
        d = lower_raw[i]
        if np.isnan(u):
            direction[i] = prev_dir
            continue
        # Pine: nz(up[1], up) then ratchet the band in the trend direction.
        prev_u = prev_up if not np.isnan(prev_up) else u
        prev_d = prev_dn if not np.isnan(prev_dn) else d
        if i > 0 and close[i - 1] > prev_u:
            u = max(u, prev_u)
        if i > 0 and close[i - 1] < prev_d:
            d = min(d, prev_d)

        if prev_dir == -1 and close[i] > prev_d:
            cur_dir = 1.0
        elif prev_dir == 1 and close[i] < prev_u:
            cur_dir = -1.0
        else:
            cur_dir = prev_dir

        up[i] = u
        dn[i] = d
        direction[i] = cur_dir
        line[i] = u if cur_dir == 1 else d
        prev_up, prev_dn, prev_dir = u, d, cur_dir

    return pd.DataFrame({"st": line, "dir": direction}, index=dataframe.index)


class BitcoinSuperFlip(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = True

    # The original script exits on the opposite Supertrend flip only.
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = True  # Only active when use_sl is enabled; see custom_stoploss.
    use_custom_roi = True  # Only active when use_sl and use_tp are enabled; see custom_roi.
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

    startup_candle_count = 400

    # --- Supertrend ---
    atr_period = IntParameter(5, 30, default=10, space="buy", optimize=True)
    atr_multiplier = DecimalParameter(1.0, 6.0, default=1.8, decimals=1, space="buy", optimize=True)

    # --- Trend filter ---
    ema_period = IntParameter(50, 400, default=200, space="buy", optimize=True)

    # --- ADX filter ---
    use_adx = BooleanParameter(default=False, space="buy", optimize=False)
    adx_period = IntParameter(7, 28, default=14, space="buy", optimize=True)
    adx_threshold = DecimalParameter(10.0, 35.0, default=20.0, decimals=1, space="buy", optimize=True)

    # --- Optional risk overlay (off by default, as in the Pine script) ---
    use_sl = BooleanParameter(default=False, space="sell", optimize=False)
    sl_pct = DecimalParameter(1.0, 20.0, default=4.0, decimals=1, space="sell", optimize=False)
    use_tp = BooleanParameter(default=False, space="sell", optimize=False)
    tp_pct = DecimalParameter(1.0, 50.0, default=8.0, decimals=1, space="sell", optimize=False)

    # Directional trading only; leverage stays at 1 like the Pine backtest.
    leverage_value = 1.0

    def custom_roi(self, pair, trade, current_time, trade_duration, entry_tag, side, **kwargs):
        """Take profit, as the ROI equivalent of the Pine limit order.

        Pine attaches the limit to ``strategy.exit``, which is only registered
        when ``enableSL`` is true, so the take profit is inert on its own.
        Routing it through ROI (rather than custom_exit) keeps the intrabar fill
        behaviour of a resting limit order.
        """
        if self.use_sl.value and self.use_tp.value:
            return self.tp_pct.value / 100.0
        return None

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        st = supertrend(dataframe, self.atr_period.value, self.atr_multiplier.value)
        dataframe["st"] = st["st"]
        dataframe["st_dir"] = st["dir"]
        dataframe["st_dir_prev"] = dataframe["st_dir"].shift(1)

        dataframe["ema"] = ta.EMA(dataframe, timeperiod=self.ema_period.value)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_period.value)

        dataframe["flip_up"] = (dataframe["st_dir"] == 1) & (dataframe["st_dir_prev"] == -1)
        dataframe["flip_down"] = (dataframe["st_dir"] == -1) & (dataframe["st_dir_prev"] == 1)
        return dataframe

    def _adx_ok(self, dataframe: pd.DataFrame) -> pd.Series:
        if not self.use_adx.value:
            return pd.Series(True, index=dataframe.index)
        return dataframe["adx"] > self.adx_threshold.value

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        adx_ok = self._adx_ok(dataframe)

        dataframe.loc[
            dataframe["flip_up"] & (dataframe["close"] > dataframe["ema"]) & adx_ok,
            ["enter_long", "enter_tag"],
        ] = (1, "superflip_long")

        dataframe.loc[
            dataframe["flip_down"] & (dataframe["close"] < dataframe["ema"]) & adx_ok,
            ["enter_short", "enter_tag"],
        ] = (1, "superflip_short")
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[dataframe["flip_down"], ["exit_long", "exit_tag"]] = (1, "st_flip_down")
        dataframe.loc[dataframe["flip_up"], ["exit_short", "exit_tag"]] = (1, "st_flip_up")
        return dataframe

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill, **kwargs):
        """Fixed percentage stop measured from the entry price, as in Pine.

        Note the Pine ``strategy.exit`` call is only issued when ``enableSL`` is
        true, so the take profit is inert unless the stop loss is enabled too.
        """
        if not self.use_sl.value:
            return None
        return stoploss_from_open(
            -self.sl_pct.value / 100.0,
            current_profit,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return self.leverage_value
