"""DualSqueezeAiSector1h -- Refined Dual Squeeze strategy for AI Theme Sector.

Enhancements over baseline DualSqueezeBtcTrend1h:
1. Volume Expansion Gate: requires breakout volume > vol_mult * SMA(volume, 20).
2. Dynamic Breakeven & Trailing: once profit >= 3.0%, stoploss moves to +0.2% (breakeven).
   Once profit >= 6.0%, trailing stop of 2.5% locks in profits.
3. Stagnation Exit: if after 6 hours trade is underwater (profit < -1.0%), exit immediately
   rather than absorbing chop decay for 18 hours.
4. Relative Strength Gate: Longs must outperform BTC over 24h; Shorts must underperform BTC.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair, stoploss_from_open


def calc_bollinger_bands(series: pd.Series, window: int = 20, stds: float = 2.0):
  mid = series.rolling(window, min_periods=window).mean()
  std = series.rolling(window, min_periods=window).std(ddof=0)
  upper = mid + stds * std
  lower = mid - stds * std
  return mid, upper, lower


def calc_keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    mult: float = 1.5,
):
  close_prev = close.shift(1)
  tr1 = high - low
  tr2 = (high - close_prev).abs()
  tr3 = (low - close_prev).abs()
  tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
  atr = tr.ewm(span=window, min_periods=window, adjust=False).mean()
  mid = close.ewm(span=window, min_periods=window, adjust=False).mean()
  upper = mid + mult * atr
  lower = mid - mult * atr
  return mid, upper, lower


class DualSqueezeAiSector1h(IStrategy):
  INTERFACE_VERSION = 3
  timeframe = "1h"
  can_short = True

  # Disaster hard stoploss
  stoploss = -0.50
  use_custom_stoploss = True
  use_exit_signal = True
  process_only_new_candles = True
  startup_candle_count = 100

  # Strategy parameters (Base defaults)
  vol_mult = 1.3  # Volume threshold multiplier
  use_volume_filter = True  # Require volume > vol_mult * volume_sma20
  use_breakeven = True  # Move stoploss to breakeven at +3.0%
  use_trailing = True  # Trail profit above +6.0%
  use_stagnation_exit = True  # Exit underwater trades at 6h
  use_rs_filter = False  # Relative strength vs BTC 24h

  hard_stop_pct = 0.06  # 6.0% hard stoploss (price move)
  take_profit_pct = 0.12  # 12.0% take profit
  max_hold_hours = 18.0  # Max hold duration
  stagnation_hours = 6.0  # Stagnation check duration

  btc_pair = "BTC/USDT:USDT"

  def informative_pairs(self):
    pairs = self.dp.current_whitelist()
    informative_pairs = [(pair, "4h") for pair in pairs]
    informative_pairs.append((self.btc_pair, "1d"))
    informative_pairs.append((self.btc_pair, "4h"))
    informative_pairs.append((self.btc_pair, "1h"))
    return informative_pairs

  def populate_indicators(
      self, dataframe: DataFrame, metadata: dict
  ) -> DataFrame:
    pair = metadata["pair"]

    # 1. Base Timeframe (1h): Bollinger Bands & Keltner Channel
    _, bb_u_1h, bb_l_1h = calc_bollinger_bands(
        dataframe["close"], window=20, stds=2.0
    )
    _, kelt_u_1h, kelt_l_1h = calc_keltner_channel(
        dataframe["high"],
        dataframe["low"],
        dataframe["close"],
        window=20,
        mult=1.5,
    )
    dataframe["bb_upper_1h"] = bb_u_1h
    dataframe["bb_lower_1h"] = bb_l_1h
    dataframe["kelt_upper_1h"] = kelt_u_1h
    dataframe["kelt_lower_1h"] = kelt_l_1h

    # 1h Squeeze condition
    dataframe["sq_1h"] = (
        dataframe["bb_upper_1h"] < dataframe["kelt_upper_1h"]
    ) & (dataframe["bb_lower_1h"] > dataframe["kelt_lower_1h"])

    # Volume SMA
    dataframe["vol_sma20"] = (
        dataframe["volume"].rolling(20, min_periods=10).mean()
    )
    dataframe["vol_surge"] = dataframe["volume"] > (
        self.vol_mult * dataframe["vol_sma20"]
    )

    # 24h Return of this coin
    dataframe["ret_24h"] = dataframe["close"].pct_change(24)

    # 2. Informative Timeframe (4h): 4h Squeeze
    if self.dp:
      inf_4h = self.dp.get_pair_dataframe(pair=pair, timeframe="4h")
      if not inf_4h.empty:
        _, bb_u_4h, bb_l_4h = calc_bollinger_bands(
            inf_4h["close"], window=20, stds=2.0
        )
        _, kelt_u_4h, kelt_l_4h = calc_keltner_channel(
            inf_4h["high"], inf_4h["low"], inf_4h["close"], window=20, mult=1.5
        )
        inf_4h["sq_4h"] = (bb_u_4h < kelt_u_4h) & (bb_l_4h > kelt_l_4h)
        dataframe = merge_informative_pair(
            dataframe,
            inf_4h[["date", "sq_4h"]],
            self.timeframe,
            "4h",
            ffill=True,
        )
      else:
        dataframe["sq_4h_4h"] = False

      # 3. Informative BTC Macro Trend
      btc_1d = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe="1d")
      if not btc_1d.empty:
        btc_1d["btc_sma50_1d"] = (
            btc_1d["close"].rolling(50, min_periods=20).mean()
        )
        btc_1d["btc_bull_1d"] = btc_1d["close"] > btc_1d["btc_sma50_1d"]
        dataframe = merge_informative_pair(
            dataframe,
            btc_1d[["date", "btc_bull_1d"]],
            self.timeframe,
            "1d",
            ffill=True,
        )
      else:
        dataframe["btc_bull_1d_1d"] = False

      btc_4h = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe="4h")
      if not btc_4h.empty:
        btc_4h["btc_ema50_4h"] = (
            btc_4h["close"].ewm(span=50, min_periods=20, adjust=False).mean()
        )
        btc_4h["btc_bull_4h"] = btc_4h["close"] > btc_4h["btc_ema50_4h"]
        dataframe = merge_informative_pair(
            dataframe,
            btc_4h[["date", "btc_bull_4h"]],
            self.timeframe,
            "4h",
            ffill=True,
        )
      else:
        dataframe["btc_bull_4h_4h"] = False

      # BTC 1h for 24h return comparison
      btc_1h = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe="1h")
      if not btc_1h.empty:
        btc_1h["btc_ret_24h"] = btc_1h["close"].pct_change(24)
        dataframe = merge_informative_pair(
            dataframe,
            btc_1h[["date", "btc_ret_24h"]],
            self.timeframe,
            "1h",
            ffill=True,
        )
      else:
        dataframe["btc_ret_24h_1h"] = 0.0
    else:
      dataframe["sq_4h_4h"] = False
      dataframe["btc_bull_1d_1d"] = False
      dataframe["btc_bull_4h_4h"] = False
      dataframe["btc_ret_24h_1h"] = 0.0

    sq_4h_col = "sq_4h_4h" if "sq_4h_4h" in dataframe.columns else "sq_4h"
    dataframe["dual_sq"] = dataframe["sq_1h"] & dataframe[sq_4h_col].fillna(
        False
    )

    btc_bull_col = (
        "btc_bull_1d_1d"
        if "btc_bull_1d_1d" in dataframe.columns
        else "btc_bull_1d"
    )
    btc_4h_col = (
        "btc_bull_4h_4h"
        if "btc_bull_4h_4h" in dataframe.columns
        else "btc_bull_4h"
    )
    dataframe["btc_macro_bull"] = dataframe[btc_bull_col].fillna(
        False
    ) & dataframe[btc_4h_col].fillna(False)
    dataframe["btc_macro_bear"] = (
        ~dataframe[btc_bull_col].fillna(True)
    ) & (~dataframe[btc_4h_col].fillna(True))

    btc_ret_col = (
        "btc_ret_24h_1h"
        if "btc_ret_24h_1h" in dataframe.columns
        else "btc_ret_24h"
    )
    dataframe["outperforming_btc"] = dataframe["ret_24h"] > dataframe[
        btc_ret_col
    ].fillna(0.0)
    dataframe["underperforming_btc"] = dataframe["ret_24h"] < dataframe[
        btc_ret_col
    ].fillna(0.0)

    return dataframe

  def populate_entry_trend(
      self, dataframe: DataFrame, metadata: dict
  ) -> DataFrame:
    # Long Entry
    long_cond = (
        dataframe["dual_sq"].shift(1).fillna(False)
        & (dataframe["close"] > dataframe["kelt_upper_1h"])
        & dataframe["btc_macro_bull"]
    )
    if self.use_volume_filter:
      long_cond = long_cond & dataframe["vol_surge"]
    if self.use_rs_filter:
      long_cond = long_cond & dataframe["outperforming_btc"]

    dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (
        1,
        "dual_sq_long",
    )

    # Short Entry
    short_cond = (
        dataframe["dual_sq"].shift(1).fillna(False)
        & (dataframe["close"] < dataframe["kelt_lower_1h"])
        & dataframe["btc_macro_bear"]
    )
    if self.use_volume_filter:
      short_cond = short_cond & dataframe["vol_surge"]
    if self.use_rs_filter:
      short_cond = short_cond & dataframe["underperforming_btc"]

    dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (
        1,
        "dual_sq_short",
    )

    return dataframe

  def populate_exit_trend(
      self, dataframe: DataFrame, metadata: dict
  ) -> DataFrame:
    dataframe["exit_long"] = 0
    dataframe["exit_short"] = 0
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
    """Dynamic Stoploss Management:

    - Default: -6.0% hard stop from open price.
    - Breakeven: if profit >= +3.0%, stoploss moves to +0.2% (locks in profit
    and covers fees).
    - Trailing: if profit >= +6.0%, trail by 2.5% from peak.
    """
    # 1. Trailing Stop
    if self.use_trailing and current_profit >= 0.06:
      # Trail 2.5% behind current profit
      return stoploss_from_open(
          current_profit - 0.025,
          current_profit,
          is_short=trade.is_short,
          leverage=trade.leverage,
      )

    # 2. Breakeven Stop
    if self.use_breakeven and current_profit >= 0.03:
      return stoploss_from_open(
          0.002,
          current_profit,
          is_short=trade.is_short,
          leverage=trade.leverage,
      )

    # 3. Initial Hard Stoploss (-6.0%)
    return stoploss_from_open(
        -self.hard_stop_pct,
        current_profit,
        is_short=trade.is_short,
        leverage=trade.leverage,
    )

  def custom_exit(
      self,
      pair: str,
      trade: Trade,
      current_time: datetime,
      current_rate: float,
      current_profit: float,
      **kwargs,
  ) -> str | None:
    # 1. Take Profit (+12.0%)
    if current_profit >= self.take_profit_pct:
      return f"tp_{trade.enter_tag}"

    if trade.open_date_utc:
      dur_hours = (current_time - trade.open_date_utc).total_seconds() / 3600.0

      # 2. Stagnation Exit: after 6h, if trade is underwater, cut it early
      if (
          self.use_stagnation_exit
          and dur_hours >= self.stagnation_hours
          and current_profit < -0.01
      ):
        return "stagnation_exit_6h"

      # 3. Maximum Holding Period (18h)
      if dur_hours >= self.max_hold_hours:
        return "time_expired_18h"

    return None

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


# Subclasses for testing specific feature combinations
class DualSqAi_V1_VolFilter(DualSqueezeAiSector1h):
  """Variant 1: Volume filter only (vol > 1.3x SMA20), classic exits."""

  use_volume_filter = True
  vol_mult = 1.3
  use_breakeven = False
  use_trailing = False
  use_stagnation_exit = False
  use_rs_filter = False


class DualSqAi_V2_VolAndBreakeven(DualSqueezeAiSector1h):
  """Variant 2: Volume filter + Breakeven (+3% -> +0.2%)."""

  use_volume_filter = True
  vol_mult = 1.3
  use_breakeven = True
  use_trailing = False
  use_stagnation_exit = False
  use_rs_filter = False


class DualSqAi_V3_VolAndStagnation(DualSqueezeAiSector1h):
  """Variant 3: Volume filter + Stagnation exit at 6h."""

  use_volume_filter = True
  vol_mult = 1.3
  use_breakeven = False
  use_trailing = False
  use_stagnation_exit = True
  use_rs_filter = False


class DualSqAi_V4_FullSuite(DualSqueezeAiSector1h):
  """Variant 4: Volume filter + Breakeven/Trailing + Stagnation exit."""

  use_volume_filter = True
  vol_mult = 1.3
  use_breakeven = True
  use_trailing = True
  use_stagnation_exit = True
  use_rs_filter = False


class DualSqAi_V5_StrictRS(DualSqueezeAiSector1h):
  """Variant 5: Full suite + 24h Relative Strength vs BTC."""

  use_volume_filter = True
  vol_mult = 1.3
  use_breakeven = True
  use_trailing = True
  use_stagnation_exit = True
  use_rs_filter = True


class DualSqAi_NoTimeExit_TP14(DualSqueezeAiSector1h):
  """No time exit: pure TP +14% / SL -7% (baseline entries, holding until TP or SL hits)."""

  use_volume_filter = False
  hard_stop_pct = 0.07
  take_profit_pct = 0.14
  max_hold_hours = float("inf")
  use_breakeven = False
  use_trailing = False
  use_stagnation_exit = False
  use_rs_filter = False


class DualSqAi_NoTimeExit_Vol_TP14(DualSqueezeAiSector1h):
  """No time exit with Vol filter > 1.3x, TP +14% / SL -7%."""

  use_volume_filter = True
  vol_mult = 1.3
  hard_stop_pct = 0.07
  take_profit_pct = 0.14
  max_hold_hours = float("inf")
  use_breakeven = False
  use_trailing = False
  use_stagnation_exit = False
  use_rs_filter = False


class DualSqAi_NoTimeExit_Vol_TP8(DualSqueezeAiSector1h):
  """No time exit with Vol filter > 1.3x, realistic TP +8% / SL -6%."""

  use_volume_filter = True
  vol_mult = 1.3
  hard_stop_pct = 0.06
  take_profit_pct = 0.08
  max_hold_hours = float("inf")
  use_breakeven = False
  use_trailing = False
  use_stagnation_exit = False
  use_rs_filter = False


class DualSqAi_ConditionalTimeExit(DualSqueezeAiSector1h):
  """Conditional Time Exit:

  - If trade duration >= 18h and profit <= 0.0: cut immediately (loss cut).
  - If trade duration >= 18h and profit > 0.0: allow it to run up to 36h.
  - If trade duration >= 36h: time_expired_36h.
  - TP: +14%, SL: -7%.
  """

  use_volume_filter = False
  hard_stop_pct = 0.07
  take_profit_pct = 0.14
  max_hold_hours = 36.0
  conditional_loss_cut_hours = 18.0
  use_breakeven = False
  use_trailing = False
  use_stagnation_exit = False
  use_rs_filter = False

  def custom_exit(
      self,
      pair: str,
      trade: Trade,
      current_time: datetime,
      current_rate: float,
      current_profit: float,
      **kwargs,
  ) -> str | None:
    if current_profit >= self.take_profit_pct:
      return f"tp_{trade.enter_tag}"

    if trade.open_date_utc:
      dur_hours = (current_time - trade.open_date_utc).total_seconds() / 3600.0
      if dur_hours >= self.conditional_loss_cut_hours and current_profit <= 0.00:
        return "time_loss_cut_18h"
      if dur_hours >= self.max_hold_hours:
        return "time_expired_36h"

    return None


class DualSqAi_ConditionalTimeExit_Vol(DualSqAi_ConditionalTimeExit):
  """Same conditional time exit + Volume filter > 1.3x."""

  use_volume_filter = True
  vol_mult = 1.3


