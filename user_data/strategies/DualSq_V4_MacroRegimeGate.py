"""DualSq_V4_MacroRegimeGate -- Direction 1: Macro Regime Filter.

Logic:
- Stricter Macro Regime Gate for BTC:
  - Bull Regime: BTC 1d close > EMA50 AND BTC 1d close > EMA200 (Golden / Bull zone)
  - In Bull Regime: Dual Squeeze Long Breakout is allowed.
  - In Bear Regime: Long trades are completely SLEEPING (dormant to avoid bear market chops).
  - Short trades only when BTC 1d < EMA50 AND BTC 1d < EMA200 AND BTC 4h < EMA50.
- Max hold 18 hours.
- Hard stop -7%, TP +14%.
"""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.persistence import Trade


def calc_bollinger_bands(series: pd.Series, window: int = 20, stds: float = 2.0):
    mid = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + stds * std
    lower = mid - stds * std
    return mid, upper, lower


def calc_keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20, mult: float = 1.5):
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


class DualSq_V4_MacroRegimeGate(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True

    stoploss = -0.07
    minimal_roi = {"0": 0.14}
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count = 200

    leverage_val = 1.5
    max_hold_hours = 18.0
    btc_pair = "BTC/USDT:USDT"

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative = [(pair, "4h") for pair in pairs]
        informative.append((self.btc_pair, "1d"))
        informative.append((self.btc_pair, "4h"))
        return informative

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        # 1h BB & KC
        _, bb_u_1h, bb_l_1h = calc_bollinger_bands(dataframe["close"], window=20, stds=2.0)
        _, kelt_u_1h, kelt_l_1h = calc_keltner_channel(
            dataframe["high"], dataframe["low"], dataframe["close"], window=20, mult=1.5
        )
        dataframe["bb_upper_1h"] = bb_u_1h
        dataframe["bb_lower_1h"] = bb_l_1h
        dataframe["kelt_upper_1h"] = kelt_u_1h
        dataframe["kelt_lower_1h"] = kelt_l_1h

        dataframe["sq_1h"] = (dataframe["bb_upper_1h"] < dataframe["kelt_upper_1h"]) & \
                             (dataframe["bb_lower_1h"] > dataframe["kelt_lower_1h"])

        if self.dp:
            inf_4h = self.dp.get_pair_dataframe(pair=pair, timeframe="4h")
            if not inf_4h.empty:
                _, bb_u_4h, bb_l_4h = calc_bollinger_bands(inf_4h["close"], window=20, stds=2.0)
                _, kelt_u_4h, kelt_l_4h = calc_keltner_channel(
                    inf_4h["high"], inf_4h["low"], inf_4h["close"], window=20, mult=1.5
                )
                inf_4h["sq_4h"] = (bb_u_4h < kelt_u_4h) & (bb_l_4h > kelt_l_4h)
                dataframe = merge_informative_pair(
                    dataframe, inf_4h[["date", "sq_4h"]], self.timeframe, "4h", ffill=True
                )
            else:
                dataframe["sq_4h_4h"] = False

            # Strict BTC Regime Gate (1d EMA50 & EMA200, 4h EMA50)
            btc_1d = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe="1d")
            if not btc_1d.empty:
                btc_1d["btc_ema50_1d"] = btc_1d["close"].ewm(span=50, min_periods=20, adjust=False).mean()
                btc_1d["btc_ema200_1d"] = btc_1d["close"].ewm(span=200, min_periods=50, adjust=False).mean()
                # Strict Macro Bull: Close > EMA50 AND Close > EMA200
                btc_1d["btc_strict_bull_1d"] = (btc_1d["close"] > btc_1d["btc_ema50_1d"]) & \
                                               (btc_1d["close"] > btc_1d["btc_ema200_1d"])
                # Strict Macro Bear: Close < EMA50 AND Close < EMA200
                btc_1d["btc_strict_bear_1d"] = (btc_1d["close"] < btc_1d["btc_ema50_1d"]) & \
                                               (btc_1d["close"] < btc_1d["btc_ema200_1d"])
                dataframe = merge_informative_pair(
                    dataframe, btc_1d[["date", "btc_strict_bull_1d", "btc_strict_bear_1d"]],
                    self.timeframe, "1d", ffill=True
                )
            else:
                dataframe["btc_strict_bull_1d_1d"] = False
                dataframe["btc_strict_bear_1d_1d"] = False

            btc_4h = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe="4h")
            if not btc_4h.empty:
                btc_4h["btc_ema50_4h"] = btc_4h["close"].ewm(span=50, min_periods=20, adjust=False).mean()
                btc_4h["btc_bull_4h"] = btc_4h["close"] > btc_4h["btc_ema50_4h"]
                btc_4h["btc_bear_4h"] = btc_4h["close"] < btc_4h["btc_ema50_4h"]
                dataframe = merge_informative_pair(
                    dataframe, btc_4h[["date", "btc_bull_4h", "btc_bear_4h"]], self.timeframe, "4h", ffill=True
                )
            else:
                dataframe["btc_bull_4h_4h"] = False
                dataframe["btc_bear_4h_4h"] = False
        else:
            dataframe["sq_4h_4h"] = False
            dataframe["btc_strict_bull_1d_1d"] = False
            dataframe["btc_strict_bear_1d_1d"] = False
            dataframe["btc_bull_4h_4h"] = False
            dataframe["btc_bear_4h_4h"] = False

        sq_4h_col = "sq_4h_4h" if "sq_4h_4h" in dataframe.columns else "sq_4h"
        dataframe["dual_sq"] = dataframe["sq_1h"] & dataframe[sq_4h_col].fillna(False)

        b_bull_1d = dataframe.get("btc_strict_bull_1d_1d", pd.Series(False, index=dataframe.index)).fillna(False)
        b_bear_1d = dataframe.get("btc_strict_bear_1d_1d", pd.Series(False, index=dataframe.index)).fillna(False)
        b_bull_4h = dataframe.get("btc_bull_4h_4h", pd.Series(False, index=dataframe.index)).fillna(False)
        b_bear_4h = dataframe.get("btc_bear_4h_4h", pd.Series(False, index=dataframe.index)).fillna(False)

        dataframe["macro_gate_bull"] = b_bull_1d & b_bull_4h
        dataframe["macro_gate_bear"] = b_bear_1d & b_bear_4h

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long Entry: Only in verified macro bull
        long_cond = (
            dataframe["dual_sq"].shift(1).fillna(False) &
            (dataframe["close"] > dataframe["kelt_upper_1h"]) &
            dataframe["macro_gate_bull"]
        )
        dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "macro_gate_long")

        # Short Entry: Only in verified macro bear
        short_cond = (
            dataframe["dual_sq"].shift(1).fillna(False) &
            (dataframe["close"] < dataframe["kelt_lower_1h"]) &
            dataframe["macro_gate_bear"]
        )
        dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "macro_gate_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> str | None:
        if trade.open_date_utc:
            trade_dur_hours = (current_time - trade.open_date_utc).total_seconds() / 3600.0
            if trade_dur_hours >= self.max_hold_hours:
                return "time_expired_18h"
        return None

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        try:
            total_equity = self.wallets.get_total_stake_amount()
        except Exception:
            total_equity = self.config.get("dry_run_wallet", 100.0)

        max_slots = self.config.get("max_open_trades", 10)
        tradable_ratio = self.config.get("tradable_balance_ratio", 0.99)
        target_stake = (total_equity * tradable_ratio) / max_slots
        min_allowed = max(min_stake or 5.0, 5.0)
        stake = max(target_stake, min_allowed)
        return min(stake, max_stake)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return min(self.leverage_val, max_leverage)
