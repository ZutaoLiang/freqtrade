"""DualSq_V5_SymmetricRefined -- Direction 2: Refined Symmetrical Long/Short.

Key Improvements:
- Long Entry:
  - Dual Squeeze + Breakout above Keltner Upper + BTC Bullish consensus.
  - Volume > 1.1x SMA20 to filter out weak breakouts.
- Short Entry (Refined to prevent short squeezes on Meme/Altcoins):
  - Dual Squeeze + Breakdown below Keltner Lower + BTC Bearish consensus.
  - Squeeze quality check: RSI_1h > 35 (Never short into deeply oversold panic wicks!).
  - Volume > 1.2x SMA20 (Confirm institutional liquidation/selling).
  - Quick take profit (+8% on shorts or 12h time exit for shorts, vs 18h for longs).
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


def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class DualSq_V5_SymmetricRefined(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True

    stoploss = -0.07
    minimal_roi = {
        "0": 0.14
    }

    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count = 60

    leverage_val = 1.5
    max_hold_hours_long = 18.0
    max_hold_hours_short = 12.0  # Shorts drop faster and rebound faster
    btc_pair = "BTC/USDT:USDT"

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative = [(pair, "4h") for pair in pairs]
        informative.append((self.btc_pair, "1d"))
        informative.append((self.btc_pair, "4h"))
        return informative

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        # 1h indicators
        _, bb_u_1h, bb_l_1h = calc_bollinger_bands(dataframe["close"], window=20, stds=2.0)
        _, kelt_u_1h, kelt_l_1h = calc_keltner_channel(
            dataframe["high"], dataframe["low"], dataframe["close"], window=20, mult=1.5
        )
        dataframe["bb_upper_1h"] = bb_u_1h
        dataframe["bb_lower_1h"] = bb_l_1h
        dataframe["kelt_upper_1h"] = kelt_u_1h
        dataframe["kelt_lower_1h"] = kelt_l_1h
        dataframe["rsi_1h"] = calc_rsi(dataframe["close"], window=14)
        dataframe["vol_sma20"] = dataframe["volume"].rolling(20, min_periods=10).mean()

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

            btc_1d = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe="1d")
            if not btc_1d.empty:
                btc_1d["btc_sma50_1d"] = btc_1d["close"].rolling(50, min_periods=20).mean()
                btc_1d["btc_bull_1d"] = btc_1d["close"] > btc_1d["btc_sma50_1d"]
                dataframe = merge_informative_pair(
                    dataframe, btc_1d[["date", "btc_bull_1d"]], self.timeframe, "1d", ffill=True
                )
            else:
                dataframe["btc_bull_1d_1d"] = False

            btc_4h = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe="4h")
            if not btc_4h.empty:
                btc_4h["btc_ema50_4h"] = btc_4h["close"].ewm(span=50, min_periods=20, adjust=False).mean()
                btc_4h["btc_bull_4h"] = btc_4h["close"] > btc_4h["btc_ema50_4h"]
                dataframe = merge_informative_pair(
                    dataframe, btc_4h[["date", "btc_bull_4h"]], self.timeframe, "4h", ffill=True
                )
            else:
                dataframe["btc_bull_4h_4h"] = False
        else:
            dataframe["sq_4h_4h"] = False
            dataframe["btc_bull_1d_1d"] = False
            dataframe["btc_bull_4h_4h"] = False

        sq_4h_col = "sq_4h_4h" if "sq_4h_4h" in dataframe.columns else "sq_4h"
        dataframe["dual_sq"] = dataframe["sq_1h"] & dataframe[sq_4h_col].fillna(False)

        btc_bull_col = "btc_bull_1d_1d" if "btc_bull_1d_1d" in dataframe.columns else "btc_bull_1d"
        btc_4h_col = "btc_bull_4h_4h" if "btc_bull_4h_4h" in dataframe.columns else "btc_bull_4h"
        dataframe["btc_macro_bull"] = dataframe[btc_bull_col].fillna(False) & dataframe[btc_4h_col].fillna(False)
        dataframe["btc_macro_bear"] = (~dataframe[btc_bull_col].fillna(True)) & (~dataframe[btc_4h_col].fillna(True))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long Entry: Dual Squeeze + Breakout + BTC Bull + Vol confirmation
        long_cond = (
            dataframe["dual_sq"].shift(1).fillna(False) &
            (dataframe["close"] > dataframe["kelt_upper_1h"]) &
            (dataframe["volume"] > 1.1 * dataframe["vol_sma20"]) &
            dataframe["btc_macro_bull"]
        )
        dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "refined_sq_long")

        # Short Entry:
        # Dual Squeeze + Breakdown + BTC Bear + NOT deeply oversold (rsi > 35) + Vol confirmation (> 1.2x)
        short_cond = (
            dataframe["dual_sq"].shift(1).fillna(False) &
            (dataframe["close"] < dataframe["kelt_lower_1h"]) &
            (dataframe["rsi_1h"] > 35) &
            (dataframe["volume"] > 1.2 * dataframe["vol_sma20"]) &
            dataframe["btc_macro_bear"]
        )
        dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "refined_sq_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> str | None:
        if trade.open_date_utc:
            trade_dur_hours = (current_time - trade.open_date_utc).total_seconds() / 3600.0
            # Shorts exit faster to protect against sudden short covering
            max_dur = self.max_hold_hours_short if trade.is_short else self.max_hold_hours_long
            if trade_dur_hours >= max_dur:
                return f"time_expired_{int(max_dur)}h"
            
            # For short positions: take profit at +8% quickly
            if trade.is_short and current_profit >= 0.08:
                return "short_tp_8pct"
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
