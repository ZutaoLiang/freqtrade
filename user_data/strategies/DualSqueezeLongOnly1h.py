"""DualSqueezeBtcTrend1h -- Scheme B: Dual Squeeze (4h + 1h) with BTC Macro Trend Consensus.

Strategy Logic:
- Long Entry:
  1. 4h Squeeze: Bollinger Bands (20, 2.0) inside Keltner Channel (20, 1.5).
  2. 1h Squeeze: Bollinger Bands (20, 2.0) inside Keltner Channel (20, 1.5).
  3. Setup: Previous 1h candle was in Dual Squeeze state.
  4. Breakout: Current 1h close breaks above 1h Keltner Upper (1.5).
  5. BTC Trend Gate: BTC 1d close > BTC 1d SMA50 AND BTC 4h close > BTC 4h EMA50.
- Short Entry:
  1. Setup: Previous 1h candle was in Dual Squeeze state.
  2. Breakdown: Current 1h close breaks below 1h Keltner Lower (1.5).
  3. BTC Trend Gate: BTC 1d close < BTC 1d SMA50 AND BTC 4h close < BTC 4h EMA50.
- Risk & Exit:
  - Hard Stop-Loss: -7.0%
  - Take-Profit: +14.0%
  - Time-based Exit: Max hold 18 hours (18 candles on 1h)
- Staking & Compounding:
  - Dynamic compounding stake = Total Equity / max_open_trades (default 10 slots).
  - Enforces exchange minimum notional floor (>= 5 USDT).
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


class DualSqueezeLongOnly1h(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    # Hard Stoploss: -7%
    stoploss = -0.07

    # Minimal ROI: +14% Take-Profit
    minimal_roi = {
        "0": 0.14
    }

    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count = 60

    # Leverage setting (1.5x - 2.0x recommended for 10U margin positions)
    leverage_val = 1.5

    # Max holding period: 18 hours
    max_hold_hours = 18.0

    # BTC macro pair
    btc_pair = "BTC/USDT:USDT"

    def informative_pairs(self):
        """Define additional, informative pair/interval combinations to be cached."""
        pairs = self.dp.current_whitelist()
        # 1. 4h timeframe for all active whitelist pairs
        informative = [(pair, "4h") for pair in pairs]
        # 2. BTC/USDT:USDT 1d and 4h for macro trend consensus
        informative.append((self.btc_pair, "1d"))
        informative.append((self.btc_pair, "4h"))
        return informative

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        # -------------------------------------------------------------
        # 1. Base Timeframe (1h): Bollinger Bands & Keltner Channel
        # -------------------------------------------------------------
        _, bb_u_1h, bb_l_1h = calc_bollinger_bands(dataframe["close"], window=20, stds=2.0)
        _, kelt_u_1h, kelt_l_1h = calc_keltner_channel(
            dataframe["high"], dataframe["low"], dataframe["close"], window=20, mult=1.5
        )
        dataframe["bb_upper_1h"] = bb_u_1h
        dataframe["bb_lower_1h"] = bb_l_1h
        dataframe["kelt_upper_1h"] = kelt_u_1h
        dataframe["kelt_lower_1h"] = kelt_l_1h

        # 1h Squeeze condition
        dataframe["sq_1h"] = (dataframe["bb_upper_1h"] < dataframe["kelt_upper_1h"]) & \
                             (dataframe["bb_lower_1h"] > dataframe["kelt_lower_1h"])

        # -------------------------------------------------------------
        # 2. Informative Timeframe (4h): 4h Squeeze
        # -------------------------------------------------------------
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

            # ---------------------------------------------------------
            # 3. Informative BTC Macro Trend (1d SMA50 and 4h EMA50)
            # ---------------------------------------------------------
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

        # Dual squeeze active when BOTH 4h and 1h squeezes are active
        sq_4h_col = "sq_4h_4h" if "sq_4h_4h" in dataframe.columns else "sq_4h"
        dataframe["dual_sq"] = dataframe["sq_1h"] & dataframe[sq_4h_col].fillna(False)

        # BTC consensus
        btc_bull_col = "btc_bull_1d_1d" if "btc_bull_1d_1d" in dataframe.columns else "btc_bull_1d"
        btc_4h_col = "btc_bull_4h_4h" if "btc_bull_4h_4h" in dataframe.columns else "btc_bull_4h"
        dataframe["btc_macro_bull"] = dataframe[btc_bull_col].fillna(False) & dataframe[btc_4h_col].fillna(False)
        dataframe["btc_macro_bear"] = (~dataframe[btc_bull_col].fillna(True)) & (~dataframe[btc_4h_col].fillna(True))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long Entry:
        # Previous bar had dual squeeze, current close breaks above Keltner upper, BTC macro is bullish
        long_cond = (
            dataframe["dual_sq"].shift(1).fillna(False) &
            (dataframe["close"] > dataframe["kelt_upper_1h"]) &
            dataframe["btc_macro_bull"]
        )
        dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "dual_sq_long")

        # Short Entry:
        # Previous bar had dual squeeze, current close breaks below Keltner lower, BTC macro is bearish
        short_cond = (
            dataframe["dual_sq"].shift(1).fillna(False) &
            (dataframe["close"] < dataframe["kelt_lower_1h"]) &
            dataframe["btc_macro_bear"]
        )
        dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "dual_sq_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> str | None:
        """Exit after maximum holding period of 18 hours."""
        if trade.open_date_utc:
            trade_dur_hours = (current_time - trade.open_date_utc).total_seconds() / 3600.0
            if trade_dur_hours >= self.max_hold_hours:
                return "time_expired_18h"
        return None

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        """Dynamic Compounding: Stake = (Total Account Equity * tradable_balance_ratio) / max_open_trades.
        
        Guarantees that stake scales up with profits and down with drawdowns, while
        strictly adhering to Binance minimum notional limits (>= 5 USDT).
        """
        try:
            total_equity = self.wallets.get_total_stake_amount()
        except Exception:
            total_equity = self.config.get("dry_run_wallet", 100.0)

        max_slots = self.config.get("max_open_trades", 10)
        tradable_ratio = self.config.get("tradable_balance_ratio", 0.99)

        target_stake = (total_equity * tradable_ratio) / max_slots

        # Ensure stake meets Binance minimum order notional floor (typically 5 USDT)
        min_allowed = max(min_stake or 5.0, 5.0)
        stake = max(target_stake, min_allowed)

        return min(stake, max_stake)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        """Set fixed leverage for futures."""
        return min(self.leverage_val, max_leverage)
