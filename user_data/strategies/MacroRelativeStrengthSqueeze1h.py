"""MacroRelativeStrengthSqueeze1h -- R1002 Multi-Timeframe Resonance & Top-Down Execution.

Strategy Architecture:
1. Macro Anchor (BTC 1D & 4H Consensus):
   - Long only when BTC 1d > SMA50 and BTC 4h > EMA50.
   - Short only when BTC 1d < SMA50 and BTC 4h < EMA50.
2. Higher Timeframe (4H) Relative Strength (RS):
   - Calculate 4H return spread: ret_alt(4h) - ret_btc(4h).
   - Long requires RS > +2% (Altcoin institutional leadership).
   - Short requires RS < -2% (Altcoin institutional abandonment).
3. Intermediate Timeframe (1H) Volatility Squeeze:
   - Bollinger Bands (20, 2.0) inside Keltner Channel (20, 1.5).
   - Entry triggers on squeeze breakout (close > upper Keltner for long, close < lower Keltner for short).
4. Microstructure Order Flow Confirmation:
   - Taker Buy Volume Ratio > 60% for long; < 40% for short.
5. Exit Rules:
   - Hard Stop-Loss: -7.0%
   - Take-Profit: +14.0%
   - Time-based Exit: 12 hours (12 bars on 1h)
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


class MacroRelativeStrengthSqueeze1h(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True

    # Hard Stoploss: -7%
    stoploss = -0.07

    # Minimal ROI: +14% Take-Profit
    minimal_roi = {
        "0": 0.14
    }

    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count = 60

    # Leverage setting
    leverage_val = 1.5

    # Max holding period: 12 hours
    max_hold_hours = 12.0

    # BTC anchor
    btc_pair = "BTC/USDT:USDT"

    def informative_pairs(self):
        """Define informative pair/interval combinations to be cached."""
        pairs = self.dp.current_whitelist()
        # 1. 4h timeframe for all active whitelist pairs (for 4H RS)
        informative = [(pair, "4h") for pair in pairs]
        # 2. BTC/USDT:USDT 1d and 4h
        informative.append((self.btc_pair, "1d"))
        informative.append((self.btc_pair, "4h"))
        return informative

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        # -------------------------------------------------------------
        # 1. Base 1h Indicators: Bollinger & Keltner Squeeze
        # -------------------------------------------------------------
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

        # Taker buy flow ratio (if available, else approximate)
        if "taker_buy_base_asset_volume" in dataframe.columns and "volume" in dataframe.columns:
            dataframe["taker_ratio"] = dataframe["taker_buy_base_asset_volume"] / (dataframe["volume"] + 1e-12)
        elif "taker_buy_volume" in dataframe.columns and "volume" in dataframe.columns:
            dataframe["taker_ratio"] = dataframe["taker_buy_volume"] / (dataframe["volume"] + 1e-12)
        else:
            # Fallback if raw taker column not populated directly
            dataframe["taker_ratio"] = 0.50

        # -------------------------------------------------------------
        # 2. Informative 4H Relative Strength vs BTC
        # -------------------------------------------------------------
        if self.dp:
            inf_4h = self.dp.get_pair_dataframe(pair=pair, timeframe="4h")
            btc_4h = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe="4h")
            btc_1d = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe="1d")

            # 4H RS Calculation
            if not inf_4h.empty and not btc_4h.empty:
                inf_4h["ret_4h"] = inf_4h["close"].pct_change(1)
                btc_4h_df = btc_4h[["date", "close"]].copy().rename(columns={"close": "btc_close"})
                btc_4h_df["btc_ret_4h"] = btc_4h_df["btc_close"].pct_change(1)

                merged_4h = pd.merge_asof(inf_4h.sort_values("date"), btc_4h_df.sort_values("date"), on="date")
                merged_4h["rs_4h"] = merged_4h["ret_4h"] - merged_4h["btc_ret_4h"]
                merged_4h["rs_lead"] = merged_4h["rs_4h"] > 0.02
                merged_4h["rs_lag"] = merged_4h["rs_4h"] < -0.02

                dataframe = merge_informative_pair(
                    dataframe, merged_4h[["date", "rs_lead", "rs_lag"]], self.timeframe, "4h", ffill=True
                )
            else:
                dataframe["rs_lead_4h"] = False
                dataframe["rs_lag_4h"] = False

            # BTC 1D SMA50 & 4H EMA50
            if not btc_1d.empty:
                btc_1d["btc_sma50_1d"] = btc_1d["close"].rolling(50, min_periods=20).mean()
                btc_1d["btc_bull_1d"] = btc_1d["close"] > btc_1d["btc_sma50_1d"]
                dataframe = merge_informative_pair(
                    dataframe, btc_1d[["date", "btc_bull_1d"]], self.timeframe, "1d", ffill=True
                )
            else:
                dataframe["btc_bull_1d_1d"] = False

            if not btc_4h.empty:
                btc_4h["btc_ema50_4h"] = btc_4h["close"].ewm(span=50, min_periods=20, adjust=False).mean()
                btc_4h["btc_bull_4h"] = btc_4h["close"] > btc_4h["btc_ema50_4h"]
                dataframe = merge_informative_pair(
                    dataframe, btc_4h[["date", "btc_bull_4h"]], self.timeframe, "4h", ffill=True
                )
            else:
                dataframe["btc_bull_4h_4h"] = False
        else:
            dataframe["rs_lead_4h"] = False
            dataframe["rs_lag_4h"] = False
            dataframe["btc_bull_1d_1d"] = False
            dataframe["btc_bull_4h_4h"] = False

        # Consensus columns
        btc_bull_col = "btc_bull_1d_1d" if "btc_bull_1d_1d" in dataframe.columns else "btc_bull_1d"
        btc_4h_col = "btc_bull_4h_4h" if "btc_bull_4h_4h" in dataframe.columns else "btc_bull_4h"
        dataframe["btc_macro_bull"] = dataframe[btc_bull_col].fillna(False) & dataframe[btc_4h_col].fillna(False)
        dataframe["btc_macro_bear"] = (~dataframe[btc_bull_col].fillna(True)) & (~dataframe[btc_4h_col].fillna(True))

        rs_lead_col = "rs_lead_4h" if "rs_lead_4h" in dataframe.columns else "rs_lead"
        rs_lag_col = "rs_lag_4h" if "rs_lag_4h" in dataframe.columns else "rs_lag"
        dataframe["rs_is_lead"] = dataframe[rs_lead_col].fillna(False)
        dataframe["rs_is_lag"] = dataframe[rs_lag_col].fillna(False)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long Condition:
        # 1. Previous bar had 1h squeeze
        # 2. Current bar breaks above upper Keltner
        # 3. BTC macro is bullish
        # 4. Altcoin 4H relative strength is leading BTC (> +2%)
        # 5. Taker buy ratio > 0.60 (or taker_ratio unavailable/fallback)
        taker_long = (dataframe["taker_ratio"] > 0.60) | (dataframe["taker_ratio"] == 0.50)
        long_cond = (
            dataframe["sq_1h"].shift(1).fillna(False) &
            (dataframe["close"] > dataframe["kelt_upper_1h"]) &
            dataframe["btc_macro_bull"] &
            dataframe["rs_is_lead"] &
            taker_long
        )
        dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "rs_squeeze_long")

        # Short Condition:
        # 1. Previous bar had 1h squeeze
        # 2. Current bar breaks below lower Keltner
        # 3. BTC macro is bearish
        # 4. Altcoin 4H relative strength is lagging BTC (< -2%)
        # 5. Taker buy ratio < 0.40 (or fallback)
        taker_short = (dataframe["taker_ratio"] < 0.40) | (dataframe["taker_ratio"] == 0.50)
        short_cond = (
            dataframe["sq_1h"].shift(1).fillna(False) &
            (dataframe["close"] < dataframe["kelt_lower_1h"]) &
            dataframe["btc_macro_bear"] &
            dataframe["rs_is_lag"] &
            taker_short
        )
        dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "rs_squeeze_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> str | None:
        """Exit after maximum holding period of 12 hours."""
        if trade.open_date_utc:
            trade_dur_hours = (current_time - trade.open_date_utc).total_seconds() / 3600.0
            if trade_dur_hours >= self.max_hold_hours:
                return "time_expired_12h"
        return None

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        """Dynamic Compounding: Stake = (Total Account Equity * tradable_balance_ratio) / max_open_trades."""
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
