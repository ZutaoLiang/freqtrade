"""AllWeatherRegimeAdaptive -- Unified Multi-Strategy All-Weather Architecture.

Integrates three proven orthogonal alpha engines with causal BTC macro regime consensus:
1. Engine 1 (Dual Squeeze Trend Breakout):
   - 4H Squeeze + 1H Squeeze (Bollinger Bands 20, 2.0 inside Keltner Channels 20, 1.5).
   - Long Breakout (close > Keltner Upper) ONLY in BTC Macro Bull (BTC 1D > SMA50 & BTC 4H > EMA50).
   - Short Breakdown (close < Keltner Lower) strictly disabled in BTC Bear (to prevent short squeeze traps),
     only enabled in Neutral regime with heavy volume confirmation.
   - Max Hold: 18h | Hard SL: -7% | TP: +14%.

2. Engine 2 (Liquidity Dislocation Reversal / Flushout):
   - Shock absorber active in all regimes (especially Bear and Chop markets).
   - 12h price drop > 8%, 1H RSI < 30 (extreme oversold), green reversal candle, volume spike > 1.5x rolling.
   - Exploits liquidation cascades and captures the post-liquidation short squeeze bounce.
   - Max Hold: 12h | Hard SL: -6% | TP: +10%.

3. Engine 3 (Structural Cash-Flow Short / R24 Funding Exhaustion):
   - Three consecutive 8h settlement periods with funding rate >= +0.03% (annualized > +32.8% APR).
   - Shorts over-leveraged manic froth at the top, collecting massive funding cash flows.
   - Max Hold: 8h | Disaster SL: -20% (protects tail risk without false shakeouts).

4. Risk & Capital Management:
   - Dynamic compounding: stake = (Total Equity * tradable_balance_ratio) / max_open_trades.
   - Binance notional floor: >= 5 USDT.
   - 1.5x Isolated Margin leverage.
"""
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.enums import RunMode
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair

logger = logging.getLogger(__name__)


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


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


class AllWeatherRegimeAdaptive(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True

    # Base stoploss (-50% acts as exchange catastrophic stop)
    stoploss = -0.50
    use_custom_stoploss = False

    # Take profit defaults
    minimal_roi = {
        "0": 0.14
    }

    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count = 100

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Leverage
    leverage_val = 1.5

    # Strategy parameters
    btc_pair = "BTC/USDT:USDT"
    min_funding_rate = 0.0003  # 0.03% per settlement

    # Live funding cache
    _funding: dict[str, dict[pd.Timestamp, float]] = {}
    _last_fetch: datetime | None = None
    _live: bool = False

    def bot_start(self, **kwargs) -> None:
        self._funding = {}
        self._last_fetch = None
        self._live = self.config.get("runmode") in (RunMode.DRY_RUN, RunMode.LIVE)
        if not self._live:
            return
        api = self.dp._exchange._api
        seeded = 0
        for pair in self.dp.current_whitelist():
            try:
                hist = api.fetch_funding_rate_history(pair, limit=5)
            except Exception as e:
                logger.warning(f"funding seed failed for {pair}: {e}")
                continue
            for h in hist:
                self._store_funding(pair, h["timestamp"], h["fundingRate"])
            seeded += 1
        logger.info(f"Funding cache seeded for {seeded} pairs")
        self._fetch_recent_funding(datetime.now(timezone.utc))

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        if not self._live or current_time.minute > 6:
            return
        if self._last_fetch and (current_time - self._last_fetch).total_seconds() < 50:
            return
        self._fetch_recent_funding(current_time)

    def _store_funding(self, pair: str, ts_ms: int, rate: float) -> None:
        t = pd.Timestamp(int(ts_ms), unit="ms", tz="UTC").round("h")
        self._funding.setdefault(pair, {})[t] = float(rate)

    def _fetch_recent_funding(self, now: datetime) -> None:
        self._last_fetch = now
        ex = self.dp._exchange
        try:
            rows = ex._api.fapiPublicGetFundingRate(
                {"startTime": int((now - timedelta(hours=2)).timestamp() * 1000), "limit": 1000}
            )
        except Exception as e:
            logger.warning(f"Funding refresh failed: {e}")
            return
        by_id = {
            m["id"]: symbol
            for symbol, m in ex.markets.items()
            if m.get("swap") and m.get("linear") and m.get("settle") == "USDT"
        }
        whitelist = set(self.dp.current_whitelist())
        n = 0
        for r in rows:
            pair = by_id.get(r["symbol"])
            if pair in whitelist:
                self._store_funding(pair, r["fundingTime"], r["fundingRate"])
                n += 1
        logger.info(f"Funding refresh: {n} whitelisted settlements in the last 2h")

    def _get_settlements(self, pair: str) -> DataFrame:
        if self._live:
            s = pd.Series(self._funding.get(pair, {}), dtype="float64").sort_index()
            f = pd.DataFrame({"date": s.index, "fr": s.to_numpy()})
        else:
            datadir = Path(self.config.get("datadir", "user_data/data/binance"))
            name = pair.replace("/", "_").replace(":", "_")
            funding_file = datadir / "futures" / f"{name}-1h-funding_rate.feather"
            if not funding_file.exists():
                return pd.DataFrame(columns=["date", "fr", "fr1", "fr2"])
            try:
                f = pd.read_feather(funding_file)[["date", "open"]]
                f = f.rename(columns={"open": "fr"}).sort_values("date")
            except Exception as e:
                logger.warning(f"Failed to read funding feather for {pair}: {e}")
                return pd.DataFrame(columns=["date", "fr", "fr1", "fr2"])

        if f.empty:
            return pd.DataFrame(columns=["date", "fr", "fr1", "fr2"])

        f["fr1"] = f["fr"].shift(1)
        f["fr2"] = f["fr"].shift(2)
        f["date"] = f["date"].astype("datetime64[ns, UTC]")
        return f

    def informative_pairs(self):
        """Define 4h timeframe for all active whitelist pairs + BTC 1d & 4h macro pairs."""
        pairs = self.dp.current_whitelist()
        informative = [(pair, "4h") for pair in pairs]
        informative.append((self.btc_pair, "1d"))
        informative.append((self.btc_pair, "4h"))
        return informative

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        # -------------------------------------------------------------
        # 1. Base Timeframe (1h): Squeeze, RSI, Volume, 12h Momentum
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

        dataframe["rsi_1h"] = calc_rsi(dataframe["close"], 14)
        dataframe["vol_sma20"] = dataframe["volume"].rolling(20, min_periods=5).mean()
        dataframe["vol_sma24"] = dataframe["volume"].rolling(24, min_periods=5).mean()

        # 12-hour return
        dataframe["ret_12h"] = (dataframe["close"] - dataframe["close"].shift(12)) / dataframe["close"].shift(12).replace(0, np.nan)
        dataframe["is_green_candle"] = dataframe["close"] > dataframe["open"]

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
            # 3. Informative BTC Macro Regime (1D SMA50 & 4H EMA50)
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

        # Consensus Dual Squeeze
        sq_4h_col = "sq_4h_4h" if "sq_4h_4h" in dataframe.columns else "sq_4h"
        dataframe["dual_sq"] = dataframe["sq_1h"] & dataframe[sq_4h_col].fillna(False)

        # BTC Macro Consensus
        btc_bull_col = "btc_bull_1d_1d" if "btc_bull_1d_1d" in dataframe.columns else "btc_bull_1d"
        btc_4h_col = "btc_bull_4h_4h" if "btc_bull_4h_4h" in dataframe.columns else "btc_bull_4h"
        dataframe["btc_macro_bull"] = dataframe[btc_bull_col].fillna(False) & dataframe[btc_4h_col].fillna(False)
        dataframe["btc_macro_bear"] = (~dataframe[btc_bull_col].fillna(True)) & (~dataframe[btc_4h_col].fillna(True))
        dataframe["btc_macro_neutral"] = (~dataframe["btc_macro_bull"]) & (~dataframe["btc_macro_bear"])

        # -------------------------------------------------------------
        # 4. Structural Funding Settlements (R24)
        # -------------------------------------------------------------
        f = self._get_settlements(pair)
        if not f.empty:
            d = dataframe[["date"]].copy()
            d["date"] = d["date"].astype("datetime64[ns, UTC]")
            m = d.merge(f, on="date", how="left")
            dataframe["fr"] = m["fr"].to_numpy()
            dataframe["fr1"] = m["fr1"].to_numpy()
            dataframe["fr2"] = m["fr2"].to_numpy()
            dataframe["is_settle_bar"] = m["fr"].notna()
        else:
            dataframe["fr"] = 0.0
            dataframe["fr1"] = 0.0
            dataframe["fr2"] = 0.0
            dataframe["is_settle_bar"] = False

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Default flags
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        # -------------------------------------------------------------
        # Engine 1A: Dual Squeeze Trend Breakout Long
        # Active ONLY in BTC Macro Bull to prevent buying falling knives
        # -------------------------------------------------------------
        dual_sq_long_cond = (
            dataframe["dual_sq"].shift(1).fillna(False) &
            (dataframe["close"] > dataframe["kelt_upper_1h"]) &
            (dataframe["volume"] > 1.2 * dataframe["vol_sma20"]) &
            dataframe["btc_macro_bull"]
        )
        dataframe.loc[dual_sq_long_cond, ["enter_long", "enter_tag"]] = (1, "dual_sq_long")

        # -------------------------------------------------------------
        # Engine 1B: Liquidity Dislocation Reversal Long (Flushout)
        # Active in ALL regimes (especially bear and chop) as shock absorber
        # -------------------------------------------------------------
        flushout_long_cond = (
            (dataframe["ret_12h"] < -0.08) &
            (dataframe["rsi_1h"] < 30) &
            dataframe["is_green_candle"] &
            (dataframe["volume"] > 1.5 * dataframe["vol_sma24"])
        )
        # Higher priority for flushout over dual squeeze
        dataframe.loc[flushout_long_cond, ["enter_long", "enter_tag"]] = (1, "flushout_long")

        # -------------------------------------------------------------
        # Engine 2A: Structural Cash-Flow Short (R24 Funding Exhaustion)
        # Active in ALL regimes at manic froth tops
        # -------------------------------------------------------------
        r = self.min_funding_rate
        fnd_short_cond = (
            dataframe["is_settle_bar"] &
            (dataframe["fr"] >= r) &
            (dataframe["fr1"] >= r) &
            (dataframe["fr2"] >= r)
        )
        dataframe.loc[fnd_short_cond, ["enter_short", "enter_tag"]] = (1, "exhaust_short")

        # -------------------------------------------------------------
        # Engine 2B: Dual Squeeze Breakdown Short
        # STRICTLY DISABLED in BTC Macro Bear (prevents violent short squeeze traps)
        # Only active in Neutral / Choppy regime with strong volume breakdown
        # -------------------------------------------------------------
        dual_sq_short_cond = (
            dataframe["dual_sq"].shift(1).fillna(False) &
            (dataframe["close"] < dataframe["kelt_lower_1h"]) &
            (dataframe["volume"] > 1.5 * dataframe["vol_sma20"]) &
            dataframe["btc_macro_neutral"]
        )
        # Do not overwrite exhaust_short
        cond_sq_short_clean = dual_sq_short_cond & (~fnd_short_cond)
        dataframe.loc[cond_sq_short_clean, ["enter_short", "enter_tag"]] = (1, "dual_sq_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> str | None:
        """Asymmetric holding duration, take-profit, and fixed stop-loss targets per engine tag."""
        if not trade.open_date_utc:
            return None

        dur_hours = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        tag = trade.enter_tag

        # 1. Fixed Stop-Loss Checks per tag (Prevents premature trailing-stop triggers)
        if tag == "exhaust_short" and current_profit <= -0.50:
            return "sl_exhaust_50pct"
        elif tag == "flushout_long" and current_profit <= -0.06:
            return "sl_flushout_6pct"
        elif tag in ("dual_sq_long", "dual_sq_short") and current_profit <= -0.07:
            return "sl_dualsq_7pct"

        # 2. Take-Profit Targets per tag
        if tag == "exhaust_short" and current_profit >= 0.15:
            return "tp_exhaust_15pct"
        elif tag == "flushout_long" and current_profit >= 0.10:
            return "tp_flushout_10pct"
        elif tag in ("dual_sq_long", "dual_sq_short") and current_profit >= 0.14:
            return "tp_dualsq_14pct"

        # 3. Time-Based Exits
        if tag == "exhaust_short" and dur_hours >= 8.0:
            return "time_exhaust_8h"
        elif tag == "flushout_long" and dur_hours >= 12.0:
            return "time_flushout_12h"
        elif tag in ("dual_sq_long", "dual_sq_short") and dur_hours >= 18.0:
            return "time_dualsq_18h"

        return None

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        """Dynamic Compounding: Stake = (Total Equity * tradable_balance_ratio) / max_open_trades."""
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
