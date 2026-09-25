"""AllWeatherRegimeAdaptiveV2 -- implementation fix of AllWeatherRegimeAdaptive (review: user_data/minute_research/allweather_review/LOG.md).

Same four engines and the same thresholds as the original; only the execution is corrected:
  * dual_sq_long  : prev 1h candle in 4h+1h squeeze, 1h close > Keltner upper, volume > 1.2x SMA20, BTC macro bull
  * dual_sq_short : same squeeze, close < Keltner lower, volume > 1.5x SMA20, BTC macro neutral only
  * flushout_long : 12h return < -8%, 1h RSI(14) < 30, green 1h candle, volume > 1.5x SMA24
  * exhaust_short : R24 exactly -- three consecutive settlements >= 0.03%, short at T+5 min, hold 8h, no TP
Fixes versus the original:
  * 5m base timeframe. 1h-engine signals are computed on 1h candles and merged without forward-fill, so they fire
    only on the 5m candle that completes the hour (entry at the next hour's open, as in the 1h original), while
    exhaust_short keeps R24's T+5 min entry.
  * leverage 1: stops and take-profits are price moves (the original's 1.5x made every threshold price/1.5).
  * stops through custom_stoploss (checked intrabar in backtests), TPs and time exits through custom_exit;
    minimal_roi disabled (the original's global 14% ROI pre-empted the per-engine TPs).
  * exhaust_short has no custom stop and no TP; the -50% strategy stoploss is its disaster stop.
Funding source as in FundingExhaustionShort5m: datadir feather in backtests, in-strategy cache in dry-run/live.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.enums import RunMode
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair, stoploss_from_open


logger = logging.getLogger(__name__)


def bollinger(close: pd.Series, window: int = 20, stds: float = 2.0):
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return mid + stds * std, mid - stds * std


def keltner(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20, mult: float = 1.5):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=window, min_periods=window, adjust=False).mean()
    mid = close.ewm(span=window, min_periods=window, adjust=False).mean()
    return mid + mult * atr, mid - mult * atr


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period, adjust=False).mean()
    return (100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))).fillna(50.0)


def squeeze(df: DataFrame) -> pd.Series:
    bbu, bbl = bollinger(df["close"])
    ku, kl = keltner(df["high"], df["low"], df["close"])
    return (bbu < ku) & (bbl > kl)


class AllWeatherRegimeAdaptiveV2(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True
    minimal_roi = {"0": 100}
    stoploss = -0.50
    use_custom_stoploss = True
    use_exit_signal = True          # freqtrade 2026.x only calls custom_exit when this is set
    process_only_new_candles = True
    startup_candle_count = 1200     # 100 hours, the original 1h strategy's warm-up
    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    btc_pair = "BTC/USDT:USDT"
    min_funding_rate = 0.0003

    # engine switches (subclasses run single engines for the selection step)
    use_dual_sq_long = True
    use_dual_sq_short = True
    use_flushout_long = True
    use_exhaust_short = True

    # per-engine exits: (stop, take-profit, max hold hours); price moves, leverage 1
    exits = {
        "dual_sq_long": (0.07, 0.14, 18.0),
        "dual_sq_short": (0.07, 0.14, 18.0),
        "flushout_long": (0.06, 0.10, 12.0),
        "exhaust_short": (None, None, 8.0),
    }

    _funding: dict[str, dict[pd.Timestamp, float]] = {}
    _last_fetch: datetime | None = None
    _live: bool = False

    # ------------------------------------------------------------------ funding (as FundingExhaustionShort5m)
    def bot_start(self, **kwargs) -> None:
        self._funding = {}
        self._last_fetch = None
        self._live = self.config["runmode"] in (RunMode.DRY_RUN, RunMode.LIVE)
        if not self._live or not self.use_exhaust_short:
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
                self._store(pair, h["timestamp"], h["fundingRate"])
            seeded += 1
        logger.info(f"funding cache seeded for {seeded} pairs")
        self._fetch_recent(datetime.now().astimezone())

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        if not self._live or not self.use_exhaust_short or current_time.minute > 6:
            return
        if self._last_fetch and (current_time - self._last_fetch).total_seconds() < 50:
            return
        self._fetch_recent(current_time)

    def _store(self, pair: str, ts_ms: int, rate) -> None:
        t = pd.Timestamp(int(ts_ms), unit="ms", tz="UTC").round("h")
        self._funding.setdefault(pair, {})[t] = float(rate)

    def _fetch_recent(self, now: datetime) -> None:
        self._last_fetch = now
        ex = self.dp._exchange
        try:
            rows = ex._api.fapiPublicGetFundingRate(
                {"startTime": int((now - timedelta(hours=2)).timestamp() * 1000), "limit": 1000}
            )
        except Exception as e:
            logger.warning(f"funding refresh failed: {e}")
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
                self._store(pair, r["fundingTime"], r["fundingRate"])
                n += 1
        logger.info(f"funding refresh: {n} whitelisted settlements in the last 2h")

    def _settlements(self, pair: str) -> DataFrame:
        if self._live:
            s = pd.Series(self._funding.get(pair, {}), dtype="float64").sort_index()
            f = pd.DataFrame({"date": s.index, "fr": s.to_numpy()})
        else:
            name = pair.replace("/", "_").replace(":", "_")
            path = Path(self.config["datadir"]) / "futures" / f"{name}-1h-funding_rate.feather"
            if not path.exists():
                return pd.DataFrame(columns=["date", "fr", "fr1", "fr2"])
            f = pd.read_feather(path)[["date", "open"]].rename(columns={"open": "fr"}).sort_values("date")
        f["fr1"] = f["fr"].shift(1)
        f["fr2"] = f["fr"].shift(2)
        f["date"] = f["date"].astype("datetime64[ns, UTC]")
        return f

    # ------------------------------------------------------------------ indicators
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        inf = [(p, "1h") for p in pairs] + [(p, "4h") for p in pairs]
        return inf + [(self.btc_pair, "1d"), (self.btc_pair, "4h")]

    def _hourly_signals(self, pair: str) -> DataFrame:
        h = self.dp.get_pair_dataframe(pair=pair, timeframe="1h").copy()
        if h.empty:
            return h
        h["sq_1h"] = squeeze(h)
        h["kelt_u"], h["kelt_l"] = keltner(h["high"], h["low"], h["close"])
        h["rsi"] = rsi(h["close"])
        h["vol_sma20"] = h["volume"].rolling(20, min_periods=5).mean()
        h["vol_sma24"] = h["volume"].rolling(24, min_periods=5).mean()
        h["ret_12h"] = h["close"] / h["close"].shift(12).replace(0, np.nan) - 1

        h4 = self.dp.get_pair_dataframe(pair=pair, timeframe="4h").copy()
        if not h4.empty:
            h4["sq"] = squeeze(h4)
            h = merge_informative_pair(h, h4[["date", "sq"]], "1h", "4h", ffill=True).drop(columns=["date_4h"])
        else:
            h["sq_4h"] = False
        b1d = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe="1d").copy()
        b1d["bull"] = b1d["close"] > b1d["close"].rolling(50, min_periods=20).mean()
        h = merge_informative_pair(h, b1d[["date", "bull"]], "1h", "1d", ffill=True).drop(columns=["date_1d"])
        b4h = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe="4h").copy()
        b4h["bull"] = b4h["close"] > b4h["close"].ewm(span=50, min_periods=20, adjust=False).mean()
        h = merge_informative_pair(h, b4h[["date", "bull"]], "1h", "4h", ffill=True).drop(columns=["date_4h"])

        bull1d = h["bull_1d"].astype("boolean").fillna(False).astype(bool)
        bull4h = h["bull_4h"].astype("boolean").fillna(False).astype(bool)
        bear = (~h["bull_1d"].astype("boolean").fillna(True).astype(bool)) & (~h["bull_4h"].astype("boolean").fillna(True).astype(bool))
        macro_bull = bull1d & bull4h
        neutral = ~macro_bull & ~bear
        dual_prev = (h["sq_1h"] & h["sq_4h"].astype("boolean").fillna(False).astype(bool)).shift(1).fillna(False).astype(bool)

        h["s_dsl"] = dual_prev & (h["close"] > h["kelt_u"]) & (h["volume"] > 1.2 * h["vol_sma20"]) & macro_bull
        h["s_dss"] = dual_prev & (h["close"] < h["kelt_l"]) & (h["volume"] > 1.5 * h["vol_sma20"]) & neutral
        h["s_fl"] = (h["ret_12h"] < -0.08) & (h["rsi"] < 30) & (h["close"] > h["open"]) & (h["volume"] > 1.5 * h["vol_sma24"])
        return h[["date", "s_dsl", "s_dss", "s_fl"]].astype({"s_dsl": float, "s_dss": float, "s_fl": float})

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        h = self._hourly_signals(pair)
        if h.empty:
            for c in ("s_dsl_1h", "s_dss_1h", "s_fl_1h"):
                dataframe[c] = 0.0
        else:
            # no forward-fill: the hourly signal lands only on the 5m candle that completes the hour
            dataframe = merge_informative_pair(dataframe, h, self.timeframe, "1h", ffill=False)
        f = self._settlements(pair)
        d = dataframe[["date"]].copy()
        d["date"] = d["date"].astype("datetime64[ns, UTC]")
        m = d.merge(f, on="date", how="left")          # only the 5m candle opening exactly at settlement T
        dataframe["fr"], dataframe["fr1"], dataframe["fr2"] = m["fr"].to_numpy(), m["fr1"].to_numpy(), m["fr2"].to_numpy()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        on = lambda c: dataframe[c].fillna(0.0) > 0.5
        r = self.min_funding_rate
        exh = (dataframe["fr"] >= r) & (dataframe["fr1"] >= r) & (dataframe["fr2"] >= r)
        if self.use_dual_sq_long:
            dataframe.loc[on("s_dsl_1h"), ["enter_long", "enter_tag"]] = (1, "dual_sq_long")
        if self.use_flushout_long:      # takes precedence over dual_sq_long, as in the original
            dataframe.loc[on("s_fl_1h"), ["enter_long", "enter_tag"]] = (1, "flushout_long")
        if self.use_dual_sq_short:
            dataframe.loc[on("s_dss_1h") & ~(exh if self.use_exhaust_short else False),
                          ["enter_short", "enter_tag"]] = (1, "dual_sq_short")
        if self.use_exhaust_short:
            dataframe.loc[exh, ["enter_short", "enter_tag"]] = (1, "exhaust_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # ------------------------------------------------------------------ exits
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs) -> float | None:
        stop = self.exits.get(trade.enter_tag, (None, None, None))[0]
        if stop is None:
            return None                  # keep the -50% strategy stop (no ratcheting for exhaust_short)
        return stoploss_from_open(-stop, current_profit, is_short=trade.is_short, leverage=trade.leverage)

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        _, tp, hold = self.exits.get(trade.enter_tag, (None, None, 8.0))
        if tp is not None and current_profit >= tp:
            return f"tp_{trade.enter_tag}"
        if current_time >= trade.open_date_utc + timedelta(hours=hold):
            return f"time_{trade.enter_tag}"
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float,
                 max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        return 1.0


class AllWeatherV2_DualSqLong(AllWeatherRegimeAdaptiveV2):
    use_dual_sq_short = use_flushout_long = use_exhaust_short = False


class AllWeatherV2_DualSqShort(AllWeatherRegimeAdaptiveV2):
    use_dual_sq_long = use_flushout_long = use_exhaust_short = False


class AllWeatherV2_Flushout(AllWeatherRegimeAdaptiveV2):
    use_dual_sq_long = use_dual_sq_short = use_exhaust_short = False


class AllWeatherV2_Exhaust(AllWeatherRegimeAdaptiveV2):
    use_dual_sq_long = use_dual_sq_short = use_flushout_long = False
