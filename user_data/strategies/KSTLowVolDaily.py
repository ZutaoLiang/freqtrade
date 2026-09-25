"""KST daily reversal, only while BTC is in a low-volatility regime (r3 round 42 candidate, scheme-C engine check).

Signal (daily candle close, fill next daily open):
  * Pring KST (pandas_ta defaults) crosses above its signal line while KST < 0 -> long;
    crosses below while KST > 0 -> short.
  * Only when BTC's 30-day realised volatility (std of daily log returns) is below its own 180-day median.
Exits, fixed at the signal candle's ATR(14, Wilder) as a fraction of the entry price:
  * stop 1.5 x ATR (custom_stoploss, anchored to the open price), take profit 3 x ATR (custom_roi),
  * time exit after 48 daily candles.
Research harness: scripts/minute_research/r3/lib3.py (same rules); this file is the freqtrade engine check.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_open


class KSTLowVolDaily(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1d"
    can_short = True
    minimal_roi = {"0": 100}
    stoploss = -0.99
    use_custom_stoploss = True
    use_custom_roi = True
    use_exit_signal = True                 # custom_exit is only called when True (freqtrade 2026.x)
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count = 240
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    sl_atr = 1.5
    tp_atr = 3.0
    max_hold_days = 48
    btc_pair = "BTC/USDT:USDT"

    def informative_pairs(self):
        return [(self.btc_pair, "1d")]

    @staticmethod
    def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
        pc = df["close"].shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        k = pta.kst(df["close"])
        df["kst"], df["kst_sig"] = k.iloc[:, 0], k.iloc[:, 1]
        df["atr"] = self._atr(df)
        btc = self.dp.get_pair_dataframe(self.btc_pair, "1d")[["date", "close"]].copy()
        rv = np.log(btc["close"]).diff().rolling(30).std()
        med = rv.rolling(180, min_periods=60).median()
        btc["lowvol"] = (rv < med).astype(int)
        df = df.drop(columns=["lowvol"], errors="ignore").merge(btc[["date", "lowvol"]], on="date", how="left")
        df["lowvol"] = df["lowvol"].fillna(0)
        return df

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        up = (df["kst"] > df["kst_sig"]) & (df["kst"].shift(1) <= df["kst_sig"].shift(1))
        dn = (df["kst"] < df["kst_sig"]) & (df["kst"].shift(1) >= df["kst_sig"].shift(1))
        on = df["lowvol"] == 1
        df.loc[up & (df["kst"] < 0) & on, ["enter_long", "enter_tag"]] = (1, "kst_up_lowvol")
        df.loc[dn & (df["kst"] > 0) & on, ["enter_short", "enter_tag"]] = (1, "kst_dn_lowvol")
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return df

    def _entry_atr_ratio(self, pair: str, trade: Trade) -> float | None:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        sig_date = trade.open_date_utc - timedelta(days=1)          # the signal candle (fill = next daily open)
        row = df.loc[df["date"] == sig_date]
        if row.empty or not np.isfinite(row["atr"].iloc[0]):
            return None
        return float(row["atr"].iloc[0]) / trade.open_rate

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs) -> float | None:
        r = self._entry_atr_ratio(pair, trade)
        if r is None:
            return None
        return stoploss_from_open(-self.sl_atr * r, current_profit, is_short=trade.is_short, leverage=trade.leverage)

    def custom_roi(self, pair: str, trade: Trade, current_time: datetime, trade_duration: int,
                   entry_tag: str | None, side: str, **kwargs) -> float | None:
        r = self._entry_atr_ratio(pair, trade)
        return None if r is None else self.tp_atr * r

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        if current_time - trade.open_date_utc >= timedelta(days=self.max_hold_days - 1):
            return "time_exit"
        return None
