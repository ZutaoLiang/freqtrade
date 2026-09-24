"""Template for one research iteration. Copy to user_data/strategies/<Name>.py and edit.

Rules the template already encodes (each one cost a wrong result once):
  * signals are computed on closed candles only; entries fill at the next candle open;
  * `use_exit_signal = True` - freqtrade 2026.x only calls custom_exit when it is True;
  * informative (higher-timeframe) data goes through merge_informative_pair, which shifts it
    so a 15m value is only visible after that 15m candle has closed - never merge by hand;
  * the engine `stoploss` is a floor; a close-based stop lives in custom_exit (1m wicks on alts
    scalp intrabar stops on trades that finish positive);
  * leverage is fixed at 1 unless the iteration is explicitly about leverage.

Indicators: use any library that is installed (see reference/indicators.md). The helpers below
fall back from TA-Lib to pandas_ta to plain pandas so the file runs on hosts without TA-Lib.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair

try:
    import talib.abstract as ta          # present in the official freqtrade docker image
except ImportError:                      # pragma: no cover
    ta = None
try:
    import pandas_ta as pta
except ImportError:                      # pragma: no cover
    pta = None


def rsi(close: pd.Series, n: int) -> pd.Series:
    if ta is not None:
        return pd.Series(ta.RSI(close, timeperiod=n), index=close.index)
    if pta is not None:
        return pta.rsi(close, length=n)
    d = close.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / dn)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n).mean()


class TemplateIndicatorStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1m"
    informative_tf = "15m"
    can_short = True
    minimal_roi = {"0": 100}              # exits are decided below, not by ROI
    stoploss = -0.99                      # floor only; real stop is close-based in custom_exit
    use_exit_signal = True
    startup_candle_count = 400
    process_only_new_candles = True
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    # --- iteration parameters (pre-register them in the iteration log before running) ---
    ema_fast, ema_slow, rsi_len = 50, 200, 14
    rsi_long, rsi_short = 30, 70
    stop_atr, take_atr = 2.0, 3.0
    max_hold_minutes = 240

    def informative_pairs(self):
        return [(p, self.informative_tf) for p in self.dp.current_whitelist()]

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df["rsi"] = rsi(df["close"], self.rsi_len)
        df["atr"] = atr(df, 14)
        inf = self.dp.get_pair_dataframe(metadata["pair"], self.informative_tf)
        inf["ema_fast"] = inf["close"].ewm(span=self.ema_fast, adjust=False).mean()
        inf["ema_slow"] = inf["close"].ewm(span=self.ema_slow, adjust=False).mean()
        df = merge_informative_pair(df, inf[["date", "ema_fast", "ema_slow"]], self.timeframe, self.informative_tf, ffill=True)
        return df

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        up = df[f"ema_fast_{self.informative_tf}"] > df[f"ema_slow_{self.informative_tf}"]
        df["enter_long"] = (up & (df["rsi"] < self.rsi_long)).astype("int8")
        df["enter_short"] = (~up & (df["rsi"] > self.rsi_short)).astype("int8")
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df["exit_long"] = 0
        df["exit_short"] = 0
        return df

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last = df.iloc[-1]
        a = float(last["atr"])            # current ATR; never slice the whole frame per call (O(n^2) on 1m)
        move = (last["close"] - trade.open_rate) * (-1 if trade.is_short else 1)
        if move <= -self.stop_atr * a:
            return "close_stop"
        if move >= self.take_atr * a:
            return "take_profit"
        if current_time - trade.open_date_utc >= timedelta(minutes=self.max_hold_minutes - 1):
            return "time_exit"
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float,
                 max_leverage: float, entry_tag, side: str, **kwargs) -> float:
        return 1.0
