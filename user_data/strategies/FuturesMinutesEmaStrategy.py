from freqtrade.strategy import IStrategy
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame

import pandas_ta as pta
from datetime import datetime
from typing import Optional


class FuturesMinutesEmaStrategy(IStrategy):
    # ROI table:
    minimal_roi = {"0": 100}

    leverage_ratio = 1

    # Stoploss:
    stoploss = -0.05 * leverage_ratio

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.144
    trailing_only_offset_is_reached = False

    can_short = True

    minutes = 5
    timeframe = f"{minutes}m"

    ema_short_period = int(240 / minutes)
    ema_mid_period = int(ema_short_period * 2)
    ema_long_period = int(ema_mid_period * 3)

    startup_candle_count = int(ema_long_period * 1.1)

    # ema_long_period_up_ratio = 1.00005

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        heikinashi = qtpylib.heikinashi(df)
        df['ha_open'] = heikinashi['open']
        df['ha_high'] = heikinashi['high']
        df['ha_low'] = heikinashi['low']
        df['ha_close'] = heikinashi['close']

        df[f"ema_short"] = pta.ema(df['ha_close'], self.ema_short_period, False)
        df[f"ema_mid"] = pta.ema(df['ha_close'], self.ema_mid_period, False)
        df[f"ema_long"] = pta.ema(df['ha_close'], self.ema_long_period, False)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 长期均线需要上升
        up_1 = (dataframe[f"ema_short"].shift(1) < dataframe[f"ema_short"]) & (dataframe[f"ema_mid"].shift(1) < dataframe[f"ema_mid"]) & (dataframe[f"ema_long"].shift(1) < dataframe[f"ema_long"])

        up_2 = (dataframe[f"ema_short"].shift(2) < dataframe[f"ema_short"].shift(1)) & (dataframe[f"ema_mid"].shift(2) < dataframe[f"ema_mid"].shift(1)) & (dataframe[f"ema_long"].shift(2) < dataframe[f"ema_long"].shift(1))

        # 短期均线大于中期均线
        up_3 = dataframe[f"ema_short"] >= dataframe[f"ema_mid"]

        # 中期均线大于长期均线
        up_4 = dataframe[f"ema_mid"] >= dataframe[f"ema_long"]

        dataframe.loc[(up_1 & up_2 & up_3 & up_4), 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        down_1 = dataframe[f"ha_close"] < dataframe[f"ema_long"]
        down_2 = dataframe[f"ema_short"] < dataframe[f"ema_long"]

        dataframe.loc[(down_1 | down_2), 'exit_long'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.leverage_ratio
