import logging
from functools import reduce

import talib.abstract as ta
import numpy as np
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class FreqaiMarketStructureStrategy(IStrategy):
    """
    FreqAI strategy implementing market structure features from feature_context.txt.
    Features include ADX trend strength, market structure (HH/HL), support/resistance,
    candle patterns, time features, and BTC dominance.
    """

    # Strategy parameters
    minimal_roi = {"0": 0.1, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "&-s_close": {"&-s_close": {"color": "blue"}},
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    # this is the maximum period fed to talib (timeframe independent)
    startup_candle_count: int = 200  # Increased for rolling windows up to 168
    can_short = True

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        :param dataframe: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        :param metadata: metadata of current pair
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """

        # Basic technical indicators
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["bb_lowerband-period"]

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This optional function will be called once with the dataframe of the base timeframe.
        This is the final function to be called, which means that the dataframe entering this
        function will contain all the features and columns created by all other
        freqai_feature_engineering_* functions.

        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).
        This function is a good place for any feature that should not be auto-expanded upon
        (e.g. day of the week).

        All features must be prepended with `%` to be recognized by FreqAI internals.

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        """
        
        # Import market structure features from feature_context.txt
        dataframe = self.feature_market_structure(dataframe)
        dataframe = self.feature_support_resistance(dataframe)
        dataframe = self.feature_candle_pattern(dataframe)
        dataframe = self.feature_time(dataframe)
        
        # Note: feature_btc_dominance requires BTC dataframe which needs to be handled separately
        # This would require informative pairs or external data source
        
        # Add day and hour features (already in feature_time but adding as %- prefixed)
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        
        # Add cyclic encoding for time features (as in feature_time)
        dataframe["%-hour_sin"] = np.sin(2 * np.pi * dataframe["date"].dt.hour / 24)
        dataframe["%-hour_cos"] = np.cos(2 * np.pi * dataframe["date"].dt.hour / 24)
        dataframe["%-dow_sin"] = np.sin(2 * np.pi * dataframe["date"].dt.dayofweek / 7)
        dataframe["%-dow_cos"] = np.cos(2 * np.pi * dataframe["date"].dt.dayofweek / 7)
        
        return dataframe
    
    # Market structure feature functions (adapted from feature_context.txt)
    def feature_market_structure(self, dataframe: DataFrame) -> DataFrame:
        """
        ADX trend strength and market structure features.
        """
        # ADX: trend strength (most important)
        # ADX > 25 trending market, ADX < 20 ranging market
        dataframe["f_adx"] = ta.ADX(
            dataframe["high"], dataframe["low"], dataframe["close"], 14
        ) / 100
        dataframe["f_adx_slope"] = dataframe["f_adx"].diff(6)
        
        # High/low structure: determine if forming higher high/higher low
        rolling_high = dataframe["high"].rolling(24).max()
        rolling_low = dataframe["low"].rolling(24).min()
        prev_high = dataframe["high"].rolling(24).max().shift(24)
        prev_low = dataframe["low"].rolling(24).min().shift(24)
        
        dataframe["f_hh"] = (rolling_high > prev_high).astype(int)  # higher high
        dataframe["f_hl"] = (rolling_low > prev_low).astype(int)    # higher low
        dataframe["f_market_structure"] = dataframe["f_hh"] + dataframe["f_hl"]
        # 2=rising structure 0=falling structure 1=chaotic structure
        
        # Price position: current price percentile over N candles
        dataframe["f_price_percentile_48"] = (
            dataframe["close"].rolling(48).rank() / 48
        )
        dataframe["f_price_percentile_168"] = (
            dataframe["close"].rolling(168).rank() / 168
        )
        
        # Add % prefix for FreqAI recognition
        dataframe["%-adx"] = dataframe["f_adx"]
        dataframe["%-adx_slope"] = dataframe["f_adx_slope"]
        dataframe["%-hh"] = dataframe["f_hh"]
        dataframe["%-hl"] = dataframe["f_hl"]
        dataframe["%-market_structure"] = dataframe["f_market_structure"]
        dataframe["%-price_percentile_48"] = dataframe["f_price_percentile_48"]
        dataframe["%-price_percentile_168"] = dataframe["f_price_percentile_168"]
        
        return dataframe

    def feature_support_resistance(self, dataframe: DataFrame) -> DataFrame:
        """
        Support and resistance distance features.
        """
        # Distance to recent high/low (normalized)
        high_24 = dataframe["high"].rolling(24).max()
        low_24 = dataframe["low"].rolling(24).min()
        high_168 = dataframe["high"].rolling(168).max()
        low_168 = dataframe["low"].rolling(168).min()
        
        dataframe["f_dist_to_high_24"] = (high_24 - dataframe["close"]) / dataframe["close"]
        dataframe["f_dist_to_low_24"] = (dataframe["close"] - low_24) / dataframe["close"]
        dataframe["f_dist_to_high_168"] = (high_168 - dataframe["close"]) / dataframe["close"]
        dataframe["f_dist_to_low_168"] = (dataframe["close"] - low_168) / dataframe["close"]
        
        # Current candle position in recent range (0=bottom 1=top)
        dataframe["f_range_position_24"] = (
            (dataframe["close"] - low_24) / (high_24 - low_24 + 1e-9)
        )
        
        # Add % prefix for FreqAI recognition
        dataframe["%-dist_to_high_24"] = dataframe["f_dist_to_high_24"]
        dataframe["%-dist_to_low_24"] = dataframe["f_dist_to_low_24"]
        dataframe["%-dist_to_high_168"] = dataframe["f_dist_to_high_168"]
        dataframe["%-dist_to_low_168"] = dataframe["f_dist_to_low_168"]
        dataframe["%-range_position_24"] = dataframe["f_range_position_24"]
        
        return dataframe

    def feature_candle_pattern(self, dataframe: DataFrame) -> DataFrame:
        """
        Candle pattern features.
        """
        # Body size (relative to range)
        body = abs(dataframe["close"] - dataframe["open"])
        total_range = dataframe["high"] - dataframe["low"] + 1e-9
        dataframe["f_body_ratio"] = body / total_range
        
        # Direction (bullish/bearish)
        dataframe["f_candle_dir"] = (
            (dataframe["close"] > dataframe["open"]).astype(int) * 2 - 1
        )  # 1=bullish -1=bearish
        
        # Upper/lower wick ratio
        upper_wick = dataframe["high"] - dataframe[["close","open"]].max(axis=1)
        lower_wick = dataframe[["close","open"]].min(axis=1) - dataframe["low"]
        dataframe["f_upper_wick"] = upper_wick / total_range
        dataframe["f_lower_wick"] = lower_wick / total_range
        
        # Consecutive N same-direction candles (inertia)
        dataframe["f_consecutive_up"] = (
            dataframe["f_candle_dir"].rolling(6).sum() / 6
        )  # range -1 to 1, 1 means 6 consecutive bullish candles
        
        # Engulfing pattern (simplified)
        prev_body = body.shift(1)
        dataframe["f_engulfing"] = (
            (body > prev_body * 1.5) & 
            (dataframe["f_candle_dir"] != dataframe["f_candle_dir"].shift(1))
        ).astype(int) * dataframe["f_candle_dir"]
        
        # Add % prefix for FreqAI recognition
        dataframe["%-body_ratio"] = dataframe["f_body_ratio"]
        dataframe["%-candle_dir"] = dataframe["f_candle_dir"]
        dataframe["%-upper_wick"] = dataframe["f_upper_wick"]
        dataframe["%-lower_wick"] = dataframe["f_lower_wick"]
        dataframe["%-consecutive_up"] = dataframe["f_consecutive_up"]
        dataframe["%-engulfing"] = dataframe["f_engulfing"]
        
        return dataframe

    def feature_time(self, dataframe: DataFrame) -> DataFrame:
        """
        Time-based features.
        """
        # Hour (crypto market has different activity levels at different times)
        dataframe["f_hour_sin"] = np.sin(2 * np.pi * dataframe["date"].dt.hour / 24)
        dataframe["f_hour_cos"] = np.cos(2 * np.pi * dataframe["date"].dt.hour / 24)
        
        # Day of week (Monday open, Friday close effects)
        dataframe["f_dow_sin"] = np.sin(2 * np.pi * dataframe["date"].dt.dayofweek / 7)
        dataframe["f_dow_cos"] = np.cos(2 * np.pi * dataframe["date"].dt.dayofweek / 7)
        # Use sin/cos encoding instead of direct numbers to avoid model thinking
        # Sunday(6) and Monday(0) are far apart
        
        # Note: %- prefixed versions already added in feature_engineering_standard
        
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        """
        # Use the same target as the example strategy for now
        # Can be modified based on market structure features
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
        )

        # Alternative target: directional prediction based on market structure
        # Could add more targets like:
        # dataframe["&-trend_direction"] = ...
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # All indicators must be populated by feature_engineering_*() functions

        # the model will return all labels created by user in `set_freqai_targets()`
        # (& appended targets), an indication of whether or not the prediction should be accepted,
        # the target mean/std values for each of the labels created by user in
        # `set_freqai_targets()` for each training period.

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] > 0.01,
        ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] < -0.01,
        ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [df["do_predict"] == 1, df["&-s_close"] < 0]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [df["do_predict"] == 1, df["&-s_close"] > 0]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True