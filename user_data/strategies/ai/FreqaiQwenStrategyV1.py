import logging
from functools import reduce

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
import freqtrade.vendor.qtpylib.indicators as qtpylib_indicators
import pandas_ta as pta

from freqtrade.strategy import IStrategy
from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy.strategy_helper import stoploss_from_absolute
from freqtrade.constants import Config

logger = logging.getLogger(__name__)


class FreqaiQwenStrategyV1(IStrategy):
    """
    FreqAI Qwen Strategy V1 - 基于机器学习的做空策略
    
    特点：
    1. 大止盈设置（不主动止盈）
    2. 自定义ATR止损逻辑
    3. 优化的特征工程（市场结构、支撑阻力、K线形态、时间特征、BTC主导性）
    4. 专注做空机会（在熊市环境中表现更好）
    """

    # 设置大止盈，基本不主动止盈
    minimal_roi = {"0": 100}
    
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
    # 设置基础止损为-0.1
    stoploss = -0.1
    use_exit_signal = True
    # 这是传递给talib的最大周期（与时间框架无关）
    startup_candle_count: int = 200
    can_short = True

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
        # 自定义止损参数
        self.base_stop_loss = self.config.get("base_stop_loss", 0.18)
        self.trade_leverage = self.config.get("trade_leverage", 5)
        self.use_custom_stoploss = self.config.get("use_custom_stoploss", True)
        self.atr_stop_loss_multiplier = self.config.get("atr_stop_loss_multiplier", 2.0)
        self.atr_period = self.config.get("atr_period", 21)
        
        # 更新实际使用的止损
        self.stoploss = -float(self.base_stop_loss * self.trade_leverage)

    def feature_market_structure(self, dataframe: DataFrame) -> DataFrame:
        """市场结构特征"""
        # ADX：趋势强度（最重要）
        # ADX > 25 趋势市，ADX < 20 震荡市
        dataframe['%-f_adx'] = ta.ADX(
            dataframe['high'], dataframe['low'], dataframe['close'], 14
        ) / 100
        dataframe['%-f_adx_slope'] = dataframe['%-f_adx'].diff(6)
        
        # 高低点结构：判断是否形成更高高点/更高低点
        rolling_high = dataframe['high'].rolling(24).max()
        rolling_low = dataframe['low'].rolling(24).min()
        prev_high = dataframe['high'].rolling(24).max().shift(24)
        prev_low = dataframe['low'].rolling(24).min().shift(24)
        
        dataframe['%-f_hh'] = (rolling_high > prev_high).astype(int)  # 更高高点
        dataframe['%-f_hl'] = (rolling_low > prev_low).astype(int)    # 更高低点
        dataframe['%-f_market_structure'] = dataframe['%-f_hh'] + dataframe['%-f_hl']
        # 2=上升结构 0=下降结构 1=混乱结构
        
        # 价格位置：当前价在过去N根K线的百分位
        dataframe['%-f_price_percentile_48'] = (
            dataframe['close'].rolling(48).rank() / 48
        )
        dataframe['%-f_price_percentile_168'] = (
            dataframe['close'].rolling(168).rank() / 168
        )
        
        return dataframe

    def feature_support_resistance(self, dataframe: DataFrame) -> DataFrame:
        """支撑阻力特征"""
        # 与近期高低点的距离（归一化）
        high_24 = dataframe['high'].rolling(24).max()
        low_24 = dataframe['low'].rolling(24).min()
        high_168 = dataframe['high'].rolling(168).max()
        low_168 = dataframe['low'].rolling(168).min()
        
        dataframe['%-f_dist_to_high_24'] = (high_24 - dataframe['close']) / dataframe['close']
        dataframe['%-f_dist_to_low_24'] = (dataframe['close'] - low_24) / dataframe['close']
        dataframe['%-f_dist_to_high_168'] = (high_168 - dataframe['close']) / dataframe['close']
        dataframe['%-f_dist_to_low_168'] = (dataframe['close'] - low_168) / dataframe['close']
        
        # 当前K线在近期区间的位置（0=底部 1=顶部）
        dataframe['%-f_range_position_24'] = (
            (dataframe['close'] - low_24) / (high_24 - low_24 + 1e-9)
        )
        
        return dataframe

    def feature_candle_pattern(self, dataframe: DataFrame) -> DataFrame:
        """K线形态特征"""
        # 实体大小（相对于波动范围）
        body = abs(dataframe['close'] - dataframe['open'])
        total_range = dataframe['high'] - dataframe['low'] + 1e-9
        dataframe['%-f_body_ratio'] = body / total_range
        
        # 方向（阳线/阴线）
        dataframe['%-f_candle_dir'] = (
            (dataframe['close'] > dataframe['open']).astype(int) * 2 - 1
        )  # 1=阳线 -1=阴线
        
        # 上下影线比例
        upper_wick = dataframe['high'] - dataframe[['close','open']].max(axis=1)
        lower_wick = dataframe[['close','open']].min(axis=1) - dataframe['low']
        dataframe['%-f_upper_wick'] = upper_wick / total_range
        dataframe['%-f_lower_wick'] = lower_wick / total_range
        
        # 连续N根同向K线（惯性）
        dataframe['%-f_consecutive_up'] = (
            dataframe['%-f_candle_dir'].rolling(6).sum() / 6
        )  # 范围 -1 到 1，1表示连续6根阳线
        
        # 吞没形态（简化版）
        prev_body = body.shift(1)
        dataframe['%-f_engulfing'] = (
            (body > prev_body * 1.5) & 
            (dataframe['%-f_candle_dir'] != dataframe['%-f_candle_dir'].shift(1))
        ).astype(int) * dataframe['%-f_candle_dir']
        
        return dataframe

    def feature_time(self, dataframe: DataFrame) -> DataFrame:
        """时间特征"""
        # 小时（加密市场不同时段活跃度差异大）
        dataframe['%-f_hour_sin'] = np.sin(2 * np.pi * dataframe['date'].dt.hour / 24)
        dataframe['%-f_hour_cos'] = np.cos(2 * np.pi * dataframe['date'].dt.hour / 24)
        
        # 星期（周一开盘、周五收盘效应）
        dataframe['%-f_dow_sin'] = np.sin(2 * np.pi * dataframe['date'].dt.dayofweek / 7)
        dataframe['%-f_dow_cos'] = np.cos(2 * np.pi * dataframe['date'].dt.dayofweek / 7)
        # 用sin/cos编码而不是直接用数字，避免模型认为周日(6)和周一(0)差距很大
        
        return dataframe


    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        扩展所有技术指标特征
        """
        # 基础技术指标
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
        基础特征工程
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        
        # ATR指标用于止损
        dataframe["%-atr"] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_period)
        dataframe["%-natr"] = pta.natr(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], length=self.atr_period, talib=False, scalar=1.0)
        
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        标准特征工程 - 添加自定义特征
        """
        # 添加市场结构特征
        dataframe = self.feature_market_structure(dataframe)
        
        # 添加支撑阻力特征
        dataframe = self.feature_support_resistance(dataframe)
        
        # 添加K线形态特征
        dataframe = self.feature_candle_pattern(dataframe)
        
        # 添加时间特征
        dataframe = self.feature_time(dataframe)
        
        # 添加日期特征
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        设置目标变量 - 预测未来收益
        """
        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]
        
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-label_period)
            .rolling(label_period)
            .mean()
            / dataframe["close"]
            - 1
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 所有指标必须由feature_engineering_*()函数填充
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        入场信号 - 专注于做空机会
        """
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] > 0.01,
        ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        # 做空条件更宽松，因为退出信号质量高
        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] < -0.005,  # 降低做空阈值
        ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        退出信号 - 由于我们设置了大止盈，主要依赖模型预测
        """
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

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """
        自定义止损逻辑，参考SimpleTrendShortV1
        """
        if not self.use_custom_stoploss:
            return None
        
        if self.atr_stop_loss_multiplier <= 0:
            return None
        
        leverage = trade.leverage
        is_short = trade.is_short
        open_rate = trade.open_rate
        _current_profit = current_profit / leverage

        # 获取最后一根K线数据
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None
            
        last_candle = dataframe.iloc[-1].squeeze()
        atr = last_candle['%-atr']
        natr = last_candle['%-natr']

        if after_fill:
            filled_orders = trade.select_filled_orders()
            count_of_orders = len(filled_orders)
            if count_of_orders == 0:
                return None
            
            if is_short:
                stop_rate_atr = open_rate + (self.atr_stop_loss_multiplier * atr) 
                stop_rate_abs = open_rate * (1 + self.base_stop_loss)
                stop_rate = min(stop_rate_atr, stop_rate_abs)
            else:
                stop_rate_atr = open_rate - (self.atr_stop_loss_multiplier * atr)
                stop_rate_abs = open_rate * (1 - self.base_stop_loss)
                stop_rate = max(stop_rate_atr, stop_rate_abs)
            
            if count_of_orders == 1:
                logger.info(f'Set {trade.pair} after fill #{count_of_orders} stoploss rate to:{stop_rate:.6f}'
                            f'(stop_rate_atr:{stop_rate_atr:.6f}, stop_rate_abs:{stop_rate_abs:.6f}), '
                            f'[new_open_rate:{open_rate:.6f}](stop/open dist:{abs(stop_rate/open_rate-1):.2%}, atr:{atr:.6f}, natr:{natr:.2%})'
                            f'current_rate:{current_rate:.6f}, '
                            f'current_profit:{current_profit:.2%}(without leverage:{_current_profit:.2%}) at {current_time}')
            else:
                logger.info(f'Set {trade.pair} after fill #{count_of_orders} stoploss rate to:{stop_rate:.6f}'
                            f'(stop_rate_atr:{stop_rate_atr:.6f}, stop_rate_abs:{stop_rate_abs:.6f}), '
                            f'[new_open_rate:{open_rate:.6f}](stop/open dist:{abs(stop_rate/open_rate-1):.2%}, atr:{atr:.6f}, natr:{natr:.2%})'
                            f'current_rate:{current_rate:.6f}, '
                            f'current_profit:{current_profit:.2%}(without leverage:{_current_profit:.2%}) at {current_time}')
            return stoploss_from_absolute(stop_rate, current_rate, is_short, leverage)
        
        return None

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return self.trade_leverage