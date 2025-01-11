from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import numpy as np
import pwlf
from freqtrade.strategy import DecimalParameter


def piecewise_linear_fit(data, max_error):
    """
    对数据进行自动分段线性拟合。

    :param data: 一维numpy数组，价格序列
    :param max_error: 最大允许拟合误差
    :return: 拟合后的数据（与输入数据长度相同），以及分段位置列表
    """
    x = np.arange(len(data))
    my_pwlf = pwlf.PiecewiseLinFit(x, data)

    try:
        # 自动确定最佳的分段位置，使得误差不超过 max_error
        breaks = my_pwlf.fitfast(max_error)
        fitted_y = my_pwlf.predict(x)
    except Exception as e:
        # 如果出现异常，返回原始数据
        print(f"Error in piecewise_linear_fit: {e}")
        fitted_y = data
        breaks = [0, len(data)]

    return fitted_y, breaks

class TrendLineAutoSegmentStrategy(IStrategy):
    # 策略参数
    timeframe = '5m'
    can_short: bool = True

    # 最大允许误差参数，可调整或优化
    max_error = DecimalParameter(0.01, 1.0, default=0.1, space='buy', optimize=True)

    # 最小回报率
    minimal_roi = {
        "0": 0.02
    }

    # 止损
    stoploss = -0.05

    # 启动所需的最小K线数量
    startup_candle_count: int = 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 获取收盘价数据
        close_prices = dataframe['close'].values

        # 检查数据长度
        if len(close_prices) < 2:
            return dataframe

        # 进行自动分段线性拟合
        max_err = self.max_error.value
        fitted_y, breaks = piecewise_linear_fit(close_prices, max_err)
        dataframe['trendline'] = fitted_y

        # 检查 breaks 是否有效
        if len(breaks) < 2:
            return dataframe

        # 计算趋势线的斜率
        slopes = np.full(len(close_prices), np.nan)

        x = np.arange(len(close_prices))
        my_pwlf = pwlf.PiecewiseLinFit(x, close_prices)
        try:
            my_pwlf.fit_with_breaks(breaks)
            for i in range(len(breaks) - 1):
                idx_start = int(breaks[i])
                idx_end = int(breaks[i + 1])
                if idx_end <= idx_start:
                    continue  # 跳过无效的分段
                slope = my_pwlf.slopes[i]
                slopes[idx_start:idx_end] = slope
        except Exception as e:
            print(f"Error in calculating slopes: {e}")

        dataframe['trendline_slope'] = slopes

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 多头进场信号：趋势线斜率为正，且超过其均值
        dataframe.loc[
            (
                (dataframe['trendline_slope'] > 0) &
                (dataframe['trendline_slope'] > dataframe['trendline_slope'].rolling(5).mean())
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'long_entry')

        # 空头进场信号：趋势线斜率为负，且低于其均值
        dataframe.loc[
            (
                (dataframe['trendline_slope'] < 0) &
                (dataframe['trendline_slope'] < dataframe['trendline_slope'].rolling(5).mean())
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'short_entry')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 多头出场信号：趋势线斜率变为负
        dataframe.loc[
            (dataframe['trendline_slope'] < 0),
            ['exit_long', 'exit_tag']
        ] = (1, 'long_exit')

        # 空头出场信号：趋势线斜率变为正
        dataframe.loc[
            (dataframe['trendline_slope'] > 0),
            ['exit_short', 'exit_tag']
        ] = (1, 'short_exit')

        return dataframe

