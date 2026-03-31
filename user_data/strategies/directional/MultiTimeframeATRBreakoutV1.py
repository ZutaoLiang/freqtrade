from pandas import DataFrame
import pandas_ta as pta
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

from freqtrade.persistence.trade_model import Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_helper import stoploss_from_absolute, stoploss_from_open
from freqtrade.constants import Config

import logging
logger = logging.getLogger(__name__)


class MultiTimeframeATRBreakoutV1(IStrategy):
    
    timeframe = '3m'
    minimal_roi = {"0": 100}
    can_short = True
    process_only_new_candles = True
    
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
        self.informative_timeframe = self.get_config("informative_timeframe", '1h')
        
        self.base_stop_loss = self.get_config("base_stop_loss", 0.05)
        self.trade_leverage = self.get_config("trade_leverage", 5)
        self.stoploss = - float(self.base_stop_loss * self.trade_leverage)

        self.trailing_stop = self.get_config("trailing_stop", True)
        if not self.trailing_stop:
            self.custom_trailing_stop = self.get_config("custom_trailing_stop", False)
        else:
            self.custom_trailing_stop = False
            self.trailing_stop_positive = self.get_config("base_trailing_stop", 0.03) * self.trade_leverage
            self.trailing_stop_positive_offset = self.get_config("base_trailing_stop_offset", 0.05) * self.trade_leverage
            self.trailing_only_offset_is_reached = self.get_config("trailing_only_offset_is_reached", True)
        
        self.use_custom_stoploss = self.get_config("use_custom_stoploss", True)
        
        # ATR相关参数
        self.atr_period = self.get_config("atr_period", 21)
        self.atr_multiplier_entry = self.get_config("atr_multiplier_entry", 4.0)
        self.atr_multiplier_entry_long = self.get_config("atr_multiplier_entry_long", None)  # 多头专用进场倍数，None则使用通用值
        self.atr_multiplier_entry_short = self.get_config("atr_multiplier_entry_short", None)  # 空头专用进场倍数，None则使用通用值
        self.atr_multiplier_stop = self.get_config("atr_multiplier_stop", 3.0)
        
        self.atr_multiplier_take_profit = self.get_config("atr_multiplier_take_profit", 3.0)
        
        # 趋势过滤参数
        self.trend_ema_period = self.get_config("trend_ema_period", 50)  # EMA周期
        self.trend_ema_timeframe = self.get_config("trend_ema_timeframe", '4h')  # 趋势判断时间框架
        self.use_trend_filter = self.get_config("use_trend_filter", True)  # 是否启用趋势过滤
        self.trend_ema_consecutive = self.get_config("trend_ema_consecutive", 3)  # EMA连续上涨/下跌次数要求
        
        # ATR百分位参数 - 用于判断ATR是否处于相对较低水平
        self.atr_percentile_period = self.get_config("atr_percentile_period", 48)  # 计算ATR百分位的回看周期
        self.atr_percentile_threshold = self.get_config("atr_percentile_threshold", 0.6)  # ATR百分位阈值，低于此值才允许进场
        
        # 成交量参数
        self.volume_short_period = self.get_config("volume_short_period", 5)
        self.volume_long_period = self.get_config("volume_long_period", 20)
        self.volume_multiplier = self.get_config("volume_multiplier", 1.2)
        
        self.long_time_hours = self.get_config("long_time_hours", 12)
        self.long_time_tp_pct = self.get_config("long_time_tp_pct", 0.01)

        self.long_time_hours_stop = self.get_config("long_time_hours_stop", 18)
        self.long_time_stop_pct = self.get_config("long_time_stop_pct", -0.02)
        
        self.startup_candle_count = int(max(self.atr_period, self.volume_long_period, self.atr_percentile_period) * 12)  # 1h转5m需要12倍
        
    def get_config(self, key: str, default):
        return self.config.get(key, default)
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            # 计算主时间框架的ATR和成交量指标
            dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_period)
            
            # 计算成交量指标
            dataframe['volume_mean_short'] = dataframe['volume'].rolling(window=self.volume_short_period).mean()
            dataframe['volume_mean_long'] = dataframe['volume'].rolling(window=self.volume_long_period).mean()
            
            # 合并参考时间框架数据
            dataframe = self.merge_informative(dataframe, metadata)
            
            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_indicators: {e}")
            return dataframe
    
    def informative_pairs(self):
        """
        定义需要获取的额外时间框架数据
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        
        # 如果启用趋势过滤，添加趋势时间框架
        if self.use_trend_filter and self.trend_ema_timeframe != self.informative_timeframe:
            for pair in pairs:
                informative_pairs.append((pair, self.trend_ema_timeframe))
        
        return informative_pairs
    
    def populate_indicators_informative(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算参考时间框架(1h)的指标
        """
        try:
            # 计算参考时间框架的ATR
            dataframe['atr_informative'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_period)
            
            # 计算ATR百分位 - 用于判断ATR是否处于相对较低水平
            # 使用rolling计算过去N根K线的ATR百分位
            dataframe['atr_percentile'] = dataframe['atr_informative'].rolling(
                window=self.atr_percentile_period, min_periods=1
            ).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5,
                raw=False
            )
            
            # 获取上一根K线的收盘价
            dataframe['prev_close'] = dataframe['close'].shift(1)
            
            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_indicators_informative: {e}")
            return dataframe
    
    def merge_informative(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        将参考时间框架的数据合并到主时间框架
        """
        try:
            # 获取参考时间框架的数据
            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
            informative = self.populate_indicators_informative(informative, metadata)
            
            # 只保留需要的列
            informative = informative[['date', 'atr_informative', 'prev_close', 'atr_percentile']].copy()
            
            # 重命名列以避免冲突
            informative.columns = [f'{col}_{self.informative_timeframe}' if col != 'date' else col for col in informative.columns]
            
            # 合并数据 - 使用左连接并按时间排序
            dataframe = dataframe.merge(informative, on='date', how='left', suffixes=('', f'_{self.informative_timeframe}'))
            
            # 使用前向填充来将1h数据传播到所有5m K线
            dataframe[f'atr_informative_{self.informative_timeframe}'] = dataframe[f'atr_informative_{self.informative_timeframe}'].ffill()
            dataframe[f'prev_close_{self.informative_timeframe}'] = dataframe[f'prev_close_{self.informative_timeframe}'].ffill()
            dataframe[f'atr_percentile_{self.informative_timeframe}'] = dataframe[f'atr_percentile_{self.informative_timeframe}'].ffill()
            
            # 如果启用趋势过滤，合并趋势时间框架的EMA数据
            if self.use_trend_filter:
                dataframe = self.merge_trend_ema(dataframe, metadata)
            
            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::merge_informative: {e}")
            return dataframe
    
    def merge_trend_ema(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        合并趋势EMA数据到主时间框架
        """
        try:
            # 获取趋势时间框架的数据
            trend_tf = self.trend_ema_timeframe
            trend_dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=trend_tf)
            
            # 计算EMA
            ema_col = f'ema_{self.trend_ema_period}'
            trend_dataframe[ema_col] = pta.ema(trend_dataframe['close'], length=self.trend_ema_period)
            
            # 判断EMA方向：当前EMA > 上一根EMA为上涨，否则为下跌
            trend_dataframe['ema_rising'] = (trend_dataframe[ema_col] > trend_dataframe[ema_col].shift(1)).astype(int)
            trend_dataframe['ema_falling'] = (trend_dataframe[ema_col] < trend_dataframe[ema_col].shift(1)).astype(int)
            
            # 计算EMA连续上涨/下跌次数
            trend_dataframe['ema_consecutive_rising'] = trend_dataframe['ema_rising'].astype(int).groupby(
                (trend_dataframe['ema_rising'] != trend_dataframe['ema_rising'].shift()).cumsum()
            ).cumsum()
            
            trend_dataframe['ema_consecutive_falling'] = trend_dataframe['ema_falling'].astype(int).groupby(
                (trend_dataframe['ema_falling'] != trend_dataframe['ema_falling'].shift()).cumsum()
            ).cumsum()
            
            # 判断趋势方向：价格在EMA上方为上升趋势，下方为下降趋势
            trend_dataframe['trend_up'] = (trend_dataframe['close'] > trend_dataframe[ema_col]).astype(int)
            
            # 判断是否满足连续上涨/下跌要求
            trend_dataframe['ema_trend_up_confirmed'] = (
                (trend_dataframe['trend_up'] == 1) & 
                (trend_dataframe['ema_consecutive_rising'] >= self.trend_ema_consecutive)
            ).astype(int)
            
            trend_dataframe['ema_trend_down_confirmed'] = (
                (trend_dataframe['trend_up'] == 0) & 
                (trend_dataframe['ema_consecutive_falling'] >= self.trend_ema_consecutive)
            ).astype(int)
            
            # 只保留需要的列
            trend_data = trend_dataframe[[
                'date', 
                ema_col, 
                'trend_up',
                'ema_consecutive_rising',
                'ema_consecutive_falling',
                'ema_trend_up_confirmed',
                'ema_trend_down_confirmed'
            ]].copy()
            
            # 重命名列
            trend_data.columns = [f'{col}_{trend_tf}' if col != 'date' else col for col in trend_data.columns]
            
            # 合并数据
            dataframe = dataframe.merge(trend_data, on='date', how='left')
            
            # 使用前向填充
            dataframe[f'{ema_col}_{trend_tf}'] = dataframe[f'{ema_col}_{trend_tf}'].ffill()
            dataframe[f'trend_up_{trend_tf}'] = dataframe[f'trend_up_{trend_tf}'].ffill()
            dataframe[f'ema_consecutive_rising_{trend_tf}'] = dataframe[f'ema_consecutive_rising_{trend_tf}'].ffill()
            dataframe[f'ema_consecutive_falling_{trend_tf}'] = dataframe[f'ema_consecutive_falling_{trend_tf}'].ffill()
            dataframe[f'ema_trend_up_confirmed_{trend_tf}'] = dataframe[f'ema_trend_up_confirmed_{trend_tf}'].ffill()
            dataframe[f'ema_trend_down_confirmed_{trend_tf}'] = dataframe[f'ema_trend_down_confirmed_{trend_tf}'].ffill()
            
            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::merge_trend_ema: {e}")
            return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        if dataframe.empty:
            return dataframe
        
        try:
            # 确定多空专用的进场倍数
            atr_multi_long = self.atr_multiplier_entry_long if self.atr_multiplier_entry_long is not None else self.atr_multiplier_entry
            atr_multi_short = self.atr_multiplier_entry_short if self.atr_multiplier_entry_short is not None else self.atr_multiplier_entry
            
            # 构建基础条件
            volume_condition = dataframe['volume_mean_short'] > self.volume_multiplier * dataframe['volume_mean_long']
            atr_percentile_condition = dataframe[f'atr_percentile_{self.informative_timeframe}'] < self.atr_percentile_threshold
            
            # 多头进场条件：
            # 1. 当前5m的high > (上一根1h的close + 多头专用ATR倍数 * 1h的ATR)
            # 2. 成交量放大
            # 3. ATR百分位较低（避免高波动）
            # 4. 如果启用趋势过滤：只在上升趋势中做多（价格在EMA上方）
            
            long_conditions = (
                (dataframe['high'] > (dataframe[f'prev_close_{self.informative_timeframe}'] + 
                                      atr_multi_long * dataframe[f'atr_informative_{self.informative_timeframe}'])) &
                volume_condition &
                atr_percentile_condition
            )
            
            # 添加趋势过滤条件（需要价格在EMA上方 且 EMA连续上涨N次）
            if self.use_trend_filter:
                long_conditions = long_conditions & (dataframe[f'ema_trend_up_confirmed_{self.trend_ema_timeframe}'] == 1)
            
            dataframe.loc[long_conditions, 'enter_long'] = 1
            
            # 空头进场条件（不对称处理）：
            # 1. 当前5m的low < (上一根1h的close - 空头专用ATR倍数 * 1h的ATR)
            # 2. 成交量放大
            # 3. ATR百分位较低（避免高波动）
            # 4. 如果启用趋势过滤：只在下降趋势中做空（价格在EMA下方 且 EMA连续下跌N次）
            
            short_conditions = (
                (dataframe['low'] < (dataframe[f'prev_close_{self.informative_timeframe}'] - 
                                      atr_multi_short * dataframe[f'atr_informative_{self.informative_timeframe}'])) &
                volume_condition &
                atr_percentile_condition
            )
            
            # 添加趋势过滤条件（需要价格在EMA下方 且 EMA连续下跌N次）
            if self.use_trend_filter:
                short_conditions = short_conditions & (dataframe[f'ema_trend_down_confirmed_{self.trend_ema_timeframe}'] == 1)
            
            dataframe.loc[short_conditions, 'enter_short'] = 1
            
            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_entry_trend: {e}")
            return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        if dataframe.empty:
            return dataframe
        
        try:
            # 出场由custom_stoploss处理，这里不设置固定的出场信号
            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_exit_trend: {e}")
            return dataframe
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        if not self.use_custom_stoploss:
            return None
        
        leverage = trade.leverage
        is_short = trade.is_short
        open_rate = trade.open_rate
        _current_profit = current_profit / leverage

        last_candle = self.get_last_candle(trade)
        atr = last_candle['atr']
        
        if after_fill:
            # 成交后使用3倍ATR作为跟踪止损
            filled_orders = trade.select_filled_orders()
            count_of_orders = len(filled_orders)
            if count_of_orders == 0:
                return None
            
            if is_short:
                # 空头：止损价 = 开仓价 + 3倍ATR
                stop_rate_atr = open_rate + (self.atr_multiplier_stop * atr)
                stop_rate_abs = open_rate * (1 + self.base_stop_loss)
                stop_rate = min(stop_rate_atr, stop_rate_abs)
            else:
                # 多头：止损价 = 开仓价 - 3倍ATR
                stop_rate_atr = open_rate - (self.atr_multiplier_stop * atr)
                stop_rate_abs = open_rate * (1 - self.base_stop_loss)
                stop_rate = max(stop_rate_atr, stop_rate_abs)
            
            if count_of_orders == 1:
                logger.info(f'Set {trade.pair} after fill #{count_of_orders} stoploss rate to:{stop_rate:.6f}'
                            f'(stop_rate_atr:{stop_rate_atr:.6f}, stop_rate_abs:{stop_rate_abs:.6f}), '
                            f'[open_rate:{open_rate:.6f}](stop/open dist:{abs(stop_rate/open_rate-1):.2%}, atr:{atr:.6f}) '
                            f'current_rate:{current_rate:.6f}, '
                            f'current_profit:{current_profit:.2%}(without leverage:{_current_profit:.2%}) at {current_time}')
            else:
                logger.info(f'Update {trade.pair} after fill #{count_of_orders} stoploss rate to:{stop_rate:.6f}'
                            f'(stop_rate_atr:{stop_rate_atr:.6f}, stop_rate_abs:{stop_rate_abs:.6f}), '
                            f'[open_rate:{open_rate:.6f}](stop/open dist:{abs(stop_rate/open_rate-1):.2%}, atr:{atr:.6f}) '
                            f'current_rate:{current_rate:.6f}, '
                            f'current_profit:{current_profit:.2%}(without leverage:{_current_profit:.2%}) at {current_time}')
            
            return stoploss_from_absolute(stop_rate, current_rate, is_short, leverage)
        
        # 使用跟踪止损
        if self.custom_trailing_stop:
            if _current_profit > self.get_config("base_trailing_stop_offset", 0.05):
                return self.get_config("base_trailing_stop", 0.03) * leverage
            
    def get_last_candle(self, trade: Trade):
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        return last_candle
    
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Tuple[bool, Optional[str]]:
        """
        自定义出场逻辑
        增加5分钟时间框架的3倍ATR止盈逻辑
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # 获取5分钟时间框架的ATR
        atr = last_candle['atr']
        
        # 计算止盈价格
        leverage = trade.leverage
        is_short = trade.is_short
        open_rate = trade.open_rate
        _current_profit = current_profit / leverage
        
        # ATR止盈目标
        take_profit_distance = self.atr_multiplier_take_profit * atr
        
        if is_short:
            # 空头止盈：当前价格 <= 开仓价 - ATR止盈距离
            take_profit_rate = open_rate - take_profit_distance
            if current_rate <= take_profit_rate:
                exit_reason = f'short_tp'
                logger.info(f'{pair} Short trade exit triggered: current_rate={current_rate:.6f} <= '
                           f'take_profit_rate={take_profit_rate:.6f} (open_rate={open_rate:.6f}, '
                           f'atr_5m={atr:.6f}, {self.atr_multiplier_take_profit}xATR={take_profit_distance:.6f}, '
                           f'profit={current_profit:.2%}) at {current_time}')
                return exit_reason
        else:
            # 多头止盈：当前价格 >= 开仓价 + ATR止盈距离
            take_profit_rate = open_rate + take_profit_distance
            if current_rate >= take_profit_rate:
                exit_reason = f'long_tp'
                logger.info(f'{pair} Long trade exit triggered: current_rate={current_rate:.6f} >= '
                           f'take_profit_rate={take_profit_rate:.6f} (open_rate={open_rate:.6f}, '
                           f'atr_5m={atr:.6f}, {self.atr_multiplier_take_profit}xATR={take_profit_distance:.6f}, '
                           f'profit={current_profit:.2%}) at {current_time}')
                return exit_reason
        
        open_hours = round((current_time - trade.open_date_utc).total_seconds() / 3600, 1)
        if open_hours >= self.long_time_hours and _current_profit > self.long_time_tp_pct:
            return 'long_time_tp'
        
        if open_hours >= self.long_time_hours_stop and _current_profit > self.long_time_stop_pct:
            return 'long_time_stop'
        
        # 不满足止盈条件，继续持有
        return None