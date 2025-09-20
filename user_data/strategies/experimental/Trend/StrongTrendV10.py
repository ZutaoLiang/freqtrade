import numpy as np
import pandas_ta as pta
import pandas as pd
# pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from pandas import DataFrame

from scipy.signal import savgol_filter

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.constants import Config
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, informative
from freqtrade.persistence import Order, Trade
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
import logging
logger = logging.getLogger(__name__)


class StrongTrendV10(IStrategy):
    timeframe = '15m'
    
    minimal_roi = {"0": 100}

    trailing_stop = False
    use_custom_stoploss = False

    can_short = True
    process_only_new_candles = True
    position_adjustment_enable = True
    
    dynamic_entry_by_signal_score = False

    MAX_PROFIT_ABS = 'max_profit_abs'
    HIGH_PROFIT = 'high_profit'
    
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
        self.use_ha_candles = self.get_config("use_ha_candles", True)
        self.base_stop_loss = self.get_config("base_stop_loss", 0.07)
        self.trade_leverage = self.get_config("trade_leverage", 3)
        self.stoploss = - float(self.base_stop_loss * self.trade_leverage)
        
        self.lookback_period = self.get_config("lookback_period", 12)

        self.ema_short_len = IntParameter(5, 100, default=self.lookback_period, space='buy')
        self.ema_mid_len = IntParameter(5, 100, default=self.lookback_period * 3, space='buy')
        self.ema_long_len = IntParameter(5, 100, default=self.lookback_period * 6, space='buy')
        self.ema_week_len = IntParameter(5, 100, default=self.lookback_period * 4 * 3, space='buy')
        
        self.close_ma_length = self.get_config("close_ma_length", 5)
        self.close_smooth_window = self.get_config("close_smooth_window", 15)
        self.close_smooth_polyorder = self.get_config("close_smooth_poly_order", 1)
        
        self.extreme_distance_threshold = self.get_config("extreme_distance_threshold", 5)
        self.extreme_ratio_threshold = self.get_config("extreme_ratio_threshold", 0.01)
        
        self.window_length = IntParameter(10, 100, default=self.lookback_period, space='buy')
        self.startup_candle_count = int(max(self.window_length.value, self.ema_week_len.value) * 1.2)
        
        self.atr_period = self.get_config("atr_period", 21)
        
        self.volume_short = self.get_config("volume_short", 2)
        self.volume_mid = self.get_config("volume_mid", 20)
        self.volume_ratio = self.get_config("volume_ratio", 1.25)
        
        self.trend_length = self.get_config("trend_length", 3)
        
        self.custom_stake_lookback_trades = self.get_config("custom_stake_lookback_trades", 3)
        self.custom_stake_profit_threshold = self.get_config("custom_stake_profit_threshold", -0.01)
        self.custom_stake_ratio_when_low_closed_profit = self.get_config("custom_stake_ratio_when_low_closed_profit", 0.5)
        
        self.addition_stake_ratio = self.get_config("addition_stake_ratio", 0.8)
        self.addition_min_profit = self.get_config("addition_min_profit", 0.08)
        self.addition_min_profit_step = self.get_config("addition_min_profit_step", 0.025)
        self.addition_profit_step = self.get_config("addition_profit_step", 0.04)
        
        self.reduction_stake_ratio = self.get_config("reduction_stake_ratio", 0.5)
        self.reduction_stoploss_ratio = self.get_config("reduction_stoploss_ratio", 0.5)
        
        activation_list = str(self.get_config("profit_drawdown_activation", "1")).split(",")
        self.profit_drawdown_activation = [float(s) for s in activation_list]
        drawdown_ratio_list = str(self.get_config("profit_drawdown_ratio", "0.4")).split(",")
        self.profit_drawdown_ratio = [float(s) for s in drawdown_ratio_list]
        
        self.fee = self.get_config("fee", 0.0005)
        self.long_time_low_profit_hours = self.get_config("long_time_low_profit_hours", 3)
        self.long_time_low_profit_max = self.get_config("long_time_low_profit_max", 0.05)
        self.long_time_low_profit_lower_bound = self.get_config("long_time_low_profit_lower_bound", 0.005)
        self.long_time_low_profit_higher_max_ratio = self.get_config("long_time_low_profit_higher_max_ratio", 0.5)
        self.long_time_hours = self.get_config("long_time_hours", 18)
        self.long_time_hours_min_profit = self.get_config("long_time_hours_min_profit", 0.005)
        self.long_time_hours_max_profit = self.get_config("long_time_hours_max_profit", 0.015)
        
        self.relative_slope_threshold = self.get_config("relative_slope_threshold", 0.004)
        self.slope_length = self.get_config("slope_length", 4)
        self.profit_info_log = self.get_config("profit_info_log", False)
        
        self.cooldown_candles = self.get_config("cooldown_candles", 1)
        self.stoploss_guard_lookback_period_candles = self.get_config("stoploss_guard_lookback_period_candles", 8)
        self.stoploss_guard_trade_limit = self.get_config("stoploss_guard_trade_limit", 4)
        self.stoploss_guard_stop_duration_candles = self.get_config("stoploss_guard_stop_duration_candles", 2)
        self.max_drawdown_lookback_period = self.get_config("max_drawdown_lookback_period", 0)
        self.max_drawdown_stop_duration = self.get_config("max_drawdown_stop_duration", 60)
        self.max_allowed_drawdown = self.get_config("max_allowed_drawdown", 0.3)
        
        self.backtesting_mode = self.get_config("backtesting_mode", False)
        
    def get_config(self, key: str, default):
        return self.config.get(key, default)
    
    def calculate_ha(self, df: DataFrame) -> DataFrame:
        if self.use_ha_candles:
            df_ref = qtpylib.heikinashi(df)
        else:
            df_ref = df
        
        df['ha_open'] = df_ref['open']
        df['ha_high'] = df_ref['high']
        df['ha_low'] = df_ref['low']
        df['ha_close'] = df_ref['close']
        return df
    
    def calc_ma(self, close, length: int):
        ma = pta.ema(close=close, length=length, talib=False)
        return ma.ffill() if ma is not None else ma
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            # haikinashi
            dataframe = self.calculate_ha(dataframe)
            
            # ema
            dataframe['ema_short'] = self.calc_ma(close=dataframe['ha_close'], length=self.ema_short_len.value)
            dataframe['ema_mid'] = self.calc_ma(close=dataframe['ha_close'], length=self.ema_mid_len.value)
            dataframe['ema_long'] = self.calc_ma(close=dataframe['ha_close'], length=self.ema_long_len.value)
            dataframe['ema_week'] = self.calc_ma(close=dataframe['ha_close'], length=self.ema_week_len.value)
            
            # atr
            dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_period)
            dataframe['natr'] = pta.natr(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], length=self.atr_period)
            
            # volume
            dataframe['obv'] = pta.obv(close=dataframe['ha_close'], volume=dataframe['volume'])
            dataframe['obv_mid_ma'] = pta.sma(close=dataframe['obv'], length=self.volume_mid)
            dataframe['volume_short_mean'] = dataframe['volume'].rolling(self.volume_short).mean()
            dataframe['volume_mid_mean'] = dataframe['volume'].rolling(self.volume_mid).mean()
            # dataframe['volume_long_mean'] = dataframe['volume'].rolling(self.volume_long).mean()
            
            # # cci
            # dataframe['cci'] = pta.cci(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], length=self.cci_period)
            
            dataframe['ha_close_ma'] = pta.ema(close=dataframe['ha_close'], length=self.close_ma_length, talib=False)
            dataframe['smooth_close'] = savgol_filter(dataframe['ha_close_ma'], window_length=self.close_smooth_window, polyorder=self.close_smooth_polyorder, mode='nearest')
            
            peaks, troughs = self.find_peaks_and_troughs(dataframe['smooth_close'])

            filtered_peaks, filtered_troughs = self.filter_extremes(peaks, troughs, dataframe['smooth_close'], dataframe.index, 
                                                                    distance_threshold=self.extreme_distance_threshold, ratio_threshold=self.extreme_ratio_threshold)

            dataframe['peak'] = np.nan
            dataframe.loc[filtered_peaks, 'peak'] = dataframe.loc[filtered_peaks, 'smooth_close']
            dataframe['trough'] = np.nan
            dataframe.loc[filtered_troughs, 'trough'] = dataframe.loc[filtered_troughs, 'smooth_close']

            fit_results = self.fit_linear_regression(dataframe, filtered_peaks, filtered_troughs)
            dataframe = pd.merge(dataframe, fit_results, left_index=True, right_index=True)
            return dataframe
        except Exception as e:
            return dataframe
    def find_peaks_and_troughs(self, series: pd.Series) -> tuple:
        peaks = series[(series.shift(1) < series) & (series.shift(-1) < series)].index
        troughs = series[(series.shift(1) > series) & (series.shift(-1) > series)].index
        return peaks, troughs

    def filter_extremes(self, peaks: pd.Index, troughs: pd.Index, series: pd.Series, index: pd.Index, distance_threshold: int, ratio_threshold: float) -> tuple:
        filtered_peaks = []
        filtered_troughs = []

        points = sorted(list(peaks) + list(troughs))
        last_point_index = None
        last_point_type = None
        last_point_value = None

        for i in range(len(points)):
            current_point = points[i]
            current_point_value = series[current_point]

            if last_point_index is not None:
                distance = i - last_point_index
                if distance >= distance_threshold:
                    if last_point_type == 'peak':
                        filtered_peaks.append(last_point)
                    else:
                        filtered_troughs.append(last_point)

                    last_point_index = i
                    last_point_type = 'peak' if current_point in peaks else 'trough'
                    last_point = current_point
                    last_point_value = current_point_value
                else:
                    if last_point_type == 'peak':
                        if abs(last_point_value / current_point_value - 1) <= ratio_threshold:
                            continue
                    else:
                        if abs(current_point_value / last_point_value - 1) <= ratio_threshold:
                            continue

                    if last_point_type == 'peak':
                        filtered_peaks.append(last_point)
                    else:
                        filtered_troughs.append(last_point)

                    last_point_index = i
                    last_point_type = 'peak' if current_point in peaks else 'trough'
                    last_point = current_point
                    last_point_value = current_point_value
            else:
                last_point_index = i
                last_point_type = 'peak' if current_point in peaks else 'trough'
                last_point = current_point
                last_point_value = current_point_value

        if last_point_type == 'peak':
            filtered_peaks.append(last_point)
        elif last_point_type == 'trough':
            filtered_troughs.append(last_point)

        return filtered_peaks, filtered_troughs

    def fit_linear_regression(self, dataframe: pd.DataFrame, peaks: list, troughs: list) -> pd.DataFrame:
        slopes = [None] * len(dataframe)
        relative_slopes = [None] * len(dataframe)
        intercepts = [None] * len(dataframe)
        fitted_lines = [None] * len(dataframe)
        segment_lengths = [None] * len(dataframe)

        control_points = sorted(set(peaks + troughs))

        for i in range(len(control_points) - 1):
            start_index = control_points[i]
            end_index = min(control_points[i + 1] + 1, len(dataframe))
            segment = dataframe.loc[start_index:end_index]

            x = np.array(segment.index).reshape(-1, 1)
            y = segment['smooth_close'].values

            slope, intercept = np.polyfit(x.flatten(), y, 1)

            for j in range(start_index, min(end_index+1, len(dataframe))):
                slopes[j] = slope
                relative_slopes[j] = slope / np.mean(y)
                intercepts[j] = intercept
                fitted_lines[j] = slope * j + intercept
                segment_lengths[j] = j - start_index

        if len(control_points) > 0:
            last_start_index = control_points[-1]
            last_end_index = len(dataframe)

            if last_start_index < len(dataframe) - 1:
                segment = dataframe.loc[last_start_index:last_end_index]

                x = np.array(segment.index).reshape(-1, 1)
                y = segment['smooth_close'].values
                slope, intercept = np.polyfit(x.flatten(), y, 1)

                for j in range(last_start_index, min(last_end_index+1, len(dataframe))):
                    slopes[j] = slope
                    relative_slopes[j] = slope / np.mean(y)
                    intercepts[j] = intercept
                    fitted_lines[j] = slope * j + intercept
                    segment_lengths[j] = j - last_start_index

        fit_results = pd.DataFrame({
            'fitted_line': fitted_lines,
            'slope': slopes,
            'relative_slope': relative_slopes,
            'segment_length': segment_lengths
        })

        return fit_results
    
    def indicator_up_n_periods_mask(self, dataframe: DataFrame, indicator: str, days: int):
        indicator_up_mask = (dataframe[f'{indicator}'] > dataframe[f'{indicator}'].shift(1))
        for i in range(2, days):
            indicator_up_mask = indicator_up_mask & (dataframe[f'{indicator}'].shift(i-1) > dataframe[f'{indicator}'].shift(i))
        return indicator_up_mask
    
    def indicator_down_n_periods_mask(self, dataframe: DataFrame, indicator: str, days: int):
        indicator_down_mask = (dataframe[f'{indicator}'] < dataframe[f'{indicator}'].shift(1))
        for i in range(2, days):
            indicator_down_mask = indicator_down_mask & (dataframe[f'{indicator}'].shift(i-1) < dataframe[f'{indicator}'].shift(i))
        return indicator_down_mask

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        if dataframe.empty:
            return dataframe
        
        try:
            dataframe.loc[
                (
                    (dataframe['ha_close'] > dataframe['ema_short'])
                    & (dataframe['ha_close'] > dataframe['ha_close'].shift(1))
                    & (dataframe['ha_close'] > dataframe['ha_open'])
                    & (dataframe['volume_short_mean'] > self.volume_ratio * dataframe['volume_mid_mean'])
                    & (dataframe['relative_slope'] > self.relative_slope_threshold)
                    & (dataframe['segment_length'] > self.slope_length)
                    & (self.indicator_up_n_periods_mask(dataframe, 'ema_short', self.trend_length))
                ), 
                ['enter_long', 'enter_tag']] = (1, 'entry_long')

            dataframe.loc[
                (
                    (dataframe['ha_close'] < dataframe['ema_short'])
                    & (dataframe['ha_close'] < dataframe['ha_close'].shift(1))
                    & (dataframe['ha_close'] < dataframe['ha_open'])
                    & (dataframe['volume_short_mean'] > self.volume_ratio * dataframe['volume_mid_mean'])
                    & (dataframe['relative_slope'] < -self.relative_slope_threshold)
                    & (dataframe['segment_length'] > self.slope_length)
                    & (self.indicator_down_n_periods_mask(dataframe, 'ema_short', self.trend_length))
                ), 
                ['enter_short', 'enter_tag']] = (1, 'entry_short')
            return dataframe
        except Exception as e:
            return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_long_trend'] = 0
        dataframe['exit_short_trend'] = 0
        
        if dataframe.empty:
            return dataframe
        
        try:
            dataframe.loc[
                (
                    (dataframe['ema_short'] < dataframe['ema_mid'])
                ), 
                'exit_long_trend'
            ] = 1
            
            dataframe.loc[
                (
                    (dataframe['ema_short'] > dataframe['ema_mid'])
                ), 
                'exit_short_trend'
            ] = 1
            
            dataframe.loc[
                (
                    (dataframe['ema_short'] < dataframe['ema_long'])
                ),
                ['exit_long', 'exit_tag']] = (1, 'exit_ma')
            
            dataframe.loc[
                (
                    (dataframe['ema_short'] > dataframe['ema_long'])
                ),
                ['exit_short', 'exit_tag']] = (1, 'exit_ma')
            
            return dataframe
        except Exception as e:
            return dataframe

    @property
    def protections(self): # type: ignore
        protections = [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_candles,
            }
        ]
        
        if self.stoploss_guard_lookback_period_candles > 0:
            protections.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": self.stoploss_guard_lookback_period_candles,
                    "trade_limit": self.stoploss_guard_trade_limit,
                    "stop_duration_candles": self.stoploss_guard_stop_duration_candles,
                    "only_per_pair": False
                }
            )
        
        if self.max_drawdown_lookback_period > 0:
            protections.append(
                {
                    "method": "MaxDrawdown",
                    "lookback_period": self.max_drawdown_lookback_period,
                    "stop_duration": self.max_drawdown_stop_duration,
                    "max_allowed_drawdown": self.max_allowed_drawdown,
                }
            )
        
        return protections

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        stake_amount = proposed_stake
        
        try:
            closed_trades = Trade.get_trades_proxy(is_open=False)
            closed_trades.sort(key=lambda x: x.close_date)
            latest_closed_trades = closed_trades[-self.custom_stake_lookback_trades:]
            
            latest_closed_profit = 0
            for closed_trade in latest_closed_trades:
                latest_closed_profit += closed_trade.close_profit_abs
            
            if latest_closed_profit < self.custom_stake_profit_threshold:
                stake_amount = stake_amount * self.custom_stake_ratio_when_low_closed_profit
                logger.warning(f'Set {pair} stake amount to:{stake_amount:.2f}(ratio:{self.custom_stake_ratio_when_low_closed_profit}). '
                            f'Latest closed total profit:{latest_closed_profit:.2f} < {self.custom_stake_profit_threshold:.2f}. '
                            f'Trades:{latest_closed_trades}')
            else:
                logger.info(f'Use proposed stake:{stake_amount} for {pair}. '
                            f'Latest closed total profit:{latest_closed_profit:.2f} > {self.custom_stake_profit_threshold:.2f}. ')
        except Exception as e:
            logger.warning(f'{pair} custom_stake_amount error:{e}')
        
        return stake_amount
    
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        # if not self.backtesting_mode:
        #     try:
        #         logger.warning(f'{pair} custom_exit')
        #         closed_trades = Trade.get_trades_proxy(is_open=False)
        #         closed_trades.sort(key=lambda x: x.close_date)
        #         latest_closed_trades = closed_trades[-self.custom_stake_lookback_trades:]
                
        #         latest_closed_profit = 0
        #         for closed_trade in latest_closed_trades:
        #             latest_closed_profit += closed_trade.close_profit_abs
                
        #         logger.warning(f'Latest closed profit:{latest_closed_profit:.2f}, trades:{latest_closed_trades}')
        #     except Exception as e:
        #         logger.warning(f'{pair} custom_exit error:{e}')
        
        open_rate = trade.open_rate
        leverage = trade.leverage
        _current_profit = current_profit / leverage
        stake_amount = trade.amount * trade.open_rate
        
        open_profit_abs = _current_profit * stake_amount
        realized_profit_abs = trade.realized_profit if trade.realized_profit else 0
        total_profit_abs = realized_profit_abs + open_profit_abs
        
        max_profit_abs = trade.get_custom_data(self.MAX_PROFIT_ABS)
        if max_profit_abs is None:
            max_profit_abs = total_profit_abs
            trade.set_custom_data(self.MAX_PROFIT_ABS, max_profit_abs)
        elif total_profit_abs > max_profit_abs:
            logger.warning(f'{trade.pair} reach new max profit: from {max_profit_abs:.4f} to {total_profit_abs:.4f}, current_rate:{current_rate:.5f} at {current_time}')
            max_profit_abs = total_profit_abs
            trade.set_custom_data(self.MAX_PROFIT_ABS, max_profit_abs)
        
        if self.profit_info_log:
            logger.info(f'{trade.pair} total profit:{total_profit_abs:.4f}(open:{open_profit_abs:.4f}, close:{realized_profit_abs:.4f}), current_rate:{current_rate:.5f}, open_rate:{trade.open_rate:.5f}, current_profit:{current_profit:.2%}, stake_amount:{stake_amount:.4f} at {current_time}')

        for _, (activation, drawdown_ratio) in enumerate(zip(self.profit_drawdown_activation, self.profit_drawdown_ratio)):
            if max_profit_abs > activation:
                drawdown_profit_threshold = max_profit_abs * (1 - drawdown_ratio)
                if total_profit_abs < drawdown_profit_threshold:
                    exit_reason = f'Profit drawdown'
                    logger.warning(f'{exit_reason} for {pair}: total profit {total_profit_abs:.4f} < {drawdown_profit_threshold:.4f}'
                                    f'(max_profit_abs:{max_profit_abs:.4f} * 1-drawdown:{drawdown_ratio:.2%}), '
                                    f'current_rate:{current_rate:.5f}, open_rate:{trade.open_rate:.5f} at {current_time}')
                    return exit_reason
                
                break

        count_of_orders = len(trade.select_filled_orders())
        if count_of_orders < 1:
            if trade.is_short:
                max_profit = (open_rate - trade.min_rate) / open_rate - 2 * self.fee
            else:
                max_profit = (trade.max_rate - open_rate) / open_rate - 2 * self.fee

            open_hours = round((current_time - trade.open_date_utc).total_seconds() / 3600, 1)
            if open_hours > self.long_time_low_profit_hours:
                if max_profit < self.long_time_low_profit_max and self.long_time_low_profit_lower_bound < _current_profit < self.long_time_low_profit_higher_max_ratio * max_profit:
                    return "longtime_low_profit"
                
            if self.long_time_hours > 0 and open_hours > self.long_time_hours:
                if self.long_time_hours_min_profit < _current_profit < self.long_time_hours_max_profit:
                    return "longtime"
        
        return None
        
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
        return None
    
        # leverage = trade.leverage
        # is_short = trade.is_short
        # factor = -1 if is_short else 1
        # _current_profit = current_profit / leverage
        # return None

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        if not self.position_adjustment_enable:
            return None
        
        has_open_orders = any(order.status == "open" and not order.ft_is_open for order in trade.orders)
        if has_open_orders:
            logger.info(f'There are open orders for {trade.pair}, skip position adjustment.')
            return None
        
        filled_orders = trade.select_filled_orders()
        count_of_orders = len(filled_orders)
        if count_of_orders == 0:
            logger.info(f'No filled orders for {trade.pair}, skip position adjustment.')
            return None
        
        entry_side_orders = [order for order in filled_orders \
            if order.ft_order_side == trade.entry_side and ('entry' in order.ft_order_tag)]
        count_of_orders = len(entry_side_orders)
        if count_of_orders == 0:
            logger.info(f'No entry orders for {trade.pair}, skip position adjustment.')
            return None
        
        leverage = trade.leverage
        _current_profit = current_profit / leverage
        
        first_entry_order = entry_side_orders[0]
        first_stake_amount = first_entry_order.stake_amount * leverage
        
        if count_of_orders == 1:
            reduction_threshold = -self.base_stop_loss * self.reduction_stoploss_ratio
            
            if self.reduction_stoploss_ratio > 0 and _current_profit < reduction_threshold:
                entry_side_orders = [order for order in filled_orders if ('reduction' in order.ft_order_tag)]
                already_reduced = len(entry_side_orders) >= 1
                if not already_reduced:
                    reduction_stake_amount = -(first_stake_amount * self.reduction_stake_ratio)
                    
                    logger.warning(f'Position reduction for {trade.pair} with stake amount {reduction_stake_amount:.5f}(total:{first_stake_amount:.5f}), '
                            f'current_profit:{current_profit:.2%}(without leverage:{_current_profit:.2%}), current_rate:{current_rate:.5f} at {current_time}')
                    
                    return (reduction_stake_amount / leverage, "reduction")
        
        addition_stake = first_stake_amount * self.addition_stake_ratio
        addition_amount = round(addition_stake / current_rate, 2)
        if addition_amount <= 0:
            logger.info(f'Addition amount for {trade.pair} is zero, skip position adjustment.')
            return None
        
        addition_stake = addition_amount * current_rate
        
        is_short = trade.is_short
        factor = -1 if is_short else 1
        
        new_open_rate = (trade.amount * trade.open_rate + addition_stake) / (trade.amount + addition_amount)
        new_open_profit = factor * (current_rate / new_open_rate - 1)
        
        enough_profit = new_open_profit > (self.addition_min_profit + self.addition_min_profit_step * (count_of_orders-1))
        last_entry_price = entry_side_orders[-1].average
        
        addition_signal = False
        if enough_profit:
            if is_short: # and last_candle['enter_short'] == 1
                addition_signal = current_rate < last_entry_price * (1 + factor * self.addition_profit_step)
            elif not is_short: # and last_candle['enter_long'] == 1
                addition_signal = current_rate > last_entry_price * (1 + factor * self.addition_profit_step)

        if addition_signal:
            if min_stake <= addition_stake <= max_stake:
                logger.warning(f'Position addition #{count_of_orders+1} for {trade.pair} with estimated new_profit:{new_open_profit*leverage:.2%} and stake amount {addition_stake:.5f}, '
                            f'current_profit:{current_profit:.2%}, current_rate:{current_rate:.5f} at {current_time}')
                return (addition_stake / leverage, f'entry-addition')
            else:
                logger.warning(f'Skip position addition for {trade.pair} with estimated new_profit:{new_open_profit*leverage:.2%} and stake amount {addition_stake:.5f} is out of range'
                            f'({min_stake:.2f}-{max_stake:.2f}), current_profit:{current_profit:.2%}, current_rate:{current_rate:.5f} at {current_time}')
        
        return None
