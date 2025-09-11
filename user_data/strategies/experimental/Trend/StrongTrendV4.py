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


class StrongTrendV4(IStrategy):
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
        
        self.addition_stake_ratio = self.get_config("addition_stake_ratio", 0.8)
        self.addition_min_profit = self.get_config("addition_min_profit", 0.08)
        self.addition_min_profit_step = self.get_config("addition_min_profit_step", 0.025)
        self.addition_profit_step = self.get_config("addition_profit_step", 0.04)
        
        activation_list = str(self.get_config("profit_drawdown_activation", "1")).split(",")
        self.profit_drawdown_activation = [float(s) for s in activation_list]
        drawdown_ratio_list = str(self.get_config("profit_drawdown_ratio", "0.4")).split(",")
        self.profit_drawdown_ratio = [float(s) for s in drawdown_ratio_list]
        
        self.fee = self.get_config("fee", 0.0005)
        self.long_time_low_profit_hours = self.get_config("long_time_low_profit_hours", 3)
        self.long_time_low_profit_max = self.get_config("long_time_low_profit_max", 0.05)
        self.long_time_low_profit_lower_bound = self.get_config("long_time_low_profit_lower_bound", 0.005)
        self.long_time_low_profit_higher_max_ratio = self.get_config("long_time_low_profit_higher_max_ratio", 0.5)
        self.relative_slope_threshold = self.get_config("relative_slope_threshold", 0.004)
        self.slope_length = self.get_config("slope_length", 4)
        self.profit_info_log = self.get_config("profit_info_log", False)
        
        self.cooldown_candles = self.get_config("cooldown_candles", 1)
        self.max_drawdown_lookback_period = self.get_config("max_drawdown_lookback_period", 720)
        self.max_drawdown_stop_duration = self.get_config("max_drawdown_stop_duration", 360)
        self.max_allowed_drawdown = self.get_config("max_allowed_drawdown", 0.3)
        
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
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # haikinashi
        dataframe = self.calculate_ha(dataframe)
        
        # ema
        dataframe['ema_short'] = pta.ema(close=dataframe['ha_close'], length=self.ema_short_len.value, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['ha_close'], length=self.ema_mid_len.value, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ha_close'], length=self.ema_long_len.value, talib=False)
        dataframe['ema_week'] = pta.ema(close=dataframe['ha_close'], length=self.ema_week_len.value, talib=False)
        
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
        dataframe['smooth_close'] = savgol_filter(dataframe['ha_close_ma'], window_length=self.close_smooth_window, polyorder=self.close_smooth_polyorder)
        
        peaks, troughs = self.find_peaks_and_troughs(dataframe['smooth_close'])

        filtered_peaks, filtered_troughs = self.filter_extremes(peaks, troughs, dataframe['smooth_close'], dataframe.index, 
                                                                distance_threshold=self.extreme_distance_threshold, ratio_threshold=self.extreme_ratio_threshold)

        dataframe['peak'] = np.nan
        dataframe.loc[filtered_peaks, 'peak'] = dataframe.loc[filtered_peaks, 'smooth_close']
        dataframe['trough'] = np.nan
        dataframe.loc[filtered_troughs, 'trough'] = dataframe.loc[filtered_troughs, 'smooth_close']

        fit_results = self.fit_linear_regression(dataframe, filtered_peaks, filtered_troughs)
        dataframe = pd.merge(dataframe, fit_results, left_index=True, right_index=True)

        if self.dynamic_entry_by_signal_score:
            dataframe = self.calc_signal_score(dataframe)
        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        if not self.dynamic_entry_by_signal_score:
            return True
        
        open_trades = len(Trade.get_open_trades())
        max_trades = self.config.get('max_open_trades')
        
        if open_trades >= max_trades:
            return False
        
        signals = {}
        for whitelist_pair in self.dp.current_whitelist():
            try:
                dataframe, _ = self.dp.get_analyzed_dataframe(whitelist_pair, self.timeframe)
                if dataframe is None or dataframe.empty:
                    continue
                    
                last_candle = dataframe.iloc[-1].squeeze()
                
                if last_candle.get('enter_long', False) or last_candle.get('enter_short', False):
                    signals[whitelist_pair] = round(float(last_candle.get('signal_score', 0)), 2)
                    
            except Exception as e:
                self.logger.warning(f"Error getting signals for {whitelist_pair}: {e}")
                continue
            
        if not signals:
            return True
        
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        top_signals = sorted_signals[:(max_trades - open_trades)]
        top_pairs = [item[0] for item in top_signals]
        
        if pair in top_pairs:
            # logger.info(f"Pair {pair}(score:{signals.get(pair)}) is in top {len(top_pairs)} signal score ranking: {top_signals}")
            return True
        else:
            # logger.info(f"Current pair {pair}(score:{signals.get(pair)}) not in top {len(top_pairs)} signal score ranking: {top_signals}, skip entry")
            return False

    def calc_signal_score(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        score = pd.Series(0.0, index=dataframe.index)
        score += (dataframe['volume_short_mean'] / dataframe['volume_mid_mean'])
        score += (dataframe['segment_length'] / 10).clip(upper=self.volume_ratio/2)
        score += (dataframe['slope'].abs() / 0.001).clip(upper=self.volume_ratio/2)
        
        dataframe['signal_score'] = score
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
        
        dataframe.loc[
            (
                (dataframe['ha_close'] > dataframe['ema_short'])
                & (dataframe['ha_close'] > dataframe['ha_close'].shift(1))
                & (dataframe['ha_close'] > dataframe['ha_open'])
                & (dataframe['volume_short_mean'] > self.volume_ratio * dataframe['volume_mid_mean'])
                & (dataframe['relative_slope'] > self.relative_slope_threshold)
                & (dataframe['segment_length'] > self.slope_length)
                & (dataframe['ema_short'] > dataframe['ema_mid'])
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
                & (dataframe['ema_short'] < dataframe['ema_mid'])
                & (self.indicator_down_n_periods_mask(dataframe, 'ema_short', self.trend_length))
            ), 
            ['enter_short', 'enter_tag']] = (1, 'entry_short')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_long_trend'] = 0
        dataframe['exit_short_trend'] = 0
        
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

    @property
    def protections(self): # type: ignore
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_candles,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period": self.max_drawdown_lookback_period,
                "stop_duration": self.max_drawdown_stop_duration,
                "max_allowed_drawdown": self.max_allowed_drawdown,
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
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
        
        first_entry_order = entry_side_orders[0]
        first_stake_amount = first_entry_order.stake_amount * leverage
        
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
                logger.info(f'Position addition #{count_of_orders+1} for {trade.pair} with estimated new_profit:{new_open_profit*leverage:.2%} and stake amount {addition_stake:.5f}, '
                            f'current_profit:{current_profit:.2%}, current_rate:{current_rate:.5f} at {current_time}')
                return (addition_stake / leverage, f'entry-addition')
            else:
                logger.warning(f'Skip position addition for {trade.pair} with estimated new_profit:{new_open_profit*leverage:.2%} and stake amount {addition_stake:.5f} is out of range'
                            f'({min_stake:.2f}-{max_stake:.2f}), current_profit:{current_profit:.2%}, current_rate:{current_rate:.5f} at {current_time}')
        
        return None
