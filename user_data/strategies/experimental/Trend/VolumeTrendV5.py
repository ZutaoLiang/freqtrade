import numpy as np
import pandas_ta as pta
import pandas as pd
# pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from pandas import DataFrame

from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter

import freqtrade.vendor.qtpylib.indicators as qtpylib
# from freqtrade.strategy.strategy_helper import stoploss_from_absolute, stoploss_from_open
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, informative
from freqtrade.persistence import Order, Trade
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
import logging
logger = logging.getLogger(__name__)


class VolumeTrendV5(IStrategy):
    timeframe = '15m'
    trade_leverage = IntParameter(1, 10, default=5, space='buy')
    
    minimal_roi = {"0": 100}
    
    use_ha_candles = True
    enable_profit_decay = False
    enable_negative_exit = False

    base_stop_loss = 0.07
    stoploss = -base_stop_loss * trade_leverage.value

    base_trailing_stop = 0.045
    base_trailing_stop_offset = 0.06

    # trailing_stop = True
    # trailing_stop_positive = base_trailing_stop * trade_leverage.value
    # trailing_stop_positive_offset = base_trailing_stop_offset * trade_leverage.value
    # trailing_only_offset_is_reached = True

    trailing_stop = False
    # use_custom_stoploss = True

    can_short = True
    position_adjustment_enable = True
    addition_stake_ratio = 2
    addition_min_profit = 0.15
    addition_profit_step = 0.03
    
    dynamic_entry_by_signal_score = True

    lookback_period = 12
    
    window_length = IntParameter(10, 100, default=lookback_period, space='buy')

    ema_short_len = IntParameter(5, 100, default=lookback_period, space='buy')
    ema_mid_len = IntParameter(5, 100, default=lookback_period * 3, space='buy')
    ema_long_len = IntParameter(5, 100, default=lookback_period * 6, space='buy')
    ema_week_len = IntParameter(5, 100, default=lookback_period * 4 * 3, space='buy')

    startup_candle_count = int(max(window_length.value, ema_week_len.value) * 1.2)

    atr_period = 21
    rsi_period = 14
    cci_period = 21
    cci_threshold = 120
    
    volume_short = 1
    volume_mid = 30
    # volume_long = ema_week_len.value
    volume_ratio = 3
    
    trend_length = 2
    
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
        
        # ao/ac
        dataframe['ao'] = pta.ao(high=dataframe['ha_high'], low=dataframe['ha_low'])
        dataframe['ac'] = dataframe['ao'] - dataframe['ao'].rolling(window=5).mean()
        
        # volume
        dataframe['obv'] = pta.obv(close=dataframe['ha_close'], volume=dataframe['volume'])
        dataframe['obv_mid_ma'] = pta.sma(close=dataframe['obv'], length=self.volume_mid)
        # dataframe['ad'] = pta.ad(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], volume=dataframe['volume'])
        # dataframe['adosc'] = pta.adosc(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], volume=dataframe['volume'])
        # dataframe['pvol'] = pta.pvol(close=dataframe['ha_close'], volume=dataframe['volume'])
        dataframe['volume_short_mean'] = dataframe['volume'].rolling(self.volume_short).mean()
        dataframe['volume_mid_mean'] = dataframe['volume'].rolling(self.volume_mid).mean()
        # dataframe['volume_long_mean'] = dataframe['volume'].rolling(self.volume_long).mean()
        
        # rsi
        # dataframe['rsi'] = pta.rsi(close=dataframe['ha_close'], length=self.rsi_period)
        
        # cci
        dataframe['cci'] = pta.cci(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], length=self.cci_period)
        
        dataframe['ha_close_ma'] = pta.ema(close=dataframe['ha_close'], length=5, talib=False)
        dataframe['smooth_close'] = savgol_filter(dataframe['ha_close_ma'], window_length=15, polyorder=1)
        
        peaks, troughs = self.find_peaks_and_troughs(dataframe['smooth_close'])

        filtered_peaks, filtered_troughs = self.filter_extremes(peaks, troughs, dataframe['smooth_close'], dataframe.index, distance_threshold=5, ratio_threshold=0.005)

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
            logger.info(f"Pair {pair}(score:{signals.get(pair)}) is in top {len(top_pairs)} signal score ranking: {top_signals}")
            return True
        else:
            logger.info(f"Current pair {pair}(score:{signals.get(pair)}) not in top {len(top_pairs)} signal score ranking: {top_signals}, skip entry")
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
        else:
            filtered_troughs.append(last_point)

        return filtered_peaks, filtered_troughs

    def fit_linear_regression(self, dataframe: pd.DataFrame, peaks: list, troughs: list) -> pd.DataFrame:
        slopes = [None] * len(dataframe)
        intercepts = [None] * len(dataframe)
        fitted_lines = [None] * len(dataframe)
        segment_lengths = [None] * len(dataframe)

        control_points = sorted(set(peaks + troughs))

        for i in range(len(control_points) - 1):
            start_index = control_points[i]
            end_index = min(control_points[i + 1], len(dataframe))
            segment = dataframe.loc[start_index:end_index]

            x = np.array(segment.index).reshape(-1, 1)
            y = segment['smooth_close'].values

            model = LinearRegression().fit(x, y)
            slope = model.coef_[0]
            intercept = model.intercept_

            for j in range(start_index, end_index):
                slopes[j] = slope
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

                model = LinearRegression().fit(x, y)
                slope = model.coef_[0]
                intercept = model.intercept_

                for j in range(last_start_index, last_end_index):
                    slopes[j] = slope
                    intercepts[j] = intercept
                    fitted_lines[j] = slope * j + intercept
                    segment_lengths[j] = j - last_start_index

        fit_results = pd.DataFrame({
            'fitted_line': fitted_lines,
            'slope': slopes,
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
                
                # & (dataframe['ema_short'] > dataframe['ema_long'])
                # & (dataframe['ema_mid'] > dataframe['ema_week'])
                # & (dataframe['ema_long'] > dataframe['ema_week'])
                # & (self.indicator_up_n_periods_mask(dataframe, 'ema_short', self.trend_length))
                # & (self.indicator_up_n_periods_mask(dataframe, 'ema_mid', self.trend_length))
                # & (self.indicator_up_n_periods_mask(dataframe, 'ema_long', self.trend_length))
                # & (self.indicator_up_n_periods_mask(dataframe, 'ema_week', self.trend_length))
            ), 
            ['enter_long', 'enter_tag']] = (1, 'entry_long')

        dataframe.loc[
            (
                (dataframe['ha_close'] < dataframe['ema_short'])
                & (dataframe['ha_close'] < dataframe['ha_close'].shift(1))
                & (dataframe['ha_close'] < dataframe['ha_open'])
                & (dataframe['volume_short_mean'] > self.volume_ratio * dataframe['volume_mid_mean'])
                
                # & (dataframe['ema_short'] < dataframe['ema_long'])
                # & (dataframe['ema_mid'] < dataframe['ema_week'])
                # & (dataframe['ema_long'] < dataframe['ema_week'])
                # & (self.indicator_down_n_periods_mask(dataframe, 'ema_short', self.trend_length))
                # & (self.indicator_down_n_periods_mask(dataframe, 'ema_mid', self.trend_length))
                # & (self.indicator_down_n_periods_mask(dataframe, 'ema_long', self.trend_length))
                # & (self.indicator_down_n_periods_mask(dataframe, 'ema_week', self.trend_length))
            ), 
            ['enter_short', 'enter_tag']] = (1, 'entry_short')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        dataframe.loc[
            (
                (dataframe['ema_short'] < dataframe['ema_mid'])
                & (dataframe['ha_close'] < dataframe['ema_short'])
                & (dataframe['cci'] < -self.cci_threshold)
                & (self.indicator_down_n_periods_mask(dataframe, 'obv', self.trend_length))
                & (dataframe['ao'] < 0)
                & (self.indicator_down_n_periods_mask(dataframe, 'ao', self.trend_length))
                & (dataframe['ac'] < 0)
                & (self.indicator_down_n_periods_mask(dataframe, 'ac', self.trend_length))
            ), 
            ['exit_long', 'exit_tag']] = (1, 'exit_signal')
        
        dataframe.loc[
            (
                (dataframe['ema_mid'] < dataframe['ema_long'])
                & (dataframe['ema_short'] < dataframe['ema_long'])
                # & (dataframe['enter_long'] == 0)
            ), 
            ['exit_long', 'exit_tag']] = (1, 'exit_ema')
        
        dataframe.loc[
            (
                (dataframe['ema_short'] > dataframe['ema_mid'])
                & (dataframe['ha_close'] > dataframe['ema_short'])
                & (dataframe['cci'] > self.cci_threshold)
                & (self.indicator_up_n_periods_mask(dataframe, 'obv', self.trend_length))
                & (dataframe['ao'] > 0)
                & (self.indicator_up_n_periods_mask(dataframe, 'ao', self.trend_length))
                & (dataframe['ac'] > 0)
                & (self.indicator_up_n_periods_mask(dataframe, 'ac', self.trend_length))
            ), 
            ['exit_short', 'exit_tag']] = (1, 'exit_signal')
        
        dataframe.loc[
            (
                (dataframe['ema_mid'] > dataframe['ema_long'])
                & (dataframe['ema_short'] > dataframe['ema_long'])
                # & (dataframe['enter_short'] == 0)
            ), 
            ['exit_short', 'exit_tag']] = (1, 'exit_ema')
        
        return dataframe

    @property
    def protections(self): # type: ignore
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 1
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage.value

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
        
        count_of_orders = len(trade.select_filled_orders())
        if count_of_orders > 1:
            if count_of_orders > 5 and _current_profit < 0.08:
                return "addition_drawdown_1"
            
            if count_of_orders > 3 and _current_profit < 0.06:
                return "addition_drawdown_2"
            
            if _current_profit < 0.03:
                return "addition_drawdown_3"
    
        if trade.is_short:
            max_profit = (open_rate - trade.min_rate) / open_rate
        else:
            max_profit = (trade.max_rate - open_rate) / open_rate

        open_hours = round((current_time - trade.open_date_utc).total_seconds() / 3600, 1)
        if open_hours > 4:
            if max_profit < 0.05 and 0 < _current_profit < 0.4 * max_profit:
                return "longtime_low_profit_40"
        
        return None
        
    # def custom_stoploss(
    #     self,
    #     pair: str,
    #     trade: Trade,
    #     current_time: datetime,
    #     current_rate: float,
    #     current_profit: float,
    #     after_fill: bool,
    #     **kwargs,
    # ) -> float | None:
    #     leverage = trade.leverage
    #     is_short = trade.is_short
    #     factor = -1 if is_short else 1
    #     _current_profit = current_profit / leverage
        
    #     if _current_profit > self.base_trailing_stop_offset:
    #         return self.base_trailing_stop * leverage
        
    #     return None

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
        
        entry_side_orders = [order for order in filled_orders
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
        
        enough_profit = new_open_profit > self.addition_min_profit
        last_entry_price = entry_side_orders[-1].average
        
        # dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # last_candle = dataframe.iloc[-1].squeeze()
        
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