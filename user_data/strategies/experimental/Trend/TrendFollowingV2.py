import json
from typing import Optional

import numpy as np
import pandas_ta as pta
import pandas as pd
pd.set_option('display.width', None)
from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy.strategy_helper import stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Order, Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib

from datetime import datetime, timezone, timedelta
import logging
logger = logging.getLogger(__name__)


class TrendFollowingV2(IStrategy):
    """
    TrendFollowingV2 - 基于V1增加4h SuperTrend多空方向过滤

    相比V1的核心改动：
      1. 新增4h SuperTrend指标作为高级别趋势过滤
      2. 做多时要求4h SuperTrend方向为多（direction == 1）
      3. 做空时要求4h SuperTrend方向为空（direction == -1）
      4. 出场增加4h SuperTrend趋势翻转退出逻辑
    """

    timeframe = '30m'

    minimal_roi = {"0": 100}
    can_short = True
    process_only_new_candles = True

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.stake_amount = self.get_config("stake_amount", 6)
        self.trade_leverage = self.get_config("trade_leverage", 3)

        self.trailing_stop = self.get_config("trailing_stop", True)
        if not self.trailing_stop:
            self.custom_trailing_stop = self.get_config("custom_trailing_stop", False)
        else:
            self.trailing_stop_positive = self.get_config("base_trailing_stop", 0.12) * self.trade_leverage
            self.trailing_stop_positive_offset = self.get_config("base_trailing_stop_offset", 0.3) * self.trade_leverage
            self.trailing_only_offset_is_reached = self.get_config("trailing_only_offset_is_reached", True)

        self.base_stop_loss = self.get_config("base_stop_loss", 0.07)
        self.stoploss = - float(self.base_stop_loss * self.trade_leverage)
        self.use_custom_stoploss = self.get_config("use_custom_stoploss", False)
        self.atr_stop_loss_multiplier = self.get_config("atr_stop_loss_multiplier", 0)

        self.use_ha_candles = self.get_config("use_ha_candles", False)

        self.trend_length = self.get_config("trend_length", 3)

        self.ma_short_length = self.get_config("ma_short_length", 0)
        self.ma_mid_length = self.get_config("ma_mid_length", 0)
        self.ma_long_length = self.get_config("ma_long_length", 0)

        self.crossover_lookback_length = self.get_config("crossover_lookback_length", 8)

        # 4h SuperTrend参数
        self.htf = self.get_config("htf", "4h")
        self.supertrend_period = self.get_config("supertrend_period", 10)
        self.supertrend_multiplier = float(self.get_config("supertrend_multiplier", 3.0))
        self.confirm_candles_htf = self.get_config("confirm_candles_htf", 1)

        self.startup_candle_count = int(max(self.ma_mid_length, self.ma_long_length, 200) * 1)

        self.atr_period = self.get_config("atr_period", 21)

        self.position_adjustment_enable = self.get_config("position_adjustment_enable", False)
        self.addition_stake_ratio = self.get_config("addition_stake_ratio", 0.8)
        self.addition_min_profit = self.get_config("addition_min_profit", 0.2)
        self.addition_min_profit_step = self.get_config("addition_min_profit_step", 0.05)
        self.addition_profit_step = self.get_config("addition_profit_step", 0.05)

        self.fee = self.get_config("fee", 0.0005)

        self.long_time_low_profit_minutes = self.get_config("long_time_low_profit_minutes", 0)
        self.long_time_low_profit_max = self.get_config("long_time_low_profit_max", 0.05)
        self.long_time_low_profit_lower_bound = self.get_config("long_time_low_profit_lower_bound", 0.003)
        self.long_time_low_profit_upper_bound = self.get_config("long_time_low_profit_upper_bound", 0.02)

        self.long_time_stoploss_minutes = self.get_config("long_time_stoploss_minutes", 0)
        self.long_time_stoploss_profit = self.get_config("long_time_stoploss_profit", 0.03)

        self.cooldown_candles = self.get_config("cooldown_candles", 1)
        self.stoploss_guard_lookback_period_candles = self.get_config("stoploss_guard_lookback_period_candles", 0)
        self.stoploss_guard_trade_limit = self.get_config("stoploss_guard_trade_limit", 4)
        self.stoploss_guard_stop_duration_candles = self.get_config("stoploss_guard_stop_duration_candles", 2)
        self.max_drawdown_lookback_period = self.get_config("max_drawdown_lookback_period", 0)
        self.max_drawdown_stop_duration = self.get_config("max_drawdown_stop_duration", 60)
        self.max_allowed_drawdown = self.get_config("max_allowed_drawdown", 0.3)

    def get_config(self, key: str, default):
        return self.config.get(key, default)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, self.htf) for pair in pairs]

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
        ma = pta.wma(close=close, length=length, talib=False)
        return ma.ffill() if ma is not None else ma

    def _calc_supertrend(self, df: DataFrame) -> DataFrame:
        """计算SuperTrend指标"""
        st = pta.supertrend(
            high=df['high'], low=df['low'], close=df['close'],
            length=self.supertrend_period, multiplier=self.supertrend_multiplier
        )
        dir_col = [c for c in st.columns if c.startswith('SUPERTd')][0]
        df['st_direction'] = st[dir_col]
        return df

    def _calc_consecutive(self, df: DataFrame, col: str, out_col: str) -> DataFrame:
        """计算某列连续相同值的根数"""
        direction = df[col]
        groups = (direction != direction.shift()).cumsum()
        df[out_col] = direction.groupby(groups).cumcount() + 1
        return df

    def _merge_htf(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """获取4h SuperTrend并合并到主timeframe"""
        try:
            htf_df = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.htf)
            if htf_df.empty:
                logger.warning(f"No {self.htf} data for {metadata['pair']}")
                dataframe['htf_st_direction'] = 0
                dataframe['htf_st_consec'] = 0
                return dataframe

            htf_df = self._calc_supertrend(htf_df)
            htf_df = self._calc_consecutive(htf_df, 'st_direction', 'st_consec')

            dataframe = merge_informative_pair(
                dataframe, htf_df, self.timeframe, self.htf, ffill=True
            )

            dataframe['htf_st_direction'] = dataframe[f'st_direction_{self.htf}']
            dataframe['htf_st_consec'] = dataframe[f'st_consec_{self.htf}']

        except Exception as e:
            logger.error(f"{self.__class__.__name__}::_merge_htf error: {e}", exc_info=True)
            dataframe['htf_st_direction'] = 0
            dataframe['htf_st_consec'] = 0

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            dataframe = self.calculate_ha(dataframe)

            # ma
            if self.ma_short_length > 0:
                dataframe['ma_short'] = self.calc_ma(close=dataframe['ha_close'], length=self.ma_short_length)

            if self.ma_mid_length > 0:
                dataframe['ma_mid'] = self.calc_ma(close=dataframe['ha_close'], length=self.ma_mid_length)

            if self.ma_long_length > 0:
                dataframe['ma_long'] = self.calc_ma(close=dataframe['ha_close'], length=self.ma_long_length)

            # atr
            dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_period)
            dataframe['natr'] = pta.natr(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], length=self.atr_period, talib=False, scalar=1.0)

            # vwap
            ema_close = pta.ema(close=(dataframe['ha_close'] + dataframe['ha_high'] + dataframe['ha_low']) / 3, length=3)
            dataframe['vwap'] = self.rolling_vwap(high=ema_close, low=ema_close, close=ema_close, volume=dataframe['volume'])

            # 合并4h SuperTrend数据
            dataframe = self._merge_htf(dataframe, metadata)

            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_indicators: {e}")
            return dataframe

    def rolling_vwap(self, high, low, close, volume, window=21):
        tp = (high + low + close) / 3
        return (tp * volume).rolling(window).sum() / volume.rolling(window).sum()

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        if dataframe.empty:
            return dataframe

        try:
            dataframe['crossover_long'] = 0
            dataframe.loc[(
                (dataframe['ma_short'] > dataframe['ma_mid'])
                & (
                    (dataframe['ma_short'].shift(1) < dataframe['ma_mid'].shift(1)) |
                    (dataframe['ma_short'].shift(2) < dataframe['ma_mid'].shift(2)) |
                    (dataframe['ma_short'].shift(3) < dataframe['ma_mid'].shift(3))
                )), 'crossover_long'] = 1

            recent_crossover_long_mask = (dataframe['crossover_long'].rolling(window=self.crossover_lookback_length, min_periods=1).sum() > 0)

            enter_long_mask = \
                (dataframe['ha_close'] >= dataframe['ha_open']) \
                & (dataframe['ha_close'] >= dataframe['ma_short']) \
                & (dataframe['ha_close'] >= dataframe['ma_mid']) \
                & (recent_crossover_long_mask) \
                & (dataframe['ma_short'] > dataframe['ma_mid']) \
                & (self.indicator_up_n_periods_mask(dataframe, 'ma_short', self.trend_length)) \
                & (self.indicator_up_n_periods_mask(dataframe, 'ma_mid', self.trend_length)) 
                
            enable_super_trend_entry = False
            
            if enable_super_trend_entry:
                enter_long_mask &= (
                    (dataframe['htf_st_direction'] == 1) \
                    & (dataframe['htf_st_consec'] >= self.confirm_candles_htf)
                )

            dataframe.loc[enter_long_mask, ['enter_long', 'enter_tag']] = (1, 'entry_long')

            dataframe['crossover_short'] = 0
            dataframe.loc[(
                (dataframe['ma_short'] < dataframe['ma_mid'])
                & (
                    (dataframe['ma_short'].shift(1) > dataframe['ma_mid'].shift(1)) |
                    (dataframe['ma_short'].shift(2) > dataframe['ma_mid'].shift(2)) |
                    (dataframe['ma_short'].shift(3) > dataframe['ma_mid'].shift(3))
                )), 'crossover_short'] = 1

            recent_crossover_short_mask = (dataframe['crossover_short'].rolling(window=self.crossover_lookback_length, min_periods=1).sum() > 0)

            enter_short_mask = \
                (dataframe['ha_close'] <= dataframe['ha_open']) \
                & (dataframe['ha_close'] <= dataframe['ma_short']) \
                & (dataframe['ha_close'] <= dataframe['ma_mid']) \
                & (recent_crossover_short_mask) \
                & (dataframe['ma_short'] <= dataframe['ma_mid']) \
                & (self.indicator_down_n_periods_mask(dataframe, 'ma_short', self.trend_length)) \
                & (self.indicator_down_n_periods_mask(dataframe, 'ma_mid', self.trend_length))
                
            if enable_super_trend_entry:
                enter_short_mask &= (
                    (dataframe['htf_st_direction'] == -1) \
                    & (dataframe['htf_st_consec'] >= self.confirm_candles_htf)
                )

            dataframe.loc[enter_short_mask, ['enter_short', 'enter_tag']] = (1, 'entry_short')

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
            # MA交叉出场
            dataframe.loc[(
                    (dataframe['exit_long'] == 0)
                    & (dataframe['ma_short'] < dataframe['ma_mid'])
                ), ['exit_long', 'exit_tag']] = (1, 'exit_ma')

            dataframe.loc[(
                    (dataframe['exit_short'] == 0)
                    & (dataframe['ma_short'] > dataframe['ma_mid'])
                ), ['exit_short', 'exit_tag']] = (1, 'exit_ma')

            # SuperTrend趋势翻转出场
            dataframe.loc[(
                    (dataframe['exit_long'] == 0)
                    & (dataframe['htf_st_direction'] == -1)
                ), ['exit_long', 'exit_tag']] = (1, 'exit_htf_reversal')

            dataframe.loc[(
                    (dataframe['exit_short'] == 0)
                    & (dataframe['htf_st_direction'] == 1)
                ), ['exit_short', 'exit_tag']] = (1, 'exit_htf_reversal')

            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_exit_trend: {e}")
            return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, after_fill: bool,
                       **kwargs) -> Optional[float]:
        if self.atr_stop_loss_multiplier <= 0:
            return None

        leverage = trade.leverage
        is_short = trade.is_short
        open_rate = trade.open_rate
        _current_profit = current_profit / leverage

        if after_fill:
            filled_orders = trade.select_filled_orders()
            count_of_orders = len(filled_orders)
            if count_of_orders == 0:
                return None

            last_candle = self.get_last_candle(trade)
            atr = last_candle['atr']
            natr = last_candle['natr']

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

        if self.custom_trailing_stop:
            if _current_profit > self.get_config("base_trailing_stop_offset", 0.3):
                return self.get_config("base_trailing_stop", 0.12) * leverage

        if self.long_time_stoploss_minutes > 0:
            open_minutes = round((current_time - trade.open_date_utc).total_seconds() / 60, 1)
            if open_minutes > self.long_time_stoploss_minutes and _current_profit > (self.long_time_stoploss_profit + 0.005):
                    return stoploss_from_open(self.long_time_stoploss_profit * leverage, current_profit, is_short, leverage)

        return None

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

        if self.long_time_low_profit_minutes > 0:
            if trade.is_short:
                max_profit = (open_rate - trade.min_rate) / open_rate - 2 * self.fee
            else:
                max_profit = (trade.max_rate - open_rate) / open_rate - 2 * self.fee

            open_minutes = round((current_time - trade.open_date_utc).total_seconds() / 60, 1)
            if open_minutes > self.long_time_low_profit_minutes:
                if max_profit < self.long_time_low_profit_max and self.long_time_low_profit_lower_bound < _current_profit < self.long_time_low_profit_upper_bound:
                    return "longtime_low_profit"

        return None

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

        has_open_orders = any(order.status == "open" and not order.ft_is_open and not order.ft_order_side == 'stoploss' for order in trade.orders)
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
            if is_short:
                addition_signal = current_rate < last_entry_price * (1 + factor * self.addition_profit_step)
            elif not is_short:
                addition_signal = current_rate > last_entry_price * (1 + factor * self.addition_profit_step)

        if addition_signal:
            if min_stake <= addition_stake <= max_stake:
                logger.warning(f'Position addition #{count_of_orders+1} for {trade.pair} with estimated new_profit:{new_open_profit*leverage:.2%}'
                               f'(without leverage:{new_open_profit:.2%}) and stake amount {addition_stake:.5f}, '
                               f'current_profit:{current_profit:.2%}, current_rate:{current_rate:.5f}, '
                               f'open_rate:{trade.open_rate:.5f} at {current_time}')
                return (addition_stake / leverage, f'entry-addition')
            else:
                logger.warning(f'Skip position addition for {trade.pair} with estimated new_profit:{new_open_profit*leverage:.2%}'
                               f'(without leverage:{new_open_profit:.2%}) and stake amount {addition_stake:.5f} is out of range'
                               f'({min_stake:.2f}-{max_stake:.2f}), current_profit:{current_profit:.2%}, current_rate:{current_rate:.5f}, '
                               f'open_rate:{trade.open_rate:.5f} at {current_time}')

        return None

    def get_last_candle(self, trade: Trade):
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        return last_candle

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

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        return self.trade_leverage

    @property
    def protections(self):
        protections = []

        if self.cooldown_candles > 0:
            protections.append(
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": self.cooldown_candles,
                }
            )

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
