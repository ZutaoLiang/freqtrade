import math
import numpy as np
import pandas_ta as pta
import pandas as pd
from pandas import DataFrame
from functools import reduce

from datetime import datetime, timedelta
from typing import Optional, Tuple, Union, Callable
from dataclasses import dataclass
from functools import wraps

from freqtrade.constants import Config
from freqtrade.strategy.interface import IStrategy, Trade
from freqtrade.strategy.strategy_helper import stoploss_from_absolute, stoploss_from_open
from freqtrade.strategy import IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib

import logging
logger = logging.getLogger(__name__)


@dataclass
class Factor:
    weight: float
    calc_score: Callable[[pd.DataFrame], pd.Series]


class FactorAnalyzer:
    def __init__(self):
        self.factors: Dict[str, Dict[str, Factor]] = {}
        
    def add_factor(self, dimension: str, name: str, weight: float, 
                  calc_score: Callable[[pd.DataFrame], pd.Series]):
        if dimension not in self.factors:
            self.factors[dimension] = {}
            
        self.factors[dimension][name] = Factor(weight=weight, calc_score=calc_score)        
    
    def _normalize_weights(self, dimension: str):
        factors = self.factors[dimension]
        total_weight = sum(f.weight for f in factors.values())
        if total_weight != 1.0:
            for f in factors.values():
                f.weight = f.weight / total_weight
    
    def analyze(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for dimension, factors in self.factors.items():
            self._normalize_weights(dimension)
            dimension_score = pd.Series(0, index=dataframe.index)
            
            for name, factor in factors.items():
                factor_score = factor.calc_score(dataframe)
                dataframe[f'{dimension}_{name}_score'] = factor_score
                dimension_score += factor_score * factor.weight
            
            dataframe[f'{dimension}_factor_score'] = dimension_score
            
        return dataframe


class StakePositionManager(IStrategy):
    """Base class for stake and position management"""
        
    # 常量
    MAX_PROFIT_ABS = 'max_profit_abs'
    HIGH_PROFIT = 'high_profit'
    
    # 官方变量，不常改
    minimal_roi = {"0": 100}
    trailing_stop = False
    use_custom_stoploss = True
    can_short = True
    position_adjustment_enable = True
    
    # 自定义变量，不常改
    enable_heikinashi = False

    # 自定义变量，可微调
    timeframe = '5m'
    trade_leverage = 10
    base_stoploss_pct = 0.075
    stoploss = -base_stoploss_pct * trade_leverage
    entry_stake_ratio = 1
    addition_stake_ratio = 0.75
    addition_profit_step = 0.025
    exit_loss_ratio = -0.2
    atr_length = 15
    atr_entry_stoploss_multiplier = 5               # 进场时基于open_rate的止损ATR倍数
    atr_entry_base_multiplier = 3.0                 # 进场时的价格ATR倍数，即当前价格超过成本价的这个ATR倍数进场
    atr_addition_base_multiplier = 2.5              # 加仓时的价格ATR倍数，即当前价格超过成本价的这个ATR倍数加仓
    atr_addition_stoploss_base_multiplier = 0.25    # 加仓后的价格ATR止损倍数，即加仓后不足成本价的这个ATR倍数止损

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # self.factor_analyzer = FactorAnalyzer()
 
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return self.trade_leverage
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        leverage = trade.leverage
        _current_profit = current_profit / leverage
        
        if (current_time - timedelta(hours=18)) > trade.open_date_utc and 0.018 < _current_profit < 0.025:
            filled_orders = trade.select_filled_orders()
            count_of_orders = len(filled_orders)
            if count_of_orders < 3:
                return 'Long time low profit'
        
        return False
    
    def atr_addition_multiplier(self, count_of_orders: int) -> float:
        return self.atr_addition_base_multiplier + (count_of_orders-1)

    def atr_addition_stoploss_multiplier(self, count_of_orders: int) -> float:
        return self.atr_addition_stoploss_base_multiplier + (count_of_orders-1) / 10

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, after_fill: bool,
                       **kwargs) -> Optional[float]:
        leverage = trade.leverage
        is_short = trade.is_short
        factor = -1 if is_short else 1
        open_rate = trade.open_rate
        _current_profit = current_profit / leverage

        filled_orders = trade.select_filled_orders()
        count_of_orders = len(filled_orders)
        
        if after_fill:
            if count_of_orders > 1:
                if count_of_orders == 2 or _current_profit < 0.05:
                    return stoploss_from_absolute(open_rate*(1+factor*self.addition_profit_step/2), current_rate, is_short, leverage)
                
        # if count_of_orders > 1:
        #     last_addition_price = filled_orders[-1].average
        #     if is_short and current_rate < last_addition_price * (1-0.01):
        #         return stoploss_from_absolute(last_addition_price*(1-factor*0.005), current_rate, is_short, leverage)
        #     if not is_short and current_rate > last_addition_price * (1+0.01):
        #         return stoploss_from_absolute(last_addition_price*(1-factor*0.005), current_rate, is_short, leverage)

        if _current_profit < 0:
            if _current_profit < -0.02:
                last_candle = self.get_last_candle(trade)
                if is_short:
                    if last_candle['enter_long'] == 1:
                        return 0.04 * leverage
                else:
                    if last_candle['enter_short'] == 1:
                        return 0.04 * leverage
            
            return None

        # if _current_profit > 0.25:
        #     return (_current_profit / 2) * leverage

        # if _current_profit > 0.18:
        #     return stoploss_from_absolute(open_rate*(1+factor*0.1), current_rate, is_short, leverage)
        
        if _current_profit > 0.10:
            return (_current_profit / 2) * leverage

        # if _current_profit > 0.10:
        #     return stoploss_from_absolute(open_rate*(1+factor*0.05), current_rate, is_short, leverage)

        # if _current_profit > 0.05:
        #     return (_current_profit / 2) * leverage
        
        if _current_profit > 0.05:
            return stoploss_from_absolute(open_rate*(1+factor*0.025), current_rate, is_short, leverage)

        if _current_profit > 0.03:
            return stoploss_from_absolute(open_rate*(1+factor*0.015), current_rate, is_short, leverage)
        
        if (current_time - timedelta(hours=4)) > trade.open_date_utc:
            if 0.01 < _current_profit < 0.015:
                # 持仓时间已经不短了，并且方向曾经有对过。后面如果方向不对了，就不等到最大止损再出局，只到一部分损失就提前止损
                return stoploss_from_absolute(open_rate*(1-factor*0.03), current_rate, is_short, leverage)
        
        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float, min_stake: float, max_stake: float, 
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        if not self.position_adjustment_enable:
            return None
        
        has_open_orders = any(order.status == "open" and not order.ft_is_open for order in trade.orders)
        if has_open_orders:
            logger.info(f'There are open orders for {trade.pair}, skip position adjustment.')
            return None
        
        filled_orders = trade.select_filled_orders()
        count_of_orders = len(filled_orders)
        if count_of_orders == 0:
            return None
        
        is_short = trade.is_short
        leverage = trade.leverage
        open_rate = trade.open_rate
        _current_profit = current_profit / leverage
        
        entry_side_orders = [order for order in filled_orders
                             if order.ft_order_side == trade.entry_side and ('entry' in order.ft_order_tag)]
        count_of_orders = len(entry_side_orders)

        addition_stake = self.get_entry_stake_without_leverage() * leverage * self.addition_stake_ratio
        min_stake_threshold = 0.75 * min_stake
        if addition_stake < min_stake_threshold:
            logger.info(f'Skip position addition for {trade.pair} while stake amount:{addition_stake:.5f} is smaller than threshold:{min_stake_threshold:.5f}(min_stake:{min_stake:.5f}) at {current_time}')
            return None

        addition_amount = round(addition_stake / current_rate)
        addition_stake = addition_amount * current_rate
        if min_stake_threshold < addition_stake < min_stake:
            logger.info(f'Adjusting {trade.pair} addition stake:{addition_stake:.5f} to around min_stake:{min_stake:.5f} at {current_time}')
            addition_amount = math.ceil(min_stake / current_rate)
            addition_stake = addition_amount * current_rate
            
        if addition_amount <= 0:
            return None
        
        new_open_rate = (trade.amount * trade.open_rate + addition_stake) / (trade.amount + addition_amount)
        
        factor = -1 if is_short else 1
        new_open_profit = factor * (current_rate / new_open_rate - 1)

        last_candle = self.get_last_candle(trade)
        atr = last_candle['atr']

        addition_signal = False
        if is_short and last_candle['enter_short'] == 1 and new_open_profit > self.addition_profit_step * (count_of_orders):
            addition_signal = True
        elif not is_short and last_candle['enter_long'] == 1 and new_open_profit > self.addition_profit_step * (count_of_orders):
            addition_signal = True

        if addition_signal:
            logger.info(f'Initialize {trade.pair} addition stake to {addition_stake:.5f}(open rate:{open_rate:.6f}, [new_open_rate:{new_open_rate:.6f}], atr:{atr:.6f}, addition amount:{addition_amount:.2f}) '
                    f'at current_rate:{current_rate:.5f} with profit:{current_profit:.2%}({_current_profit:.2%}) at {current_time}')
            
            logger.info(f'Position addition #{count_of_orders} for {trade.pair} with stake amount {addition_stake:.5f} triggered at addition signal, current_profit:{current_profit:.2f}, current_rate:{current_rate:.5f} at {current_time}')
            return (addition_stake / leverage, f'entry-addition')
        
        return None
        
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        stake_amount = min(proposed_stake * self.entry_stake_ratio, max_stake)
        logger.info(f'Stake amount for {pair}={stake_amount:.5f} with leverage:{leverage}(after leverage={stake_amount*leverage:.5f}), '
                    f'proposed:{proposed_stake:.5f}, min_stake:{min_stake:.5f}, max_stake:{max_stake:.5f}, '
                    f'entry_tag:{entry_tag}, current_rate:{current_rate:.5f} at {current_time}')
        return stake_amount
    
    @property
    def protections(self):
        return  [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2
            }
        ]
        
    # @property
    # def protections(self): # type: ignore
    #     return [
    #         {
    #             "method": "StoplossGuard",
    #             "lookback_period_candles": 36,
    #             "trade_limit": 2,
    #             "stop_duration_candles": 6,
    #             "required_profit": 0.0,
    #             "only_per_pair": False,
    #             "only_per_side": True
    #         }
    #     ]

    def get_last_candle(self, trade: Trade):
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        return last_candle
        
    def get_entry_stake_without_leverage(self) -> float:
        return self.wallets.get_total_stake_amount() * self.entry_stake_ratio / self.max_open_trades
    
    def heikinashi(self, dataframe: DataFrame) -> DataFrame:
        if self.enable_heikinashi:
            ha = qtpylib.heikinashi(dataframe)
        else:
            ha = dataframe
            
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.heikinashi(dataframe)
        dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_length, talib=False)
        dataframe['addition_plus_atr'] = dataframe['ha_close'] + (self.atr_entry_base_multiplier * dataframe['atr'])
        dataframe['addition_minus_atr'] = dataframe['ha_close'] - (self.atr_entry_base_multiplier * dataframe['atr'])
        dataframe['stoploss_plus_atr'] = dataframe['ha_close'] + (self.atr_entry_stoploss_multiplier * dataframe['atr'])
        dataframe['stoploss_minus_atr'] = dataframe['ha_close'] - (self.atr_entry_stoploss_multiplier * dataframe['atr'])
        dataframe['atr_pct'] = 100 * dataframe['atr'] / dataframe['ha_close']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # self.factor_analyzer.analyze(dataframe)

        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['addition'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
    
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
    

class Dream3V4ManualStrategy(StakePositionManager):
    """Trading strategy implementation"""
    
    is_long = True
    bidirectional = True
    
    # Strategy parameters
    period = 10
    ema_length = period
    ema_mid_length = 6 * period
    ema_long_length = 12 * period
    ema_trend = 4
    ema_mid_trend = ema_trend
    ema_long_trend = ema_trend
    
    breakout_period = 4
    
    rumi_multiplier = 150
    low_atr_pct = 2.5

    adx_length = period
    adx_threshold = 32
    rsi_length = period
    rsi_long_threshold = 55
    rsi_short_threshold = 30
    
    startup_candle_count = int(ema_long_length)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        
        dataframe['ema'] = pta.ema(close=dataframe['ha_close'], length=self.ema_length, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['ha_close'], length=self.ema_mid_length, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ha_close'], length=self.ema_long_length, talib=False)
        dataframe['ema_long_plus_atr'] = dataframe['ema_long'] + self.atr_addition_base_multiplier * dataframe['atr']
        dataframe['ema_long_minus_atr'] = dataframe['ema_long'] - self.atr_addition_base_multiplier * dataframe['atr']
        dataframe['recent_high'] = dataframe['ha_close'].rolling(window=self.breakout_period).max()
        dataframe['recent_low'] = dataframe['ha_close'].rolling(window=self.breakout_period).min()
        dataframe['adx'] = pta.adx(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.adx_length)[f'ADX_{self.adx_length}']
        dataframe['rsi'] = pta.rsi(dataframe['ha_close'], length=self.rsi_length, talib=False)

        dataframe['rumi_fast'] = pta.sma(dataframe['ha_close'], length=self.ema_length)
        dataframe['rumi_slow'] = pta.wma(dataframe['ha_close'], length=self.ema_mid_length)
        dataframe['rumi'] = pta.sma(dataframe['rumi_fast'] - dataframe['rumi_slow'], length=self.ema_length)

        dataframe['rumi_long_slow'] = pta.wma(dataframe['ha_close'], length=self.ema_long_length)
        dataframe['rumi_long'] = pta.sma(dataframe['rumi_fast'] - dataframe['rumi_long_slow'], length=self.ema_length)
        
        # # long base
        # self.factor_analyzer.add_factor('long_entry_base', 'close_above_ema_long', 2, 
        #     lambda df: (dataframe['ha_close'] > dataframe['ema_long_plus_atr'])
        # )
        # self.factor_analyzer.add_factor('long_entry_base', 'close_above_ema', 1, 
        #     lambda df: (dataframe['ha_close'] > dataframe['ema'])
        # )
        # self.factor_analyzer.add_factor('long_entry_base', 'ema_mid_up', 1, 
        #     lambda df: (self.indicator_up_n_periods_mask(dataframe, 'ema_mid', self.ema_mid_trend))
        # )
        
        # # long
        # self.factor_analyzer.add_factor('long_entry', 'ema_up', 1, 
        #     lambda df: (self.indicator_up_n_periods_mask(dataframe, 'ema', self.ema_trend))
        # )
        # self.factor_analyzer.add_factor('long_entry', 'rumi_up', 1, 
        #     lambda df: (self.indicator_up_n_periods_mask(dataframe, 'rumi', self.ema_trend))
        # )
        # self.factor_analyzer.add_factor('long_entry', 'breakout', 1, 
        #     lambda df: (dataframe['ha_close'] > dataframe['recent_high'].shift(1))
        # )
        # self.factor_analyzer.add_factor('long_entry', 'ema_above_mid', 1, 
        #     lambda df: (dataframe['ema'] > dataframe['ema_mid'])
        # )
        # self.factor_analyzer.add_factor('long_entry', 'rumi_close', 1, 
        #     lambda df: (dataframe['rumi'] * self.rumi_multiplier > dataframe['ha_close'])
        # )
        # # self.factor_analyzer.add_factor('long_entry', 'low_atr_pct', 1, 
        # #     lambda df: (dataframe['atr_pct'] < self.low_atr_pct)
        # # )
        # # self.factor_analyzer.add_factor('long_entry', 'rsi', 1, 
        # #     lambda df: (df['rsi'] > self.rsi_long_threshold)
        # # )
        
        # # short base
        # self.factor_analyzer.add_factor('short_entry_base', 'close_below_ema_long', 2, 
        #     lambda df: (dataframe['ha_close'] < dataframe['ema_long_minus_atr'])
        # )
        # self.factor_analyzer.add_factor('short_entry_base', 'close_below_ema', 1, 
        #     lambda df: (dataframe['ha_close'] < dataframe['ema'])
        # )
        # self.factor_analyzer.add_factor('short_entry_base', 'ema_mid_up', 1, 
        #     lambda df: (self.indicator_down_n_periods_mask(dataframe, 'ema_mid', self.ema_mid_trend))
        # )
        
        # # short
        # self.factor_analyzer.add_factor('short_entry', 'ema_down', 1, 
        #     lambda df: (self.indicator_down_n_periods_mask(dataframe, 'ema', self.ema_trend))
        # )
        # self.factor_analyzer.add_factor('short_entry', 'rumi_down', 1, 
        #     lambda df: (self.indicator_down_n_periods_mask(dataframe, 'rumi', self.ema_trend))
        # )
        # self.factor_analyzer.add_factor('short_entry', 'breakout', 1, 
        #     lambda df: (dataframe['ha_close'] < dataframe['recent_low'].shift(1))
        # )
        # self.factor_analyzer.add_factor('short_entry', 'ema_below_mid', 1, 
        #     lambda df: (dataframe['ema'] < dataframe['ema_mid'])
        # )
        # self.factor_analyzer.add_factor('short_entry', 'rumi_close', 1, 
        #     lambda df: (dataframe['rumi'] * -self.rumi_multiplier > dataframe['ha_close'])
        # )
        # # self.factor_analyzer.add_factor('short_entry', 'low_atr_pct', 1, 
        # #     lambda df: (dataframe['atr_pct'] < self.low_atr_pct)
        # # )
        # # self.factor_analyzer.add_factor('short_entry', 'rsi', 1, 
        # #     lambda df: (df['rsi'] < self.rsi_short_threshold)
        # # )
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        return dataframe
        
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_exit_trend(dataframe, metadata)
        return dataframe


class Dream3V4Strategy(Dream3V4ManualStrategy):
        
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        
        if self.bidirectional or self.is_long:
            dataframe.loc[
                    (
                        # entry base
                        (dataframe['ha_close'] > dataframe['ema_long_plus_atr'])
                        & (dataframe['ha_close'] > dataframe['ema'])
                        & (self.indicator_up_n_periods_mask(dataframe, 'ema_mid', self.ema_mid_trend))
                        # entry
                        & (self.indicator_up_n_periods_mask(dataframe, 'ema', self.ema_trend))
                        & (self.indicator_up_n_periods_mask(dataframe, 'rumi', self.ema_trend))
                        & (dataframe['ha_close'] > dataframe['recent_high'].shift(1))
                        & (dataframe['ema'] > dataframe['ema_mid'])
                        & (dataframe['rumi'] * self.rumi_multiplier > dataframe['ha_close'])
                        & (dataframe['atr_pct'] < self.low_atr_pct)
                    ),
                    ['enter_long', 'enter_tag']] = (1, 'entry_long')
            
            # dataframe.loc[
            #         (
            #             (dataframe['long_entry_base_factor_score'] >= 0.9)
            #             & (dataframe['long_entry_factor_score'] >= 0.9)
            #         ),
            #         ['enter_long', 'enter_tag']] = (1, 'entry_long')
        
        if self.bidirectional or not self.is_long:
            dataframe.loc[
                    (
                        # entry base
                        (dataframe['ha_close'] < dataframe['ema_long_minus_atr'])
                        & (dataframe['ha_close'] < dataframe['ema'])
                        & (self.indicator_down_n_periods_mask(dataframe, 'ema_mid', self.ema_mid_trend))
                        
                        # entry
                        & (self.indicator_down_n_periods_mask(dataframe, 'ema', self.ema_trend))
                        & (self.indicator_down_n_periods_mask(dataframe, 'rumi', self.ema_trend))
                        & (dataframe['ha_close'] < dataframe['recent_low'].shift(1))
                        & (dataframe['ema'] < dataframe['ema_mid'])
                        & (dataframe['rumi'] * -self.rumi_multiplier > dataframe['ha_close'])
                        & (dataframe['atr_pct'] < self.low_atr_pct)
                    ),
                    ['enter_short', 'enter_tag']] = (1, 'entry_short')
            
            # dataframe.loc[
            #         (
            #             (dataframe['short_entry_base_factor_score'] >= 0.9)
            #             & (dataframe['short_entry_factor_score'] >= 0.9)
            #         ),
            #         ['enter_short', 'enter_tag']] = (1, 'entry_short')
            
        return dataframe
        
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_exit_trend(dataframe, metadata)
        return dataframe
