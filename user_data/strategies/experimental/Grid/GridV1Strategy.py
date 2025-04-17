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
from freqtrade.strategy import merge_informative_pair, IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib

import logging
logger = logging.getLogger(__name__)


class StakePositionManager(IStrategy):
    """Base class for stake and position management"""

    trade_leverage = 4
    
    # 官方变量，不常改
    minimal_roi = {"0": 100}
    trailing_stop = False
    use_custom_stoploss = True
    can_short = True
    position_adjustment_enable = False
    
    # 自定义变量，不常改
    enable_heikinashi = True

    # 自定义变量，可微调
    timeframe = '5m'
    
    base_stoploss_pct = 0.1
    stoploss = -base_stoploss_pct * trade_leverage
    
    entry_stake_ratio = 0.25
    
    addition_stake_ratio = 0.25
    addition_min_new_profit = 0.15
    addition_price_atr_multiplier = 10
    
    # addition_profit_step = 0.01
    # exit_loss_ratio = -0.2
    atr_length = 15

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return self.trade_leverage
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        return False
        
        filled_orders = trade.select_filled_orders()
        if not filled_orders:
            return False
        
        leverage = trade.leverage
        open_rate = trade.open_rate
        _current_profit = current_profit / leverage
    
        entry_side_orders = [order for order in filled_orders
                            if order.ft_order_side == trade.entry_side and ('entry' in order.ft_order_tag)]

        last_entry_order = filled_orders[-1]
        last_entry_order_timestamp = last_entry_order.order_filled_utc
    
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        df_after_last_entry_order = dataframe[dataframe['date'] >= pd.to_datetime(last_entry_order_timestamp)]
        
        if df_after_last_entry_order.empty:
            return False
    
        count_of_entry_orders = len(entry_side_orders)
        
        if count_of_entry_orders <= 3:
            # if (current_time - timedelta(hours=8)) > trade.open_date_utc:
            #     profit_spread = 0.03
            #     drawback_ratio = 0.5
            # else:
            #     return False
            
            return False
        else:
            profit_spread = 0.02
            drawback_ratio = 0.5
            
        if trade.is_short:
            max_profit_rate = df_after_last_entry_order.loc[df_after_last_entry_order['low'].idxmin()]['low']
            if max_profit_rate < (1 - profit_spread) * open_rate:
                drawback_rate = max_profit_rate + (open_rate - max_profit_rate) * drawback_ratio
                if current_rate > drawback_rate:
                    logger.info(f'{pair} short drawback reach rate:{drawback_rate:.6f} with max_profit_rate:{max_profit_rate:.6f} at {current_time}')
                    return f'Drawback-{count_of_entry_orders}'
        else:
            max_profit_rate = df_after_last_entry_order.loc[df_after_last_entry_order['high'].idxmax()]['high']
            if max_profit_rate > (1 + profit_spread) * open_rate:
                drawback_rate = max_profit_rate - (max_profit_rate - open_rate) * drawback_ratio
                if current_rate < drawback_rate:
                    logger.info(f'{pair} long drawback reach rate:{drawback_rate:.6f} with max_profit_rate:{max_profit_rate:.6f} at {current_time}')
                    return f'Drawback-{count_of_entry_orders}'
    
        # if (current_time - timedelta(hours=36)) > trade.open_date_utc and self.addition_min_new_profit*0.5 < _current_profit < self.addition_min_new_profit*0.75:
        #     filled_orders = trade.select_filled_orders()
        #     count_of_orders = len(filled_orders)
        #     if count_of_orders < 2:
        #         return 'Long time low profit'
        
        return False
    
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

        last_candle = self.get_last_candle(trade)
        atr = last_candle['atr']
        
        # if after_fill:
        #     if count_of_orders > 1:
        #         if count_of_orders <= 3:
        #             stoploss_price = filled_orders[-2].average
        #         elif count_of_orders <= 5:
        #             stoploss_price = filled_orders[-3].average
        #         else:
        #             stoploss_price = filled_orders[-4].average
                    
        #         logger.info(f'{pair} after fill new profit:{_current_profit:.2%}, stoploss_price:{stoploss_price:.6f}, current_rate:{current_rate:.6f} at {current_time}')
        #         return stoploss_from_absolute(stoploss_price, current_rate, is_short, leverage)
       
        # if trade.is_short:
        #     atr_multiplier = self.atr_short_multiplier
        #     # if _current_profit < self.addition_min_new_profit / 4:
        #     #     atr_multiplier *= 0.5
        #     chandelier_stop = last_candle['chandelier_exit_low'] + atr_multiplier * last_candle['atr']
        #     if chandelier_stop > current_rate:
        #         return stoploss_from_absolute(chandelier_stop, current_rate, is_short, leverage)
        # else:
        #     atr_multiplier = self.atr_long_multiplier
        #     # if _current_profit < self.addition_min_new_profit / 4:
        #     #     atr_multiplier *= 0.5
        #     chandelier_stop = last_candle['chandelier_exit_high'] - atr_multiplier * last_candle['atr']
        #     if chandelier_stop < current_rate:
        #         return stoploss_from_absolute(chandelier_stop, current_rate, is_short, leverage)
        
        # if _current_profit > 0.25:
        #     return (_current_profit / 2) * leverage

        # if _current_profit > 0.18:
        #     return stoploss_from_absolute(open_rate*(1+factor*0.1), current_rate, is_short, leverage)
        
        # if _current_profit > 0.20:
        #     return (_current_profit / 2) * leverage

        # # if _current_profit > 0.10:
        # #     return stoploss_from_absolute(open_rate*(1+factor*0.05), current_rate, is_short, leverage)

        if _current_profit > 0.1:
            return (_current_profit * 0.5) * leverage
        
        # if _current_profit > 0.06:
        #     return (_current_profit / 2) * leverage
        
        # # if _current_profit > 0.07:
        # #     return stoploss_from_absolute(open_rate*(1+factor*0.05), current_rate, is_short, leverage)

        # if _current_profit > 0.05:
        #     return stoploss_from_absolute(open_rate*(1+factor*0.03), current_rate, is_short, leverage)

        # if _current_profit > 0.04:
        #     return stoploss_from_absolute(open_rate*(1+factor*0.02), current_rate, is_short, leverage)

        # if _current_profit > 0.03:
        #     return stoploss_from_absolute(open_rate*(1+factor*0.015), current_rate, is_short, leverage)
        
        if (current_time - timedelta(hours=2)) > trade.open_date_utc:
            if _current_profit > 0.05:
                return (_current_profit * 0.75) * leverage
        elif (current_time - timedelta(hours=3)) > trade.open_date_utc:
            if _current_profit > 0.03:
                return (_current_profit * 0.6) * leverage
        elif (current_time - timedelta(hours=4)) > trade.open_date_utc:
            if _current_profit > 0.02:
                return (_current_profit * 0.5) * leverage
        elif (current_time - timedelta(hours=6)) > trade.open_date_utc:
            if _current_profit > 0:
                return 0.01 * leverage
            if _current_profit > -0.02:
                return 0.01 * leverage
            
            # if 0.01 < _current_profit < 0.015:
            #     # 持仓时间已经不短了，并且方向曾经有对过。后面如果方向不对了，就不等到最大止损再出局，只到一部分损失就提前止损
            #     return stoploss_from_absolute(open_rate*(1-factor*0.03), current_rate, is_short, leverage)
        
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
        
        # todo: 多次加仓后的减仓逻辑，每次退到上一个加仓位置之后进行一次减仓，比如30%

        addition_stake = self.get_entry_stake_without_leverage() * leverage * self.addition_stake_ratio
        min_stake_threshold = 0.75 * min_stake
        if addition_stake < min_stake_threshold:
            logger.info(f'Skip position addition for {trade.pair} while stake amount:{addition_stake:.5f} is smaller than threshold:{min_stake_threshold:.5f}(min_stake:{min_stake:.5f}) at {current_time}')
            return None

        addition_amount = round(addition_stake / current_rate)
        addition_stake = addition_amount * current_rate
        if min_stake_threshold < addition_stake < min_stake:
            logger.debug(f'Adjusting {trade.pair} addition stake:{addition_stake:.5f} to around min_stake:{min_stake:.5f} at {current_time}')
            addition_amount = math.ceil(min_stake / current_rate)
            addition_stake = addition_amount * current_rate
            
        if addition_amount <= 0:
            return None
        
        new_open_rate = (trade.amount * trade.open_rate + addition_stake) / (trade.amount + addition_amount)
        
        factor = -1 if is_short else 1
        new_open_profit = factor * (current_rate / new_open_rate - 1) - (0.0005 * 2)

        last_candle = self.get_last_candle(trade)
        atr = last_candle['atr']

        addition_signal = False
        enough_profit = new_open_profit > self.addition_min_new_profit
        last_entry_price = entry_side_orders[-1].average
        if is_short and last_candle['addition'] == 1 and enough_profit:
            addition_signal = current_rate < last_entry_price + factor * self.addition_price_atr_multiplier * atr
        elif not is_short and last_candle['addition'] == 1 and enough_profit:
            addition_signal = current_rate > last_entry_price + factor * self.addition_price_atr_multiplier * atr

        if addition_signal:
            logger.info(f'Initialize {trade.pair} addition stake to {addition_stake:.5f}(open rate:{open_rate:.6f}, [new_open_rate:{new_open_rate:.6f}], atr:{atr:.6f}, addition amount:{addition_amount:.2f})'
                    f'at current_rate:{current_rate:.5f} with profit:{current_profit:.2%}({_current_profit:.2%}) at {current_time}')
            
            logger.info(f'Position addition #{count_of_orders} for {trade.pair} with estimated new_profit:{new_open_profit:.2%} and stake amount {addition_stake:.5f} triggered at addition signal, current_profit:{current_profit:.2f}, current_rate:{current_rate:.5f} at {current_time}')
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
                "stop_duration_candles": 12
            }
        ]
        
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
        dataframe['chandelier_exit_high'] = dataframe['ha_close'].rolling(window=self.atr_length).max()
        dataframe['chandelier_exit_low'] = dataframe['ha_close'].rolling(window=self.atr_length).min()
        
        # dataframe['chandelier_exit_long'] = dataframe['chandelier_exit_high'] - self.atr_long_multiplier * dataframe['atr']
        # dataframe['chandelier_exit_short'] = dataframe['chandelier_exit_low'] + self.atr_long_multiplier * dataframe['atr']
        
        # dataframe['addition_plus_atr'] = dataframe['ha_close'] + (self.atr_addition_base_multiplier * dataframe['atr'])
        # dataframe['addition_minus_atr'] = dataframe['ha_close'] - (self.atr_addition_base_multiplier * dataframe['atr'])
        # dataframe['stoploss_plus_atr'] = dataframe['ha_close'] + (self.atr_entry_stoploss_multiplier * dataframe['atr'])
        # dataframe['stoploss_minus_atr'] = dataframe['ha_close'] - (self.atr_entry_stoploss_multiplier * dataframe['atr'])
        dataframe['atr_pct'] = 100 * dataframe['atr'] / dataframe['ha_close']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
    

class GridV1Strategy(StakePositionManager):
    
    is_long = True
    bidirectional = True
 
    # Strategy parameters
    period = 12
    
    ema_short = period
    ema_mid = ema_short * 20
    ema_long = ema_mid * 4
    
    ema_trend = 10
    
    cci_length = 240
    cci_c = 0.015
    
    startup_candle_count = int(ema_long)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        
        dataframe['ema_short'] = pta.ema(close=dataframe['ha_close'], length=self.ema_short, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['ha_close'], length=self.ema_mid, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ha_close'], length=self.ema_long, talib=False)
        
        dataframe['cci'] = pta.cci(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], length=self.cci_length, c=self.cci_c, talib=False)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        if self.bidirectional or self.is_long:
            addition_mask = (dataframe['ema_short'] > dataframe['ema_mid']) 
            dataframe.loc[addition_mask, 'addition'] = 1
                        
            dataframe.loc[(
                addition_mask
                & (dataframe['ha_close'] > dataframe['ema_short'])
                & (self.indicator_up_n_periods_mask(dataframe, 'ema_short', self.ema_trend)) 
                & (self.indicator_up_n_periods_mask(dataframe, 'ema_mid', self.ema_trend))
                & (self.indicator_up_n_periods_mask(dataframe, 'ema_long', 2 * self.ema_trend))
                & (dataframe['cci'] > 75)
                ), ['enter_long', 'enter_tag']] = (1, 'entry_long')
        return dataframe
        
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_exit_trend(dataframe, metadata)
        # if self.bidirectional or self.is_long:
            # dataframe.loc[(
            #     (dataframe['cci'] < -30)
            #     ), ['exit_long', 'exit_tag']] = (1, 'exit_long_cci')
        
            # dataframe.loc[(
            #     (dataframe['cci'] < 0) & ((dataframe['ema_short'] < dataframe['ema_mid']) | (dataframe['ha_close'] < dataframe['ema_mid']))
            #     ), ['exit_long', 'exit_tag']] = (1, 'exit_long_ema')
        
        return dataframe

