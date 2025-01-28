import numpy as np
import pandas_ta as pta
import pandas as pd
from pandas import DataFrame
from functools import reduce

from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

from freqtrade.strategy.interface import IStrategy, Trade
from freqtrade.strategy.strategy_helper import stoploss_from_absolute, stoploss_from_open
from freqtrade.strategy import IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib

import logging
logger = logging.getLogger(__name__)


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
    trade_leverage = 5
    base_stoploss_pct = 0.07
    stoploss = -base_stoploss_pct * trade_leverage
    stake_ratio = 0.25
    addition_price_pct = 0.005
    exit_loss_ratio = -0.25
    atr_length = 15
    atr_stoploss_multiplier = int(6)

        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return self.trade_leverage
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        leverage = trade.leverage
        entry_stake = self.get_entry_stake_without_leverage()
        entry_stake_with_leverage = entry_stake * leverage
        stake_amount = trade.amount * trade.open_rate
        _current_profit = current_profit / leverage
        
        open_profit_abs = _current_profit * stake_amount
        realized_profit_abs = trade.realized_profit if trade.realized_profit else 0
        total_profit_abs = realized_profit_abs + open_profit_abs
        
        max_profit_abs = trade.get_custom_data(self.MAX_PROFIT_ABS)
        if max_profit_abs is None:
            max_profit_abs = total_profit_abs
            trade.set_custom_data(self.MAX_PROFIT_ABS, max_profit_abs)
        elif total_profit_abs > max_profit_abs:
            logger.info(f'New max profit for {trade.pair}, from {max_profit_abs:.4f} to {total_profit_abs:.4f}, current_rate:{current_rate:.6f} at {current_time}')
            max_profit_abs = total_profit_abs
            trade.set_custom_data(self.MAX_PROFIT_ABS, max_profit_abs)
        
        logger.info(f'{trade.pair} total profit:{total_profit_abs:.4f}(open:{open_profit_abs:.4f}, close:{realized_profit_abs:.4f}), current_rate:{current_rate:.6f}, open_rate:{trade.open_rate:.6f}, current_profit:{current_profit:.2%}, stake_amount:{stake_amount:.4f} at {current_time}')

        market_value_threshold_array = [entry_stake_with_leverage, entry_stake_with_leverage * 0.5, entry_stake_with_leverage * 0.2]
        draw_back_ratio_array = [0.75, 0.65, 0.2]
        
        for index, (market_value_threshold, draw_back_ratio) in enumerate(zip(market_value_threshold_array, draw_back_ratio_array)):
            reach_profit = max_profit_abs > market_value_threshold
            if index == 0:
                trade.set_custom_data(self.HIGH_PROFIT, reach_profit)
                
            reach_drawback = reach_profit and total_profit_abs < max_profit_abs * draw_back_ratio
            logger.debug(f'Checking {trade.pair} drawback result:{reach_drawback} on threshold #{index+1}:{market_value_threshold:.4f}(market_value) and total_profit_abs:{total_profit_abs:.4f} vs {max_profit_abs*draw_back_ratio:.4f}(max_profit_abs:{max_profit_abs:.4f}*ratio:{draw_back_ratio:.2%}) at {current_time}')
            if reach_drawback:
                exit_reason = f'Profit drawback-{index+1}'
                logger.info(f'{exit_reason} for {pair}: total profit {total_profit_abs:.4f} < (max_profit_abs {max_profit_abs:.4f} * {draw_back_ratio:.2%}), current_rate:{current_rate:.4f} at {current_time}')
                return exit_reason
        
        profit_drawdown_threshold = entry_stake_with_leverage * self.exit_loss_ratio
        reach_max_loss = total_profit_abs < profit_drawdown_threshold
        logger.debug(f'Checking {trade.pair} max loss result:{reach_max_loss}, total_profit_abs:{total_profit_abs:.4f}, threshold:{profit_drawdown_threshold:.4f}=(entry_stake_with_leverage:{entry_stake_with_leverage:.4f}*exit_loss_ratio:{self.exit_loss_ratio:.2%}), current_profit:{current_profit:.2%} at {current_time}')
        if reach_max_loss:
            exit_reason = 'Max loss'
            logger.info(f'{exit_reason} for {pair}:{total_profit_abs:.4f} < {profit_drawdown_threshold:.4f}, current_rate:{current_rate:.6f} at {current_time}')
            return exit_reason
        
        return False

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, after_fill: bool,
                       **kwargs) -> Optional[float]:
        leverage = trade.leverage
        is_short = trade.is_short
        open_rate = trade.open_rate
        _current_profit = current_profit / leverage

        if after_fill:
            filled_orders = trade.select_filled_orders()
            count_of_orders = len(filled_orders)
            last_candle = self.get_last_candle(trade)
            factor = 1 if is_short else -1
            stop_rate_atr = filled_orders[-1].average + (factor * self.atr_stoploss_multiplier * last_candle['atr'])
        
            if count_of_orders <= 1:
                logger.info(f'Setting {trade.pair} #{count_of_orders} after fill stoploss rate atr to:{stop_rate_atr:.6f}, open_rate:{open_rate:.6f}(stop/open ratio:{abs(stop_rate_atr/open_rate-1):.2%}), current_rate:{current_rate:.6f}, current_profit:{current_profit:.2%}(without leverage:{current_profit/leverage:.2%}) at {current_time}')
                return stoploss_from_absolute(stop_rate_atr, current_rate, is_short, leverage)
            
            if _current_profit < 0.015:
                relative_factor = 0.005
            elif _current_profit < 0.03:
                relative_factor = 0.0075
            elif _current_profit < 0.06:
                relative_factor = 0.01
            elif _current_profit < 0.06:
                relative_factor = 0.02
            elif _current_profit < 0.06:
                relative_factor = 0.03
            else:
                return self.stoploss
                
            if is_short:
                relative_factor *= -1
                stop_rate = min(stop_rate_atr, open_rate*(1+relative_factor))
            else:
                stop_rate = max(stop_rate_atr, open_rate*(1+relative_factor))
            
            logger.info(f'Setting {trade.pair} #{count_of_orders} after fill stoploss rate to:{stop_rate:.6f}, open_rate:{open_rate:.6f}(stop/open ratio:{abs(stop_rate/open_rate-1):.2%}), current_rate:{current_rate:.6f}, current_profit:{current_profit:.2%}(without leverage:{_current_profit:.2%}) at {current_time}')
            
            return stoploss_from_absolute(stop_rate, current_rate, is_short, leverage)
        
        filled_orders = trade.select_filled_orders()
        count_of_orders = len(filled_orders)
        addition_count_array = [2, 4]
        holding_minutes_array = [90, 240]
        for index, (addition_count_threshold, holding_minutes) in enumerate(zip(addition_count_array, holding_minutes_array)):
            if count_of_orders < addition_count_threshold and (current_time - timedelta(minutes=holding_minutes)) > trade.open_date_utc:
                if 0.006 < _current_profit < 0.015:
                    relative_factor = -1 if is_short else 1
                    stop_rate = open_rate * (1 + relative_factor * (_current_profit-0.001))
                    
                    logger.info(f'{trade.pair} filled count:{count_of_orders}(threshold:{addition_count_threshold}) for {holding_minutes}mins, '
                                f'current_profit:{current_profit:.2%} at {current_time}')
                    logger.info(f'Stoploss for pair:{trade.pair} as long time low profit, current_profit:{current_profit:.2%}(without leverage:{_current_profit:.2%}), '
                                f'stoploss rate:{stop_rate:.6f}, open_rate:{open_rate:.6f}(stop/open ratio:{abs(stop_rate/open_rate-1):.2%}), current_rate:{current_rate:.6f}, at {current_time}')
                    return stoploss_from_absolute(stop_rate, current_rate, is_short, leverage)
        
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
        _current_profit = current_profit / leverage
        
        entry_side_orders = [order for order in filled_orders
                             if order.ft_order_side == trade.entry_side and ('entry' in order.ft_order_tag)]
        count_of_orders = len(entry_side_orders)
        count_of_addition_orders = count_of_orders - 1
        entry_order = entry_side_orders[0]
        entry_order_price = entry_order.average
        
        price_direction = 1
        if is_short:
            price_direction = -1
            
        price_threshold = entry_order_price * (1 + price_direction * len(entry_side_orders) * self.addition_price_pct)
        
        last_candle = self.get_last_candle(trade)
        
        addition_signal = False
        profit_signal = False
        
        if is_short:
            addition_signal = current_rate <= price_threshold and last_candle['close'] <= price_threshold \
                and last_candle['addition'] == 1
        else:
            addition_signal = current_rate >= price_threshold and last_candle['close'] >= price_threshold \
                and last_candle['addition'] == 1
        
        if count_of_addition_orders < 1:
            profit_signal = _current_profit > 0.02
        else:
            last_addition_price = entry_side_orders[-1].average
            price_change_pct = (current_rate - last_addition_price) / last_addition_price
            if is_short:
                price_change_pct *= -1
            
            profit_signal = (_current_profit > min(6, count_of_addition_orders + 1) * 0.01) and (price_change_pct > 0)

        min_stake /= trade.leverage
        max_stake /= trade.leverage
        entry_stake = self.get_entry_stake_without_leverage()
        if addition_signal and profit_signal:
            base_profit_step = 0.1
            profit_factor = max(min(_current_profit, base_profit_step * 3), base_profit_step)
            addition_multiplier = int(round(profit_factor / base_profit_step))
            addition_stake = entry_stake * 0.5 * addition_multiplier
            logger.info(f'Initialize {trade.pair} addition stake #{addition_multiplier} to {addition_stake:.5f} at {current_time}')
            
            position_addition = True
            if addition_stake < min_stake:
                if addition_stake < 0.75 * min_stake:
                    logger.info(f'Skip position addition for {trade.pair} while stake amount:{addition_stake:.5f} is smaller than min_stake:{min_stake:.5f} at {current_time}')
                    position_addition = False
                else:
                    logger.info(f'Adjusting {trade.pair} addition stake:{addition_stake:.5f} to min_stake:{min_stake:.5f} at {current_time}')
                    addition_stake = min_stake
            
            if position_addition:
                logger.info(f'Position addition for {trade.pair} with stake amount {addition_stake:.5f}(multiplier:#{addition_multiplier}) triggered at entry signal, current_profit:{current_profit:.2f}, current_rate:{current_rate:.5f} at {current_time}')
                return (addition_stake, f'entry-addition-{addition_multiplier}')
        
        return None
        
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        stake_amount = min(proposed_stake * self.stake_ratio, max_stake)
        logger.info(f'Stake amount for {pair}={stake_amount:.5f} with leverage:{leverage}(after leverage={stake_amount*leverage:.5f}), proposed:{proposed_stake:.5f}, min_stake:{min_stake:.5f}, max_stake:{max_stake:.5f}, rate:{current_rate:.5f} at {current_time}')
        return stake_amount

    def get_last_candle(self, trade: Trade):
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        return last_candle
        
    def get_entry_stake_without_leverage(self) -> float:
        return self.wallets.get_total_stake_amount() * self.stake_ratio / self.max_open_trades
    
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
        dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_length)
        dataframe[f'close_plus_atr'] = dataframe['ha_close'] + self.atr_stoploss_multiplier * dataframe['atr']
        dataframe[f'close_minus_atr'] = dataframe['ha_close'] - self.atr_stoploss_multiplier * dataframe['atr']
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
    

class Dream2V2ManualStrategy(StakePositionManager):
    """Trading strategy implementation"""
    
    timeframe = '3m'
    is_long = False
    
    # Strategy parameters
    period = 10
    ema_length = period
    ema_mid_length = 9 * period
    ema_long_length = 24 * period
    ema_trend = 6
    ema_mid_trend = ema_trend
    ema_long_trend = ema_trend * 3
    ema_dist_ratio = 1.02
    
    breakout_period = 4
    
    adx_length = period
    adx_threshold = 40
    rsi_length = period
    rsi_long_threshold = 55
    rsi_short_threshold = 30
    
    startup_candle_count = int(ema_long_length)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        
        dataframe['ema'] = pta.ema(close=dataframe['ha_close'], length=self.ema_length, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['ha_close'], length=self.ema_mid_length, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ha_close'], length=self.ema_long_length, talib=False)
        dataframe['recent_high'] = dataframe['ha_close'].rolling(window=self.breakout_period).max()
        dataframe['recent_low'] = dataframe['ha_close'].rolling(window=self.breakout_period).min()
        dataframe['adx'] = pta.adx(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.adx_length)[f'ADX_{self.adx_length}']
        dataframe['rsi'] = pta.rsi(dataframe['ha_close'], length=self.rsi_length, talib=False)
        
        return dataframe
        

class Dream2V2Strategy(Dream2V2ManualStrategy):
        
    def ema_up_n_days_mask(self, dataframe: DataFrame, ema: str, days: int):
        ema_up_mask = (dataframe[f'{ema}'] > dataframe[f'{ema}'].shift(1))
        for i in range(2, days):
            ema_up_mask = ema_up_mask & (dataframe[f'{ema}'].shift(i-1) > dataframe[f'{ema}'].shift(i))
        return ema_up_mask
    
    def ema_down_n_days_mask(self, dataframe: DataFrame, ema: str, days: int):
        ema_down_mask = (dataframe[f'{ema}'] < dataframe[f'{ema}'].shift(1))
        for i in range(2, days):
            ema_down_mask = ema_down_mask & (dataframe[f'{ema}'].shift(i-1) < dataframe[f'{ema}'].shift(i))
        return ema_down_mask
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        
        if self.is_long:
            ema_up_mask = self.ema_up_n_days_mask(dataframe, 'ema', self.ema_trend)
            ema_mid_up_mask = self.ema_up_n_days_mask(dataframe, 'ema_mid', self.ema_mid_trend)
            ema_long_up_mask = self.ema_up_n_days_mask(dataframe, 'ema_long', self.ema_long_trend)
        
            addition_trend_mask = (dataframe['ha_close'] > self.ema_dist_ratio * dataframe['ema_long']) \
                                    & (dataframe['ha_close'] > dataframe['ema']) \
                                    & (ema_long_up_mask)
                                    
            dataframe.loc[addition_trend_mask, 'addition'] = 1
            
            dataframe.loc[
                    (
                        addition_trend_mask
                        & (ema_up_mask)
                        & (ema_mid_up_mask)
                        & (dataframe['ema'] > dataframe['ema_mid'])
                        & (dataframe['ha_close'] > dataframe['recent_high'].shift(1))
                        & (dataframe['adx'] > self.adx_threshold)
                    ),
                    ['enter_long', 'enter_tag']] = (1, 'entry_long')
        else:
            ema_down_mask = self.ema_down_n_days_mask(dataframe, 'ema', self.ema_trend)
            ema_mid_down_mask = self.ema_down_n_days_mask(dataframe, 'ema_mid', self.ema_mid_trend)
            ema_long_down_mask = self.ema_down_n_days_mask(dataframe, 'ema_long', self.ema_long_trend)
            
            addition_trend_mask = (dataframe['ha_close'] * self.ema_dist_ratio < dataframe['ema_long']) \
                        & (dataframe['ha_close'] < dataframe['ema']) \
                        & (ema_long_down_mask)
            
            dataframe.loc[addition_trend_mask, 'addition'] = 1
            
            dataframe.loc[
                    (
                        addition_trend_mask 
                        & (ema_down_mask) 
                        & (ema_mid_down_mask) 
                        & (dataframe['ema'] < dataframe['ema_mid']) 
                        & (dataframe['ha_close'] < dataframe['recent_low'].shift(1))
                        & (dataframe['adx'] > self.adx_threshold)
                    ),
                    ['enter_short', 'enter_tag']] = (1, 'entry_short')
            
        return dataframe
        
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_exit_trend(dataframe, metadata)
        return dataframe
