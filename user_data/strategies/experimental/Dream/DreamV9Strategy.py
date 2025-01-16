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


class DreamV9Strategy(IStrategy):
    
    # common
    trade_leverage = 5
    
    enable_roi = False
    ignore_roi_if_entry_signal = True
    if enable_roi:
        minimal_roi = {"60": 0.04 * trade_leverage, "120": 0.06 * trade_leverage}
    else:
        minimal_roi = {"0": 100}

    timeframe = '3m'
    
    base_stoploss_pct = 0.08
    stoploss = -base_stoploss_pct * trade_leverage
    trailing_stop = False
    use_custom_stoploss = True

    can_short = True
 
    position_adjustment_enable = True
    stake_ratio = 0.25
    order_interval_seconds = 50
    addition_price_pct = 0.015
    
    period = 10
    
    ema_length = period
    ema_mid_length = 6 * period
    ema_long_length = 24 * period
    ema_trend = 6
    ema_mid_trend = ema_trend
    ema_long_trend = ema_trend * 3
    ema_up_ratio = 1.005
    
    breakout_period = 4
    
    adx_length = period
    adx_threshold = 25
    rsi_length = period
    rsi_long_threshold = 55
    rsi_short_threshold = 30
    
    startup_candle_count = ema_long_length
    
    exit_loss_ratio = -0.2

    is_long = True
    
    # atr_length = int(1.5 * period)
    
    enable_mean_reversion = False # default to False
    mean_reversion_change_pct = 0.005
    
    LAST_ADDITION_PRICE = 'last_addition_price'
    MAX_PROFIT_ABS = 'max_profit_abs'
    HIGH_PROFIT = 'high_profit'

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return self.trade_leverage
         
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        leverage = trade.leverage
        entry_stake = self.calc_entry_stake_without_leverage()
        entry_stake_with_leverage = entry_stake * leverage
        stake_amount = trade.amount * trade.open_rate
        
        # # 长时间stake amount上不来，例如不到相当于4次加仓的市值，说明加仓少，趋势不明朗，尽早退出
        # low_stake_threshold = 4 * entry_stake_with_leverage
        # is_low_stake = stake_amount < low_stake_threshold
        # logger.info(f'Checking {trade.pair} low stake result:{is_low_stake}, stake_amount:{stake_amount:.4f}, threshold:{low_stake_threshold:.4f}, current_profit:{current_profit:.2%} at {current_time}')
        # if is_low_stake:
        #     if (current_time - timedelta(minutes=180)) > trade.open_date_utc and 0.002 * leverage < current_profit < 0.02 * leverage:
        #         exit_reason = 'Long time low profit'
        #         logger.info(f'{exit_reason} for pair:{trade.pair}, current_rate:{current_rate:.6f}, open_rate:{trade.open_rate:.6f}, current_profit:{current_profit:.2%}, stake_amount:{stake_amount:.4f} at {current_time}')
        #         return exit_reason
        #     if (current_time - timedelta(minutes=300)) > trade.open_date_utc and -0.01 * leverage < current_profit < 0.002 * leverage:
        #         exit_reason = 'Long time low profit-2'
        #         logger.info(f'{exit_reason} for pair:{trade.pair}, current_rate:{current_rate:.6f}, open_rate:{trade.open_rate:.6f}, current_profit:{current_profit:.2%}, stake_amount:{stake_amount:.4f} at {current_time}')
        #         return exit_reason
        #     if (current_time - timedelta(minutes=480)) > trade.open_date_utc and current_profit < 0.01 * leverage:
        #         exit_reason = 'Long time low stake'
        #         logger.info(f'{exit_reason} for pair:{trade.pair}, current_rate:{current_rate:.6f}, open_rate:{trade.open_rate:.6f}, current_profit:{current_profit:.2%}, stake_amount:{stake_amount:.4f} at {current_time}')
        #         return exit_reason
        
        open_profit_abs = current_profit / leverage * stake_amount
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
        
        market_value_threshold_array = [entry_stake_with_leverage, entry_stake_with_leverage * 0.5]
        # draw_back_ratio_array = [1-(0.1*leverage), 1-(0.12*leverage), 1-(0.18*leverage)]
        draw_back_ratio_array = [0.6, 0.5]
        
        for index, (market_value_threshold, draw_back_ratio) in enumerate(zip(market_value_threshold_array, draw_back_ratio_array)):
            reach_profit = max_profit_abs > market_value_threshold
            if index == 0:
                trade.set_custom_data(self.HIGH_PROFIT, reach_profit)
                
            reach_drawback = reach_profit and total_profit_abs < max_profit_abs * draw_back_ratio
            logger.info(f'Checking {trade.pair} drawback result:{reach_drawback} on threshold #{index+1}:{market_value_threshold:.4f} and total_profit_abs:{total_profit_abs:.4f} with {max_profit_abs*draw_back_ratio:.4f}(max_profit_abs:{max_profit_abs:.4f}*ratio:{draw_back_ratio:.2%}) at {current_time}')
            if reach_drawback:
                exit_reason = f'Profit drawback-{index+1}'
                logger.info(f'{exit_reason} for {pair}: total profit {total_profit_abs:.4f} < (max_profit_abs {max_profit_abs:.4f} * {draw_back_ratio:.2%}), current_rate:{current_rate:.4f} at {current_time}')
                return exit_reason
        
        profit_drawdown_threshold = entry_stake_with_leverage * self.exit_loss_ratio
        reach_max_loss = total_profit_abs < profit_drawdown_threshold
        logger.info(f'Checking {trade.pair} max loss result:{reach_max_loss}, total_profit_abs:{total_profit_abs:.4f}, threshold:{profit_drawdown_threshold:.4f}=(entry_stake_with_leverage:{entry_stake_with_leverage:.4f}*exit_loss_ratio:{self.exit_loss_ratio:.2%}), current_profit:{current_profit:.2%} at {current_time}')
        if reach_max_loss:
            exit_reason = 'Max loss'
            logger.info(f'{exit_reason} for {pair}:{total_profit_abs:.4f} < {profit_drawdown_threshold:.4f}, current_rate:{current_rate:.6f} at {current_time}')
            return exit_reason
        
        # 出现反转信号并且利润比较低时退出
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        if last_candle['reverse_signal'] == 1 and current_profit < 0.01 * leverage:
            exit_reason = f'reverse|{current_profit:.1f}'
            logger.info(f'{exit_reason} for pair:{trade.pair}, current_rate:{current_rate:.6f}, open_rate:{trade.open_rate:.6f}, current_profit:{current_profit:.2%}, stake_amount:{stake_amount:.4f} at {current_time}')
            return exit_reason
        
        return False

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        leverage = trade.leverage

        if after_fill:
            return self.stoploss
        
        # profit_pct = current_profit / leverage
        # if profit_pct >= 0.25:
        #     return 0.1 * leverage
        # elif profit_pct >= 0.15:
        #     return 0.07 * leverage
        # elif profit_pct >= 0.08:
        #     return 0.05 * leverage
            
        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float, min_stake: float, max_stake: float, 
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        # Note: 这个函数需要返回的是不带杠杆的金额，具体代码参考freqtradebot.execute_entry()
        if not self.position_adjustment_enable:
            return None

        filled_orders = trade.select_filled_orders()
        count_of_orders = len(filled_orders)
        if count_of_orders == 0:
            return None
        
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        is_short = trade.is_short
        leverage = trade.leverage
        
        latest_entry_order = None
        for order in reversed(filled_orders):
            if order.ft_order_side == trade.entry_side:
                latest_entry_order = order
                break
            
        if latest_entry_order is None:
            return None

        # TODO: 根据加仓订单的密集程度动态调整加仓比例
        # filled_or_open_orders = self.select_filled_or_open_orders()
        # orders_json = [order.to_json(self.entry_side, minified) for order in filled_or_open_orders]
        
        # order_json = latest_order.to_json(trade.entry_side, True)
        # logger.info(f'{trade.pair} latest order:{order_json}')
        
        # 处理是否需要浮盈加仓
        last_addition_price = trade.get_custom_data(self.LAST_ADDITION_PRICE)
        if last_addition_price is None:
            last_addition_price = latest_entry_order.average
            trade.set_custom_data(self.LAST_ADDITION_PRICE, last_addition_price)
        
        addition_signal = False
        # match_order_interval = (current_time - timedelta(seconds=self.order_interval_seconds)) > latest_entry_order.order_filled_utc
        price_change_pct = (current_rate - last_addition_price) / last_addition_price
        if price_change_pct > self.addition_price_pct:
            # and match_order_interval
            # addition_price_ratio = 0.98
            if is_short and last_candle['enter_short'] == 1:
                # and current_rate < last_addition_price / addition_price_ratio
                addition_signal = True
            elif not is_short and last_candle['enter_long'] == 1:
                # and current_rate > last_addition_price * addition_price_ratio
                addition_signal = True
                
        min_stake /= trade.leverage
        max_stake /= trade.leverage
        
        entry_stake = self.calc_entry_stake_without_leverage()
        if addition_signal:
            base_profit_step = 0.2
            profit_factor = max(min(current_profit, base_profit_step * 3), base_profit_step)
            addition_multiplier = int(round(profit_factor / base_profit_step))
            addition_stake = entry_stake * addition_multiplier
            logger.info(f'Initialize {trade.pair} addition stake #{addition_multiplier} to {addition_stake:.5f} at {current_time}')
            
            position_addition = True
            if addition_stake < min_stake:
                if addition_stake < 0.3 * min_stake:
                    logger.info(f'Skip position addition for {trade.pair} while stake amount:{addition_stake:.5f} is smaller than min_stake:{min_stake:.5f} at {current_time}')
                    position_addition = False
                else:
                    logger.info(f'Adjusting {trade.pair} addition stake:{addition_stake:.5f} to min_stake:{min_stake:.5f} at {current_time}')
                    addition_stake = min_stake
            
            if position_addition:
                trade.set_custom_data(self.LAST_ADDITION_PRICE, current_rate)
                
                logger.info(f'Position addition for {trade.pair} with stake amount {addition_stake:.5f}(multiplier:#{addition_multiplier}) triggered at entry signal, current_profit:{current_profit:.2f}, current_rate:{current_rate:.5f} at {current_time}')
                return (addition_stake, f'entry-addition-{addition_multiplier}')

        if not self.enable_mean_reversion:
            return None

        # 均值回归处理
        last_reversion_price = filled_orders[-1].average
        
        price_change = (last_reversion_price - current_rate) / last_reversion_price
        if abs(price_change) < self.mean_reversion_change_pct:
            return None
        
        current_market_value = trade.amount * current_rate
        
        # 达到单个品种的标准市值（即总资金/max_open_trades）才考虑均值回归处理，未达到之前等待趋势加仓或者趋势没起来打到止损
        if current_market_value < entry_stake * leverage:
            return None
        
        reach_high_profit = trade.get_custom_data(self.HIGH_PROFIT, default=None)
        if reach_high_profit is None:
            reach_high_profit = False
            
        if current_profit < 0.1 and not reach_high_profit:
            return None
        
        # last_market_value = trade.amount * last_reversion_price
        # market_value_change = last_market_value - current_market_value
        # reversion_stake = market_value_change / leverage
        
        reversion_direction = 1
        if trade.is_short:
            reversion_direction = -1 if price_change > 0 else 1
        else:
            reversion_direction = 1 if price_change > 0 else -1
        
        reversion_stake = entry_stake * reversion_direction / leverage
        logger.info(f'Initialize reversion stake for {trade.pair} with stake amount:{reversion_stake:.4f}(amount:{trade.amount}, last_price:{last_reversion_price:.4f}, current_rate:{current_rate:.4f}, price_change:{price_change:.2%})')
        
        if abs(reversion_stake) > min_stake:
            # and abs(adjustment_value) < max_stake:
            logger.info(f'Mean reversion adjustment for {trade.pair} with stake amount:{reversion_stake:.4f}(amount:{trade.amount}, last_price:{last_reversion_price:.4f}, current_rate:{current_rate:.4f}, price_change:{price_change:.2%}) at {current_time}')
            if reversion_stake > 0:
                return (reversion_stake, 'reversion-addition')
            else:
                return (reversion_stake, 'reversion-decrease')
        else:
            logger.info(f'Skip mean reversion adjustment for {trade.pair} while stake amount:{reversion_stake:.4f} is not in the valid range({min_stake:.4f}, {max_stake:.4f}) at {current_time}')
            return None
        
    def calc_entry_stake_without_leverage(self) -> float:
        return self.wallets.get_total(self.stake_currency) * self.stake_ratio / self.max_open_trades
        
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        # Note: 这个函数需要返回的是不带杠杆的金额，具体代码参考freqtradebot.execute_entry()
        stake_amount = min(max(min_stake, proposed_stake * self.stake_ratio), max_stake)
        logger.info(f'Stake amount for {pair}={stake_amount:.5f} with leverage:{leverage}(after leverage={stake_amount*leverage:.5f}), proposed:{proposed_stake:.5f}, min_stake:{min_stake:.5f}, max_stake:{max_stake:.5f}, rate:{current_rate:.5f} at {current_time}')
        return stake_amount

    def heikinashi(self, dataframe: DataFrame) -> DataFrame:
        # ha = qtpylib.heikinashi(dataframe)
        ha = dataframe
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']
        return dataframe
       
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.heikinashi(dataframe)
        
        dataframe['ema'] = pta.ema(close=dataframe['ha_close'], length=self.ema_length, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['ha_close'], length=self.ema_mid_length, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ha_close'], length=self.ema_long_length, talib=False)
        dataframe['recent_high'] = dataframe['ha_close'].rolling(window=self.breakout_period).max()
        dataframe['recent_low'] = dataframe['ha_close'].rolling(window=self.breakout_period).min()
        # dataframe['adx'] = pta.adx(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.adx_length)[f'ADX_{self.adx_length}']
        # dataframe['rsi'] = pta.rsi(dataframe['ha_close'], length=self.rsi_length, talib=False)
        # dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_length)
         
        return dataframe
        
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
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        if self.is_long:
            ema_up_mask = self.ema_up_n_days_mask(dataframe, 'ema', self.ema_trend)
            ema_mid_up_mask = self.ema_up_n_days_mask(dataframe, 'ema_mid', self.ema_mid_trend)
            ema_long_up_mask = self.ema_up_n_days_mask(dataframe, 'ema_long', self.ema_long_trend)
        
            dataframe.loc[
                    (
                        (dataframe['ha_close'] > self.ema_up_ratio * dataframe['ema_long']) &
                        (ema_up_mask) &
                        (ema_mid_up_mask) &
                        (ema_long_up_mask) &
                        (dataframe['ema'] > dataframe['ema_mid']) &
                        (dataframe['ema_mid'] > dataframe['ema_long']) &
                        (dataframe['ha_close'] > dataframe['recent_high'].shift(1))
                        # (dataframe['rsi'] > self.rsi_long_threshold) & 
                        # (dataframe['adx'] > self.adx_threshold)
                    ),
                    ['enter_long', 'enter_tag']] = (1, 'entry-long')
        else:
            ema_down_mask = self.ema_down_n_days_mask(dataframe, 'ema', self.ema_trend)
            ema_mid_down_mask = self.ema_down_n_days_mask(dataframe, 'ema_mid', self.ema_mid_trend)
            ema_long_down_mask = self.ema_down_n_days_mask(dataframe, 'ema_long', self.ema_long_trend)
            
            dataframe.loc[
                    (
                        (dataframe['ha_close'] * self.ema_up_ratio < dataframe['ema_long']) &
                        (ema_down_mask) &
                        (ema_mid_down_mask) &
                        (ema_long_down_mask) &
                        (dataframe['ema'] < dataframe['ema_mid']) &
                        (dataframe['ema_mid'] < dataframe['ema_long']) &
                        (dataframe['ha_close'] < dataframe['recent_low'].shift(1))
                        # (dataframe['rsi'] < self.rsi_short_threshold) & 
                        # (dataframe['adx'] > self.adx_threshold)
                    ),
                    ['enter_short', 'enter_tag']] = (1, 'entry-short')
            
        return dataframe
        
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        if self.is_long:
            ema_mid_down_mask = self.ema_down_n_days_mask(dataframe, 'ema_mid', self.ema_mid_length)
            dataframe.loc[
                    (
                        (
                            (dataframe['ha_close'] < dataframe['ema_mid']) | (dataframe['ha_close'] < dataframe['ema_long'])
                        )
                        & (ema_mid_down_mask)
                        & (dataframe['ha_close'] < self.ema_up_ratio * dataframe['ema_long'])
                    ), 'reverse_signal'] = 1
            
        return dataframe
