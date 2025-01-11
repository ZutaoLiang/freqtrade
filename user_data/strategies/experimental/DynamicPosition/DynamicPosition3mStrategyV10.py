from math import isnan
import numpy as np
import talib.abstract as ta
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

from freqtrade.rpc.rpc import RPC

import logging
logger = logging.getLogger(__name__)


class TotalProfitDrawdownExit:
    def __init__(self):
        self.pair_profits = {}
        self.LOW_PROFIT = -100
        self.max_total_profit = self.LOW_PROFIT
        self.open_total_profit = self.LOW_PROFIT
        self.closed_total_profit = 0
        self.global_exit_triggered = False
        self.processed_pairs = set()
        self.PROFIT_CHECK_THRESHOLD = 20
        self.TOTAL_DRAWDOWN_THRESHOLD = 0.15
        self.PAIR_DRAWDOWN_THRESHOLD = 0.03

    def init_pairs(self, pairs: set, closed_total_profit: float) -> None:
        if self.global_exit_triggered and self.pair_profits and self.processed_pairs == set(self.pair_profits.keys()):
            current_total = sum(self.pair_profits.values())
            self.max_total_profit = current_total + self.closed_total_profit
            logger.info(f'Reset max_total_profit:{self.max_total_profit:.2f} after global exit')
        
        for pair in pairs:
            if pair not in self.pair_profits.keys():
                logger.info(f'Initial profit 0 for {pair}')
                self.pair_profits[pair] = 0.0
            
        for pair in list(self.pair_profits.keys()):
            if pair not in pairs:
                logger.info(f'Remove {pair} from profit dict')
                self.pair_profits.pop(pair)
                
        self.processed_pairs.clear()
        self.open_total_profit = self.LOW_PROFIT
        self.closed_total_profit = closed_total_profit
        self.global_exit_triggered = False
    
    def update_and_check_exit(self, pair: str, open_profit: float, min_rate: float, max_rate: float, current_rate: float, is_short: bool) -> bool:
        first_sum = False
        if pair not in self.pair_profits.keys():
            # 第一次把所有的pair利润初始化好
            first_sum = True
        
        self.pair_profits[pair] = open_profit
        
        if first_sum:
            return False
        
        self.processed_pairs.add(pair)

        if any(profit == 0 for profit in self.pair_profits.values()):
            return False

        self.open_total_profit = sum(self.pair_profits.values())
        total_profit = self.open_total_profit + self.closed_total_profit
        logger.info(f'Current total profit:{total_profit:.4f}, open:{self.open_total_profit:.4f}, closed:{self.closed_total_profit:.4f}')
        if total_profit > self.max_total_profit:
            logger.info(f'Update max_total_profit from {self.max_total_profit:.4f} to {total_profit:.4f}')
            self.max_total_profit = total_profit
        
        if self.max_total_profit <= self.LOW_PROFIT:
            return False
        
        max_abs_profit = max(abs(total_profit), abs(self.max_total_profit))
        if max_abs_profit < self.PROFIT_CHECK_THRESHOLD:
            logger.info(f'Total max abs profit:{max_abs_profit:.2f} is lower than {self.PROFIT_CHECK_THRESHOLD:.2f}, do not check drawdown')
            return False
            
        drawdown = (self.max_total_profit - total_profit) / abs(self.max_total_profit)
        if drawdown >= self.TOTAL_DRAWDOWN_THRESHOLD:
            self.global_exit_triggered = True
            logger.info(f'Total profit large drawdown:{drawdown:.2%}, max_profit:{self.max_total_profit:.4f}, current total:{total_profit:.4f}')
            
            if is_short:
                pair_drawdown = (current_rate - min_rate) / min_rate
            else:
                pair_drawdown = (max_rate - current_rate) / max_rate
                
            large_drawdown = (pair_drawdown >= self.PAIR_DRAWDOWN_THRESHOLD)
            low_profit = open_profit < 0
            if large_drawdown or low_profit:
                logger.info(f'Pair {pair} exit because of pair profit large drawdown:{large_drawdown}(drawdown={pair_drawdown:.2%}) or low_profit:{low_profit}(profit={open_profit:.4f})')
                return True
            
        return False

    
class DynamicPosition3mStrategyV10(IStrategy):
    
    # common
    minimal_roi = {"0": 100}
    
    trade_leverage = 4

    timeframe = '3m'
    
    stoploss = -0.05 * trade_leverage
    trailing_stop = False
    use_custom_stoploss = True

    can_short = True
 
    enable_logging = False
    
    position_adjustment_enable = True
    initial_position_ratio = 1/6
    
    position_adjustment_stake_ratio = 0.9
    
    period = 10
    
    atr_length = int(1.5 * period)
    
    ema_length = period
    ema_mid_length = 6 * period
    ema_long_length = 12 * period
    ema_trend_length = 6
    breakout_period = 5
    
    adx_length = period
    adx_threshold = 30
    rsi_length = period
    rsi_long_threshold = 60
    rsi_short_threshold = 30
    
    order_interval_minutes = 1
    
    startup_candle_count = ema_long_length
    
    is_long = False
    
    absolute_drawdown_profit_ratio = 0.06

    require_balance = False
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.total_profit_drawdown_exit = TotalProfitDrawdownExit()
    
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        open_pairs = {trade.pair for trade in Trade.get_open_trades()}
        closed_total_profit = Trade.get_total_closed_profit()
        self.total_profit_drawdown_exit.init_pairs(open_pairs, closed_total_profit)
        
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        open_profit = current_profit * trade.stake_amount
        drawdown_exit = self.total_profit_drawdown_exit.update_and_check_exit(pair, open_profit, trade.min_rate, trade.max_rate, current_rate, trade.is_short)
        if drawdown_exit:
            return 'Total profit drawdown'
        
        total = self.wallets.get_total(self.stake_currency)
        profit_drawdown_threshold = -(total * self.absolute_drawdown_profit_ratio) / self.max_open_trades
        if open_profit < profit_drawdown_threshold:
            exit_reason = f'Low open profit'
            logger.info(f'{exit_reason}:{open_profit:.2f} < {profit_drawdown_threshold:.2f} for {pair} at {current_time}')
            return exit_reason
        
        return False
        
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        stake_amount = min(max(min_stake, proposed_stake * self.initial_position_ratio), max_stake)
        logger.info(f'Stake amount for {pair}={stake_amount:.5f} with leverage:{leverage}(total={stake_amount*leverage:.5f}), rate:{current_rate:.5f} at {current_time}')
        return stake_amount
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float, min_stake: float, max_stake: float, 
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        if not self.position_adjustment_enable:
            return None

        has_open_orders = any(order.status == "open" and not order.ft_is_open for order in trade.orders)
        if has_open_orders:
            logger.info(f'Skip {trade.pair} position adjustment when there are open orders')
            return None
        
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        is_short = trade.is_short
        
        drawback = 0
        if is_short:
            if trade.min_rate is not None:
                drawback = (current_rate - trade.min_rate) / trade.min_rate
        else:
            if trade.max_rate is not None:
                drawback = (trade.max_rate - current_rate) / trade.max_rate
        large_drawback = drawback > 0.08
        
        exit_signal = False
        break_ema_ratio = 0.96
        if is_short and current_rate > last_candle['ema_mid'] / break_ema_ratio:
            exit_signal = True
            
        if not is_short and current_rate < last_candle['ema_mid'] * break_ema_ratio:
            exit_signal = True
        
        if large_drawback or exit_signal:
            stake_to_decrease = 0
            
            # 统计有盈利的部分先减仓
            for order in trade.select_filled_orders():
                price_percent = (current_rate - order.average) / order.average
                if is_short:
                    price_percent *= -1
                if price_percent >= 0.01:
                    stake_to_decrease -= order.average * order.filled
            
            if stake_to_decrease < -0.001:
                if abs(stake_to_decrease) < abs(min_stake):
                    logger.info(f'Adjusting partial decrease stake {stake_to_decrease:.5f} to {-abs(min_stake):5f} according to min_stake:{min_stake:.5f} for {trade.pair} at {current_time}')
                    stake_to_decrease = -abs(min_stake)
                elif abs(stake_to_decrease) > abs(max_stake):
                    logger.info(f'Adjusting partial decrease stake {stake_to_decrease:.5f} to {-abs(max_stake):5f} according to max_stake:{max_stake:.5f} for {trade.pair} at {current_time}')
                    stake_to_decrease = -abs(max_stake)
                    
                logger.info(f'Position partial decrease for {trade.pair} with amount {stake_to_decrease:.5f} at {current_rate:.5f}, current profit:{current_profit:.2f}. Drawback:{drawback:.2%}, large_drawback={large_drawback}, exit_signal={exit_signal}. {current_time}')
                
                if large_drawback:
                    return (stake_to_decrease, 'large drawback')
                elif exit_signal:
                    return (stake_to_decrease, 'partial exit')

        filled_entries = trade.select_filled_orders()
        count_of_entries = len(filled_entries)
        if count_of_entries == 0:
            return None
       
        latest_order = trade.select_order(order_side=trade.entry_side, is_open=False, only_filled=True)
        if latest_order is None:
            return None
        
        entry_signal = False

        addition_price_offset = 0.97
        if is_short and last_candle['enter_short'] == 1 and current_profit > 0 and current_rate < latest_order.average / addition_price_offset \
            and (current_time - timedelta(minutes=self.order_interval_minutes)) > latest_order.order_filled_utc:
            entry_signal = True

        if not is_short and last_candle['enter_long'] == 1 and current_profit > 0 and current_rate > latest_order.average * addition_price_offset \
            and (current_time - timedelta(minutes=self.order_interval_minutes)) > latest_order.order_filled_utc:
            entry_signal = True
        
        if entry_signal:
            stake_ratio = pow(self.position_adjustment_stake_ratio, len(filled_entries))
            first_stake = filled_entries[0].stake_amount
            stake_to_increase = stake_ratio * first_stake
            
            logger.info(f'Set {trade.pair} stake to increase to {stake_to_increase:.5f} with stake_ratio:{stake_ratio:.2f} for #{len(filled_entries)} based on first stake:{first_stake:.5f} at {current_time}')
            
            if stake_to_increase < min_stake:
                if stake_to_increase < 0.3 * min_stake:
                    logger.info(f'Try to increase {trade.pair} position while amount:{stake_to_increase:.5f} is smaller than min_stake:{min_stake:.5f} at {current_time}')
                    return None
                else:
                    logger.info(f'Adjusting {trade.pair} increase stake:{stake_to_increase:.5f} to min_stake:{min_stake:.5f} at {current_time}')
                    stake_to_increase = min_stake
            
            available_balance = self.wallets.get_available_stake_amount()
            if available_balance < stake_to_increase:
                if current_profit > 0.05 * trade.leverage:
                    self.require_balance = True
                    logger.info(f'Required addition stake:{stake_to_increase:.5f} for {trade.pair} is greater than available:{available_balance:.5f} at {current_time}')
                return None
            else:
                self.require_balance = False
                    
            logger.info(f'Position addition #{count_of_entries+1} for {trade.pair} with amount {stake_to_increase:.5f} triggered at entry signal, profit:{current_profit:.2f} at {current_time}')
            return (stake_to_increase, 'addition')
                    
        return None
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        filled_count = len(trade.select_filled_orders())
        
        leverage = trade.leverage
        if not after_fill:
            if self.require_balance and (0.002 * leverage) < current_profit and current_profit < (0.03 * leverage) \
                and (current_time - timedelta(minutes=30)) > trade.open_date_utc:
                new_stoploss = 0.002 * leverage
                abs_rate = current_rate*(1 + (1 if trade.is_short else -1) * new_stoploss/trade.leverage)
                logger.info(f'Set {pair} low profit stoploss rate:{abs_rate:.5f}({new_stoploss:.2%}) while other pair require balance at {current_time}')
                return new_stoploss
            
            if (0.01 * leverage) < current_profit and current_profit < (0.03 * leverage) and (current_time - timedelta(hours=4)) > trade.open_date_utc:
                new_stoploss = 0.002 * leverage
                abs_rate = current_rate*(1 + (1 if trade.is_short else -1) * new_stoploss/trade.leverage)
                logger.info(f'Set {pair} long time low profit stoploss rate:{abs_rate:.5f}({new_stoploss:.2%}) with profit:{current_profit:.2%} at {current_time}')
                return new_stoploss
            
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return None

            last_candle = dataframe.loc[dataframe['date'] <= current_time]
            if last_candle.empty:
                return None
            
            min_rate = trade.min_rate if trade.min_rate is not None else trade.open_rate
            max_rate = trade.max_rate if trade.max_rate is not None else trade.open_rate
            
            last_candle = last_candle.iloc[-1]
            atr = last_candle['atr']
            
            profit_multiplier = 5
            if current_profit > 0.03 * leverage:
                profit_multiplier = 4
            
            stop_loss_price = None
            if trade.is_short:
                stop_loss_price = min_rate + (profit_multiplier * atr)
            else:
                stop_loss_price = max_rate - (profit_multiplier * atr)

            return stoploss_from_absolute(stop_rate=stop_loss_price, current_rate=current_rate, is_short=trade.is_short,
                                          leverage=trade.leverage)
        
        if filled_count <= 1:
            logger.info(f'First fill for {pair}, will not refresh stoploss, current_profit:{current_profit:.2%}, open_rate:{trade.open_rate:.5f}/current_rate:{current_rate:.5f} at {current_time}')
            return None
        
        if current_profit > 0.32 * leverage:
            new_stoploss = 0.08 * leverage
        elif current_profit > 0.16 * leverage:
            new_stoploss = 0.06 * leverage
        elif current_profit > 0.05 * leverage:
            new_stoploss = 0.04 * leverage
        # elif current_profit > 0.02 * leverage:
        #     new_stoploss = 0.03 * leverage
        else:
            return None

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
        dataframe['adx'] = pta.adx(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.adx_length)[f'ADX_{self.adx_length}']
        dataframe['rsi'] = pta.rsi(dataframe['ha_close'], length=self.rsi_length, talib=False)
        dataframe['recent_high'] = dataframe['ha_close'].rolling(window=self.breakout_period).max()
        dataframe['recent_low'] = dataframe['ha_close'].rolling(window=self.breakout_period).min()
        dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_length)
        
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
            ema_up_mask = self.ema_up_n_days_mask(dataframe, 'ema', self.ema_trend_length)
            ema_mid_up_mask = self.ema_up_n_days_mask(dataframe, 'ema_mid', self.ema_trend_length)
            ema_long_up_mask = self.ema_up_n_days_mask(dataframe, 'ema_long', self.ema_trend_length)
            
            dataframe.loc[
                (
                    (dataframe['ha_close'] > dataframe['recent_high'].shift(1)) &
                    (dataframe['ha_close'] > dataframe['ema']) &
                    (dataframe['ema'] > dataframe['ema_mid']) &
                    (ema_up_mask) &
                    (ema_mid_up_mask) &
                    (ema_long_up_mask) &
                    (dataframe['rsi'] > self.rsi_long_threshold) & 
                    (dataframe['adx'] > self.adx_threshold)
                ),
                ['enter_long', 'enter_tag']] = (1, 'entry')
        else:
            ema_down_mask = self.ema_down_n_days_mask(dataframe, 'ema', self.ema_trend_length)
            ema_mid_down_mask = self.ema_down_n_days_mask(dataframe, 'ema_mid', self.ema_trend_length)
            ema_long_down_mask = self.ema_down_n_days_mask(dataframe, 'ema_long', self.ema_trend_length)
            
            dataframe.loc[
                (
                    (dataframe['ha_close'] < dataframe['recent_low'].shift(1)) &
                    (dataframe['ha_close'] < dataframe['ema']) &
                    (dataframe['ema'] < dataframe['ema_mid']) &
                    (ema_down_mask) &
                    (ema_mid_down_mask) &
                    (ema_long_down_mask) &
                    (dataframe['rsi'] < self.rsi_short_threshold) & 
                    (dataframe['adx'] > self.adx_threshold)
                ),
                ['enter_short', 'enter_tag']] = (1, 'entry')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        return dataframe
 
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage
