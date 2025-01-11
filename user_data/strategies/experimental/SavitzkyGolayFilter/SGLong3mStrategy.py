from scipy.signal import savgol_filter
from freqtrade.strategy.interface import IStrategy
import numpy as np
import pandas_ta as pta
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IntParameter, DecimalParameter, informative
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Order, Trade
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
import logging
logger = logging.getLogger(__name__)


class TotalProfitDrawdownExit:
    def __init__(self, leverage):
        self.pair_profits = {}
        self.LOW_PROFIT = -100
        self.max_total_profit = self.LOW_PROFIT
        self.open_total_profit = self.LOW_PROFIT
        self.closed_total_profit = 0
        self.global_exit_triggered = False
        self.processed_pairs = set()
        self.PROFIT_CHECK_THRESHOLD = 20
        self.PAIR_DRAWDOWN_THRESHOLD = 0.04
        self.LARGE_PAIR_DRAWDOWN_THRESHOLD = 0.08
        self.TOTAL_DRAWDOWN_THRESHOLD = self.PAIR_DRAWDOWN_THRESHOLD * 0.5 * leverage

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
    
    def update_and_check_exit(self, pair: str, open_profit: float, min_rate: float, max_rate: float, current_rate: float, is_short: bool, leverage, current_time) -> bool:
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
        logger.debug(f'Current total profit:{total_profit:.4f}, open:{self.open_total_profit:.4f}, closed:{self.closed_total_profit:.4f} at {current_time}')
        if total_profit > self.max_total_profit:
            logger.info(f'Update max_total_profit from {self.max_total_profit:.4f} to {total_profit:.4f} at {current_time}')
            self.max_total_profit = total_profit
        
        if self.max_total_profit <= self.LOW_PROFIT:
            return False
        
        max_abs_profit = max(abs(total_profit), abs(self.max_total_profit))
        if max_abs_profit < self.PROFIT_CHECK_THRESHOLD:
            logger.debug(f'Total max abs profit:{max_abs_profit:.2f} is lower than {self.PROFIT_CHECK_THRESHOLD:.2f}, do not check drawdown at {current_time}')
            return False
            
        drawdown = (self.max_total_profit - total_profit) / abs(self.max_total_profit)
        if drawdown >= self.TOTAL_DRAWDOWN_THRESHOLD:
            self.global_exit_triggered = True
            logger.info(f'Total profit large drawdown:{drawdown:.2%}, max_profit:{self.max_total_profit:.4f}, current total:{total_profit:.4f} at {current_time}')
            
            if is_short:
                pair_drawdown = (current_rate - min_rate) / min_rate
            else:
                pair_drawdown = (max_rate - current_rate) / max_rate
                
            large_drawdown = ((pair_drawdown >= self.PAIR_DRAWDOWN_THRESHOLD) and open_profit > 0) or (pair_drawdown > self.LARGE_PAIR_DRAWDOWN_THRESHOLD)
            # low_profit = 0 < open_profit and open_profit < 0.02 * leverage
            if large_drawdown:
                logger.info(f'Pair {pair} exit because of pair profit large drawdown:{large_drawdown}(drawdown={pair_drawdown:.2%})(profit={open_profit:.4f}) at {current_time}')
                return True
            else:
                logger.info(f'Keep pair {pair} while profit large drawdown:{large_drawdown}(pair drawdown={pair_drawdown:.2%})(profit={open_profit:.4f}) at {current_time}')
            
        return False


class SGLong3mStrategy(IStrategy):
    
    minimal_roi = {"0": 100}

    trade_leverage = 10
    
    timeframe = '3m'
    
    base_stoploss_pct = 0.05
    stoploss = -base_stoploss_pct * trade_leverage
    trailing_stop = False
    use_custom_stoploss = True
    use_custom_exit = True
    
    absolute_drawdown_profit_ratio = 0.1
    
    can_short = True
    
    is_long = True
 
    enable_custom_stake = False
    initial_position_ratio = 1
    
    position_adjustment_enable = False
    position_adjustment_stake_ratio = 0.95
    order_interval_minutes = 1

    period = 10
    
    window_length = IntParameter(10, 100, default=period, space='buy')
    polyorder = IntParameter(1, 5, default=1, space='fixed')

    ema_period = IntParameter(5, 100, default=period, space='buy')
    ema_mid_period = IntParameter(5, 100, default=period * 3, space='buy')

    startup_candle_count = int(max(window_length.value, ema_mid_period.value) * 1.2)
    
    up_ratio = DecimalParameter(1.0001, 1.0010, default=1.0008, decimals=5, space='buy')
    down_ratio = DecimalParameter(1.0001, 1.0010, default=1.0001, decimals=5, space='buy')
    
    highest_period = period
    lowest_period = period

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.total_profit_drawdown_exit = TotalProfitDrawdownExit(self.trade_leverage)
    
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        open_pairs = {trade.pair for trade in Trade.get_open_trades()}
        closed_total_profit = Trade.get_total_closed_profit()
        self.total_profit_drawdown_exit.init_pairs(open_pairs, closed_total_profit)
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        if not self.use_custom_exit:
            return False
        
        leverage = trade.leverage
        filled_entries = trade.select_filled_orders()
        count_of_entries = len(filled_entries)
        if count_of_entries <= 3 and (0.01 * leverage) < current_profit and current_profit < (0.03 * leverage) and (current_time - timedelta(minutes=30)) > trade.open_date_utc:
            return 'Long time low profit'
        
        open_profit = current_profit * trade.stake_amount
        drawdown_exit = self.total_profit_drawdown_exit.update_and_check_exit(pair, open_profit, trade.min_rate, trade.max_rate, current_rate, trade.is_short, leverage, current_time)
        if drawdown_exit:
            return 'Total profit drawdown'
        
        total = self.wallets.get_total(self.stake_currency)
        starting_balance = self.wallets.get_starting_balance()
        
        profit_drawdown_threshold = -0.3 * (total * leverage * self.initial_position_ratio * self.absolute_drawdown_profit_ratio) / self.max_open_trades
        # if total < 0.95 * starting_balance:
        #     logger.info(f'total:{total:.4f}, starting_balance:{starting_balance:.4f}')
        #     profit_drawdown_threshold /= 3
        
        if open_profit < profit_drawdown_threshold:
            exit_reason = f'Low open profit'
            logger.info(f'{exit_reason}:{open_profit:.2f} < {profit_drawdown_threshold:.2f} for {pair} at {current_time}')
            return exit_reason
        
        return False
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        if not self.enable_custom_stake:
            return proposed_stake
        
        stake_amount = min(max(min_stake, proposed_stake * self.initial_position_ratio), max_stake)
        logger.info(f'Stake amount for {pair}={stake_amount:.5f} with leverage:{leverage}(total={stake_amount*leverage:.5f}), rate:{current_rate:.5f} at {current_time}')
        return stake_amount
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float, min_stake: float, max_stake: float, 
                          current_entry_rate: float, current_exit_rate: float,
                          current_entry_profit: float, current_exit_profit: float, **kwargs) -> Optional[float]:
        if not self.position_adjustment_enable:
            return None
        
        filled_entries = trade.select_filled_orders()
        count_of_entries = len(filled_entries)
        if count_of_entries == 0:
            return None
        
        has_open_orders = any(order.status == "open" and not order.ft_is_open for order in trade.orders)
        if has_open_orders:
            logger.info(f'Skip {trade.pair} position adjustment when there are open orders')
            return None
        
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        is_short = trade.is_short
       
        latest_order = trade.select_order(order_side=trade.entry_side, is_open=False, only_filled=True)
        if latest_order is None:
            return None
        
        entry_signal = False

        has_profit = current_profit > 0.0001 * trade.leverage
        match_order_interval = (current_time - timedelta(minutes=self.order_interval_minutes)) > latest_order.order_filled_utc
        if match_order_interval and has_profit:
            addition_price_offset = 0.97
            if is_short and last_candle['enter_short'] == 1 and current_rate < latest_order.average / addition_price_offset:
                entry_signal = True
            elif not is_short and (last_candle['enter_long'] == 1 and current_rate > latest_order.average * addition_price_offset):
                entry_signal = True
        
        if entry_signal:
            stake_to_increase = latest_order.stake_amount * self.position_adjustment_stake_ratio
            logger.info(f'Set {trade.pair} stake to increase to {stake_to_increase:.5f} for #{len(filled_entries)} based on latest stake:{latest_order.stake_amount:.5f} at {current_time}')
            
            min_stake /=  trade.leverage
            if stake_to_increase < min_stake:
                if stake_to_increase < 0.2 * min_stake:
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
        if not self.use_custom_stoploss:
            return None
        
        leverage = trade.leverage
        profit_pct = current_profit / leverage

        if after_fill:
            return 0.07 * leverage
        
        if profit_pct >= 0.25:
            return 0.1 * leverage
        elif profit_pct >= 0.15:
            return 0.07 * leverage
        elif profit_pct >= 0.08:
            return 0.05 * leverage
        elif profit_pct >= 0.05:
            return 0.04 * leverage
            
        return None
    
    def savgol_smooth(self, data):
        smoothed_data = savgol_filter(data, self.window_length.value, self.polyorder.value, mode='nearest')
        return smoothed_data
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['ema'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_period.value, talib=False)
        dataframe['smoothed_ema'] = self.savgol_smooth(dataframe['ema'].values)

        dataframe['ema_mid'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_mid_period.value, talib=False)
        dataframe['smoothed_ema_mid'] = self.savgol_smooth(dataframe['ema_mid'].values)
        
        dataframe['highest'] = dataframe['ohlc4'].rolling(window=self.highest_period).max()
        dataframe['lowest'] = dataframe['ohlc4'].rolling(window=self.lowest_period).min()
        
        dataframe['prev_diff'] = dataframe['smoothed_ema'] / dataframe['smoothed_ema'].shift(1)
        dataframe['prev_diff_mid'] = dataframe['smoothed_ema_mid'] / dataframe['smoothed_ema_mid'].shift(1)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.is_long:
            dataframe.loc[
                (
                    (dataframe['smoothed_ema_mid'] > self.up_ratio.value * dataframe['smoothed_ema_mid'].shift(1))
                    & (dataframe['smoothed_ema'] > dataframe['smoothed_ema_mid'])
                    & (dataframe['ohlc4'] > dataframe['smoothed_ema'])
                    # & (dataframe['ohlc4'] > dataframe['highest'].shift(1))
                ), 
                'enter_long'] = 1
        else:
            dataframe.loc[
                (
                    (dataframe['smoothed_ema_mid'] * self.up_ratio.value < dataframe['smoothed_ema_mid'].shift(1))
                    & (dataframe['smoothed_ema'] < dataframe['smoothed_ema_mid'])
                    & (dataframe['ohlc4'] < dataframe['smoothed_ema'])
                    # & (dataframe['ohlc4'] < dataframe['lowest'].shift(1))
                ), 
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        if self.use_custom_exit:
            return dataframe
        
        if self.is_long:
            dataframe.loc[
                (
                    (dataframe['smoothed_ema_mid'] * self.down_ratio.value < dataframe['smoothed_ema_mid'].shift(1))
                    | (dataframe['smoothed_ema'] < dataframe['smoothed_ema_mid'])
                    | (dataframe['ohlc4'] < dataframe['smoothed_ema_mid'])
                ), 
                'exit_long'] = 1
        else:
            dataframe.loc[
                (
                    (dataframe['smoothed_ema_mid'] > self.down_ratio.value * dataframe['smoothed_ema_mid'].shift(1))
                    | (dataframe['smoothed_ema'] > dataframe['smoothed_ema_mid'])
                    | (dataframe['ohlc4'] > dataframe['smoothed_ema_mid'])
                ), 
                'exit_short'] = 1

        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage
