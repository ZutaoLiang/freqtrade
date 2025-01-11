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


class DreamV2Strategy(IStrategy):
    
    # common
    minimal_roi = {"0": 100}
    
    trade_leverage = 10

    timeframe = '3m'
    
    base_stoploss_pct = 0.05
    stoploss = -base_stoploss_pct * trade_leverage
    trailing_stop = False
    use_custom_stoploss = True

    can_short = True
 
    position_adjustment_enable = True
    stake_ratio = 2/trade_leverage
    
    period = 10
    
    ema_length = period
    ema_mid_length = 6 * period
    ema_long_length = 24 * period
    ema_trend = 5
    ema_mid_trend = ema_trend
    ema_long_trend = ema_trend * 3
    ema_up_ratio = 1.01
    
    breakout_period = 5
    
    adx_length = period
    adx_threshold = 25
    rsi_length = period
    rsi_long_threshold = 55
    rsi_short_threshold = 30
    
    startup_candle_count = ema_long_length
    
    exit_loss_ratio = -0.2
    profit_drawback_threshold = 10
    profit_drawback_ratio = 0.03 * trade_leverage

    is_long = True
    
    # atr_length = int(1.5 * period)
    
    mean_reversion_change_pct = 0.005
    mean_reversion_stake_ratio = 0.2
    
    LAST_ADDITION_PRICE = 'last_addition_price'
    REVERSION_PRICE = 'reversion_price'
    MAX_PROFIT = 'max_profit'

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return self.trade_leverage
         
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        # TODO: 长期的stake amount上不来，说明加仓少，可以放弃掉了
        
        total = self.wallets.get_total(self.stake_currency)
        leverage = trade.leverage
        stake_amount = trade.amount * trade.open_rate
        open_profit_abs = current_profit / leverage * stake_amount
        close_profit_abs = trade.close_profit_abs if trade.close_profit_abs else 0
        total_profit_abs = close_profit_abs + open_profit_abs
        
        max_profit = trade.get_custom_data(self.MAX_PROFIT)
        if max_profit is None:
            max_profit = total_profit_abs
            trade.set_custom_data(self.MAX_PROFIT, max_profit)
        elif total_profit_abs > max_profit:
            logger.info(f'Update {trade.pair} max_profit from {max_profit:.4f} to {total_profit_abs:.4f} at {current_time}')
            max_profit = total_profit_abs
            trade.set_custom_data(self.MAX_PROFIT, max_profit)
        
        logger.info(f'{trade.pair} total profit:{total_profit_abs:.4f}(open:{open_profit_abs:.4f}, close:{close_profit_abs:.4f}), current_rate:{current_rate:.6f}, open_rate:{trade.open_rate:.6f}, current_profit:{current_profit:.2%}, stake_amount:{stake_amount:.4f} at {current_time}')
        
        if max_profit > self.profit_drawback_threshold and total_profit_abs < self.profit_drawback_ratio * max_profit:
            exit_reason = 'Profit drawback'
            logger.info(f'{exit_reason} for {pair}: total profit {total_profit_abs:.4f} < max_profit {max_profit:.4f} * {self.profit_drawback_ratio:.2%}, current_rate:{current_rate:.4f} at {current_time}')
            return exit_reason
        
        profit_drawdown_threshold = total * leverage * self.stake_ratio * self.exit_loss_ratio / self.max_open_trades
        if total_profit_abs < profit_drawdown_threshold:
            exit_reason = 'Max loss'
            logger.info(f'{exit_reason} for {pair}:{total_profit_abs:.4f} < {profit_drawdown_threshold:.4f}, current_rate:{current_rate:.6f} at {current_time}')
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
        if not self.position_adjustment_enable:
            return None

        filled_entries = trade.select_filled_orders()
        count_of_entries = len(filled_entries)
        if count_of_entries == 0:
            return None
        
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        is_short = trade.is_short
        leverage = trade.leverage
            
        latest_order = trade.select_order(order_side=trade.entry_side, is_open=False, only_filled=True)
        if latest_order is None:
            return None
        
        # 处理是否需要浮盈加仓
        last_addition_price = trade.get_custom_data(self.LAST_ADDITION_PRICE)
        if last_addition_price is None:
            last_addition_price = latest_order.average
            trade.set_custom_data(self.LAST_ADDITION_PRICE, last_addition_price)
        
        entry_signal = False
        
        if current_profit > 0.001 * leverage:
            addition_price_ratio = 0.98
            if is_short and last_candle['enter_short'] == 1 and current_rate < last_addition_price / addition_price_ratio:
                entry_signal = True
            elif not is_short and (last_candle['enter_long'] == 1 and current_rate > last_addition_price * addition_price_ratio):
                entry_signal = True
                
        min_stake /= trade.leverage
        max_stake /= trade.leverage
        
        if entry_signal:
            addition_stake = self.calc_proposed_stake()
            logger.info(f'Initialize {trade.pair} addition stake to {addition_stake:.5f} at {current_time}')
            
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
                trade.set_custom_data(self.REVERSION_PRICE, current_rate)
                
                logger.info(f'Position addition for {trade.pair} with stake amount {addition_stake:.5f} triggered at entry signal, current_profit:{current_profit:.2f}, current_rate:{current_rate:.5f} at {current_time}')
                return (addition_stake, 'entry-addition')

        # 处理均值回归
        last_reversion_price = trade.get_custom_data(self.REVERSION_PRICE)
        if last_reversion_price is None:
            last_reversion_price = latest_order.average
            trade.set_custom_data(self.REVERSION_PRICE, last_reversion_price)
        
        price_change = (last_reversion_price - current_rate) / last_reversion_price
        
        if abs(price_change) < self.mean_reversion_change_pct:
            return None
        
        reversion_stake = trade.amount * last_reversion_price * price_change
        # if reversion_stake > 0:
        reversion_stake *= self.mean_reversion_stake_ratio
        logger.info(f'Initialize reversion stake for {trade.pair} with stake amount:{reversion_stake:.4f}(amount:{trade.amount}, last_price:{last_reversion_price:.4f}, price_change:{price_change:.2%})')
        
        if abs(reversion_stake) > min_stake:
            # and abs(adjustment_value) < max_stake:
            logger.info(f'Mean reversion adjustment for {trade.pair} with stake amount:{reversion_stake:.4f}(amount:{trade.amount}, last_price:{last_reversion_price:.4f}, price_change:{price_change:.2%})')
            trade.set_custom_data(self.REVERSION_PRICE, current_rate)
            if reversion_stake > 0:
                return (reversion_stake, 'reversion-addition')
            else:
                return (reversion_stake, 'reversion-decrease')
        else:
            logger.info(f'Skip mean reversion adjustment for {trade.pair} while stake amount:{reversion_stake:.4f} is not in the valid range({min_stake:.4f}, {max_stake:.4f}) at {current_time}')
            return None
        
    def calc_proposed_stake(self) -> float:
        return self.wallets.get_total(self.stake_currency) * self.stake_ratio / self.max_open_trades
        
    def calc_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float, min_stake: float, max_stake: float, leverage: float):
        stake_amount = min(max(min_stake, proposed_stake * self.stake_ratio), max_stake)
        logger.info(f'Stake amount for {pair}={stake_amount:.5f} with leverage:{leverage}(total={stake_amount*leverage:.5f}), rate:{current_rate:.5f} at {current_time}')
        return stake_amount

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        return self.calc_stake_amount(pair, current_time, current_rate, proposed_stake, min_stake, max_stake, leverage)

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
        # dataframe['adx'] = pta.adx(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.adx_length)[f'ADX_{self.adx_length}']
        # dataframe['rsi'] = pta.rsi(dataframe['ha_close'], length=self.rsi_length, talib=False)
        # dataframe['recent_low'] = dataframe['ha_close'].rolling(window=self.breakout_period).min()
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
                    ['enter_long', 'enter_tag']] = (1, 'entry')
        
        return dataframe
        
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        return dataframe
