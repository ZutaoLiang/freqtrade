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

import logging
logger = logging.getLogger(__name__)


class DynamicPositionLong3mStrategyV7(IStrategy):
    
    # common
    minimal_roi = {"0": 100}
    
    trade_leverage = 3

    timeframe = '3m'
    
    stoploss = -0.04 * trade_leverage
    trailing_stop = False
    use_custom_stoploss = True

    can_short = True
 
    enable_logging = False
    
    position_adjustment_enable = True
    initial_position_ratio = 1/5
    
    position_adjustment_pct = 0.01
    position_adjustment_stake_ratio = 0.96
    
    period = 10
    ema_length = period
    ema_mid_length = 60
    ema_long_length = 120
    adx_length = period
    adx_threshold = 20
    rsi_length = period
    rsi_long_threshold = 55
    
    startup_candle_count = ema_long_length
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        stake_amount = min(max(min_stake, proposed_stake * self.initial_position_ratio), max_stake)
        logger.info(f'Stake amount for {pair}={stake_amount:.5f} rate:{current_rate:.5f} at {current_time}')
        return stake_amount
    
    def get_leverage(self, trade: Trade):
        return trade.leverage if trade.leverage else 1
    
    def is_low_profit(self, current_profit: float, trade: Trade):
        return current_profit < self.position_adjustment_pct
    
    def is_long_time(self, current_time: datetime, trade: Trade):
        return (current_time - timedelta(minutes=32)) > trade.open_date_utc
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs,) -> Optional[Union[str, bool]]:
        # if self.is_long_time(current_time, trade) and self.is_low_profit(current_profit, trade) and (current_profit > 0.005 * trade.leverage):
        #     logger.info(f'Long time low profit so exit all {trade.pair} at {current_rate:.5f}, current profit:{current_profit:.2f}')
        #     return 'Long time low profit'
        return None
    
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
        
        drawback = 0
        large_drawback = False
        if trade.max_rate is not None:
            drawback = (trade.max_rate - current_rate) / current_rate
            large_drawback = drawback > 0.08
        
        exit_long = False
        if current_rate < last_candle['ema_mid'] * 0.96:
            exit_long = True
        
        if large_drawback or exit_long:
            stake_to_decrease = 0
            
            # 统计有盈利的部分先减仓
            for order in trade.select_filled_orders():
                price_percent = (current_rate - order.average) / order.average
                if trade.is_short:
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
                    
                logger.info(f'Position partial decrease for {trade.pair} with amount {stake_to_decrease:.5f} at {current_rate:.5f}, current profit:{current_profit:.2f}. Drawback:{drawback:.2%}, large_drawback={large_drawback}, exit_long={exit_long}. {current_time}')
                
                if large_drawback:
                    return (stake_to_decrease, 'large drawback')
                elif exit_long:
                    return (stake_to_decrease, 'partial exit')

        filled_entries = trade.select_filled_orders()
        count_of_entries = len(filled_entries)
        if count_of_entries == 0:
            return None
       
        latest_order = trade.select_order(order_side=trade.entry_side, is_open=False, only_filled=True)
        if latest_order is None:
            return None
        
        if last_candle['enter_long'] == 1 and current_profit > 0 and current_rate > latest_order.average * 0.97 \
            and (current_time - timedelta(minutes=2)) > latest_order.order_filled_utc:
            stake_ratio = pow(self.position_adjustment_stake_ratio, len(filled_entries))
            first_stake = filled_entries[0].stake_amount
            stake_to_increase = stake_ratio * first_stake
            
            logger.info(f'Set {trade.pair} stake to increase to {stake_to_increase:.5f} with stake_ratio:{stake_ratio:.2f} for #{len(filled_entries)} based on first stake:{first_stake:.5f} at {current_time}')
            
            if stake_to_increase < min_stake:
                if stake_to_increase < 0.3 * min_stake:
                    if self.wallets.get_available_stake_amount() < stake_to_increase:
                        return None
                    
                    logger.info(f'Try to increase {trade.pair} position while amount:{stake_to_increase:.5f} is smaller than min_stake:{min_stake:.5f} at {current_time}')
                    return None
                else:
                    if self.wallets.get_available_stake_amount() < min_stake:
                        return None
                    
                    logger.info(f'Adjusting {trade.pair} increase stake:{stake_to_increase:.5f} to min_stake:{min_stake:.5f} at {current_time}')
                    stake_to_increase = min_stake
                    
            logger.info(f'Position addition #{count_of_entries+1} for {trade.pair} with amount {stake_to_increase:.5f} triggered at enter_long signal, profit:{current_profit:.2f} at {current_time}')
            return (stake_to_increase, 'addition')
                    
        return None
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        leverage = trade.leverage
        if not after_fill:
            if current_profit > 0.1 * leverage:
                new_stoploss = 0.05 * leverage
                return new_stoploss
        
            if (0.01 * leverage) < current_profit and current_profit < (0.03 * leverage) and (current_time - timedelta(hours=8)) > trade.open_date_utc:
                new_stoploss = 0.001 * leverage
                logger.info(f'Set {pair} long time low profit stoploss rate:{current_rate*(1-new_stoploss/trade.leverage):.5f}({new_stoploss:.2%}) at {current_time}')
                return new_stoploss
                
            # profitable_key = 'profitable'
            # new_stoploss = None
            # if (0.002 * leverage) < current_profit and current_profit < (0.02 * leverage):
            #     if (current_time - timedelta(hours=1)) > trade.open_date_utc:
            #         new_stoploss = 0.001 * leverage
            #         logger.info(f'Set {pair} long time low profit stoploss rate:{current_rate*(1-new_stoploss/trade.leverage):.5f}({new_stoploss:.2%}) at {current_time}')
            #         return new_stoploss
                
            #     profitable = trade.get_custom_data(profitable_key, default=0)
            #     if profitable and (current_time - timedelta(minutes=60)) > trade.open_date_utc:
            #         trade.set_custom_data(key=profitable_key, value=False)
            #         new_stoploss = 0.001 * leverage
            #         logger.info(f'Set {pair} profitable stoploss rate:{current_rate*(1-new_stoploss/trade.leverage):.5f}({new_stoploss:.2%}) at {current_time}')
            #         return new_stoploss
                
            # elif current_profit < 0:
            #     if trade.max_rate > trade.open_rate:
            #         trade.set_custom_data(key=profitable_key, value=True)
            #         return None
                
            return None
        
        filled_count = len(trade.select_filled_orders())
        if filled_count <= 1:
            # 第一次成交维持最原始的止损线
            logger.info(f'First fill for {pair}, will not refresh stoploss, current_profit:{current_profit:.2%}, open_rate:{trade.open_rate:.5f}/current_rate:{current_rate:.5f} at {current_time}')
            return None
        
        if current_profit > 0.32 * leverage:
            new_stoploss = 0.08 * leverage
        elif current_profit > 0.16 * leverage:
            new_stoploss = 0.06 * leverage
        elif current_profit > 0.08 * leverage:
            new_stoploss = 0.05 * leverage
        # elif current_profit > 0.02 * leverage:
        #     new_stoploss = 0.03 * leverage
        else:
            if current_profit >= 0.01 * leverage:
                logger.info(f'Setting {pair} stoploss with low profit:{current_profit:.2%}, current_rate:{current_rate:.5f}, open_rate:{trade.open_rate:.5f} at {current_time}')
                new_stoploss = current_profit - 0.01
            else:
                # 很小的止损线，基本相当于立即止损退出
                new_stoploss = 0.005 * leverage

            # 低利润或者已经亏损的情况下，往下放一些作为承受风险博取收益的止损价位
            stoploss_add_pct = 0.03 * leverage
            new_stoploss += stoploss_add_pct
            logger.info(f'Adding {stoploss_add_pct:.2%} to new stoploss:{new_stoploss:.2%} for {pair} at {current_time}')
        
        logger.info(f'Set {pair} stoploss rate:{current_rate*(1-new_stoploss/trade.leverage):.5f}({new_stoploss:.2%}) after fill relative to current_rate:{current_rate:.5f}, open_rate:{trade.open_rate:.5f}, current_profit:{current_profit:.2%} at {current_time}')
        
        return new_stoploss

    def heikinashi(self, dataframe: DataFrame) -> DataFrame:
        ha = qtpylib.heikinashi(dataframe)
        # ha = dataframe
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
        dataframe['recent_high'] = dataframe['ha_close'].rolling(window=self.period).max()
        # dataframe['recent_low'] = dataframe['close'].rolling(window=self.period).min()
        
        return dataframe
    
    def ema_up_n_days_mask(self, dataframe: DataFrame, ema: str, days: int):
        ema_up_mask = (dataframe[f'{ema}'] > dataframe[f'{ema}'].shift(1))
        for i in range(2, days):
            ema_up_mask = ema_up_mask & (dataframe[f'{ema}'].shift(i-1) > dataframe[f'{ema}'].shift(i))
        return ema_up_mask
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        ema_up_mask = self.ema_up_n_days_mask(dataframe, 'ema', 5)
        ema_mid_up_mask = self.ema_up_n_days_mask(dataframe, 'ema_mid', 5)
        ema_long_up_mask = self.ema_up_n_days_mask(dataframe, 'ema_long', 5)
        
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
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
 
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage
