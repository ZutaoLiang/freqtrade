import numpy as np
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy import IStrategy
from pandas import DataFrame
from datetime import datetime, timezone
from typing import Dict, List, Optional
from freqtrade.persistence import Trade
from freqtrade.exchange import Exchange
from freqtrade.strategy import DecimalParameter, IntParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib

import logging
logger = logging.getLogger(__name__)


class EqualWeightStrategyV1(IStrategy):
    minimal_roi = {"0": 100}

    trade_leverage = DecimalParameter(
        1.0, 10.0, default=5.0,
        space="buy",
        optimize=True,
        load=True
    )
    
    position_adjustment_threshold = DecimalParameter(
        0.01, 0.05, default=0.015, 
        space="buy", 
        optimize=True,
        load=True
    )

    use_ha_candles = True
    timeframe = '5m'
    process_only_new_candles = True
    can_short = True
    position_adjustment_enable = True

    base_stoploss = 0.1
    stoploss = -base_stoploss * trade_leverage.value
    
    lookback_period = 12
    ema_short_len = IntParameter(5, 100, default=lookback_period, space='buy')
    ema_mid_len = IntParameter(5, 100, default=lookback_period * 3, space='buy')
    ema_long_len = IntParameter(5, 100, default=lookback_period * 6, space='buy')
    ema_week_len = IntParameter(5, 100, default=lookback_period * 72, space='buy')

    trend_length = 2

    def get_trade_data(self, trade: Trade) -> dict:
        """
        获取交易的自定义数据，如果不存在则初始化
        """
        # 获取上次调整价格
        last_price = trade.get_custom_data('last_adjustment_price')
        if last_price is None:
            # 如果是首次调整，初始化所有数据
            trade.set_custom_data('last_adjustment_price', trade.open_rate)
            trade.set_custom_data('last_adjustment_time', trade.open_date_utc.timestamp())
            trade.set_custom_data('adjustment_count', 0)
        
        # 返回完整的数据字典
        return {
            'last_adjustment_price': trade.get_custom_data('last_adjustment_price'),
            'last_adjustment_time': trade.get_custom_data('last_adjustment_time'),
            'adjustment_count': trade.get_custom_data('adjustment_count'),
        }

    def update_trade_data(self, trade: Trade, current_rate: float, current_time: datetime) -> None:
        """
        更新交易的自定义数据
        """
        # 获取当前调仓次数并增加
        adjustment_count = trade.get_custom_data('adjustment_count')
        if adjustment_count is None:
            adjustment_count = 0
            
        # 更新所有数据
        trade.set_custom_data('last_adjustment_price', current_rate)
        trade.set_custom_data('last_adjustment_time', current_time.timestamp())
        trade.set_custom_data('adjustment_count', adjustment_count + 1)

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
        dataframe = self.calculate_ha(dataframe)
        
        dataframe['ema_short'] = pta.ema(close=dataframe['ha_close'], length=self.ema_short_len.value, talib=False)
        dataframe['ema_mid'] = pta.ema(close=dataframe['ha_close'], length=self.ema_mid_len.value, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ha_close'], length=self.ema_long_len.value, talib=False)
        dataframe['ema_week'] = pta.ema(close=dataframe['ha_close'], length=self.ema_week_len.value, talib=False)

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
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        dataframe.loc[
            (
                (dataframe['ha_close'] > dataframe['ema_short'])
                & (dataframe['ha_close'] > dataframe['ha_close'].shift(1))
                & (dataframe['ha_close'] > dataframe['ha_open'])
                
                & (dataframe['ema_short'] > dataframe['ema_long'])
                & (dataframe['ema_mid'] > dataframe['ema_week'])
                & (dataframe['ema_long'] > dataframe['ema_week'])
                
                & (self.indicator_up_n_periods_mask(dataframe, 'ema_short', self.trend_length))
                & (self.indicator_up_n_periods_mask(dataframe, 'ema_mid', self.trend_length))
                & (self.indicator_up_n_periods_mask(dataframe, 'ema_long', self.trend_length))
                & (self.indicator_up_n_periods_mask(dataframe, 'ema_week', self.trend_length))
            ), 
            ['enter_long', 'enter_tag']] = (1, 'entry_long')

        dataframe.loc[
            (
                (dataframe['ha_close'] < dataframe['ema_short'])
                & (dataframe['ha_close'] < dataframe['ha_close'].shift(1))
                & (dataframe['ha_close'] < dataframe['ha_open'])
                
                & (dataframe['ema_short'] < dataframe['ema_long'])
                & (dataframe['ema_mid'] < dataframe['ema_week'])
                & (dataframe['ema_long'] < dataframe['ema_week'])
                
                & (self.indicator_down_n_periods_mask(dataframe, 'ema_short', self.trend_length))
                & (self.indicator_down_n_periods_mask(dataframe, 'ema_mid', self.trend_length))
                & (self.indicator_down_n_periods_mask(dataframe, 'ema_long', self.trend_length))
                & (self.indicator_down_n_periods_mask(dataframe, 'ema_week', self.trend_length))
            ), 
            ['enter_short', 'enter_tag']] = (1, 'entry_short')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        dataframe.loc[
            (
                (dataframe['ha_close'] < dataframe['ema_week'])
                # & (dataframe['ema_short'] < dataframe['ema_long'])
                # & (dataframe['enter_long'] == 0)
            ), 
            ['exit_long', 'exit_tag']] = (1, 'exit_ema')
        
        dataframe.loc[
            (
                (dataframe['ha_close'] > dataframe['ema_week'])
                # & (dataframe['ema_short'] > dataframe['ema_long'])
                # & (dataframe['enter_short'] == 0)
            ), 
            ['exit_short', 'exit_tag']] = (1, 'exit_ema')
        
        return dataframe
        
    # def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
    #                     current_rate: float, current_profit: float, after_fill: bool,
    #                     **kwargs) -> Optional[float]:
    #     current_leverage = trade.leverage

    #     if current_profit > 0.2 * current_leverage:
    #         return 0.15 * current_leverage

    #     return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime, 
                            current_rate: float, current_profit: float,
                            min_stake: float, max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs) -> Optional[float]:
        """
        调整持仓金额以实现均仓效果
        返回值：
            正数: 需要追加的资金金额
            负数: 需要减少的资金金额
            None: 不需要调整
        """
        
        # 如果已经触及止损线，不进行调仓
        if current_profit <= self.stoploss:
            logger.info(
                f"{trade.pair} 当前亏损 {current_profit:.2%} 已达到止损线 {self.stoploss:.2%}，"
                f"不再进行调仓"
            )
            return None

        # 获取交易的自定义数据
        custom_data = self.get_trade_data(trade)
        # logger.info(f'custom_data:{custom_data}')
        last_price = custom_data['last_adjustment_price']
        
        # 计算价格变化百分比
        price_change = abs(current_rate - last_price) / last_price
        
        # 如果价格变化不够大，不进行调整
        if price_change < self.position_adjustment_threshold.value:
            return None
            
        target_value = self.config.get('stake_amount') * self.trade_leverage.value
        
        # 计算当前市值（考虑杠杆因素）
        current_value = trade.amount * current_rate
        
        # 计算需要调整的金额，即杠杆后
        adjustment_value = (target_value - current_value)
        
        # 检查调整金额是否在允许范围内
        if abs(adjustment_value) < min_stake:
            logger.info(
                f"{trade.pair} 调整金额 {adjustment_value:.4f}(当前市值:{current_value:.4f},目标市值:{target_value:.4f}) 超出允许范围"
                f"[{min_stake:.4f}, {max_stake:.4f}]，本次不进行调整"
            )
            return None
        
        # 记录调仓信息到日志
        logger.info(
            f"调整仓位 {trade.pair}: "
            f"调仓次数={custom_data['adjustment_count']}, "
            f"上次价格={custom_data['last_adjustment_price']:.4f}, "
            f"当前价格={current_rate:.4f}, "
            f"价格变动={price_change:.2%}, "
            f"目标市值={target_value:.4f}, "
            f"当前市值={current_value:.4f}, "
            f"调整金额={adjustment_value:.4f}, "
            f"当前杠杆={trade.leverage}x, "
            f"止损线={self.stoploss:.2%}, "
            f"调整时间={current_time}"
        )

         # 更新调仓相关数据
        self.update_trade_data(trade, current_rate, current_time)
        
        adjustment_stake_without_leverage = adjustment_value / self.trade_leverage.value
        return adjustment_stake_without_leverage

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        filled_orders = trade.select_filled_orders()
        first_order = filled_orders[0]
        
        first_price = first_order.average
        
        # if current_profit / trade.leverage < -0.05:
        #     return "low_profit"
        
        factor = -1 if trade.is_short else 1
        price_diff = factor * (current_rate - first_price) / first_price
        if price_diff < -0.1:
            return "price_diff"
        
        return None
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage.value
