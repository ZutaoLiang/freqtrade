import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame
from datetime import datetime, timezone
from typing import Dict, List, Optional
from freqtrade.persistence import Trade
from freqtrade.exchange import Exchange
from freqtrade.strategy import DecimalParameter, IntParameter

import logging
logger = logging.getLogger(__name__)


class EqualWeightStrategy(IStrategy):
    minimal_roi = {"0": 100}

    trade_leverage = DecimalParameter(
        1.0, 10.0, default=2.0,
        space="buy",
        optimize=True,
        load=True
    )
    
    position_adjustment_threshold = DecimalParameter(
        0.01, 0.05, default=0.02, 
        space="buy", 
        optimize=True,
        load=True
    )

    timeframe = '1m'
    process_only_new_candles = True
    use_exit_signal = True
    can_short = True
    position_adjustment_enable = True

    stoploss = -0.2 * trade_leverage.value

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """
        自定义入场金额
        使用proposed_stake的50%进行初始购买
        """
        return (proposed_stake / 2)

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """计算技术指标"""
        # 计算EMA5
        dataframe['ema5'] = ta.EMA(dataframe['close'], timeperiod=5)
        # 计算EMA5斜率
        dataframe['ema5_slope'] = dataframe['ema5'] - dataframe['ema5'].shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """生成入场信号"""
        dataframe.loc[
            (
                (dataframe['ema5_slope'] > 0) &  # EMA5向上
                (dataframe['close'] > dataframe['ema5']) &  # 收盘价在EMA5上方
                (dataframe['volume'] > 0)  # 确保有交易量
            ),
            'enter_long'] = 1
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """生成出场信号 - 主要通过adjust_trade_position和止损来管理"""
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
        
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        current_leverage = trade.leverage

        if current_profit > 0.2 * current_leverage:
            return 0.15 * current_leverage
        
        return None

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
            
        target_value = self.wallets.get_total(self.stake_currency) / self.max_open_trades
        
        # 计算当前市值（考虑杠杆因素）
        current_value = trade.amount * current_rate
        
        # 计算需要调整的金额，即杠杆后
        adjustment_value = (target_value - current_value)
        
        # 检查调整金额是否在允许范围内
        if abs(adjustment_value) < min_stake or abs(adjustment_value) > max_stake:
            logger.debug(
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
        
        return adjustment_value

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage.value
 