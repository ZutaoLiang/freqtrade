from freqtrade.strategy.interface import IStrategy
import numpy as np
import pandas as pd
import pandas_ta as pta
from pandas import DataFrame
from datetime import datetime, timedelta
from typing import Optional
from freqtrade.persistence import Trade
import logging
logger = logging.getLogger(__name__)

class ScalpingStrategy(IStrategy):
    timeframe = '3m'
    trade_leverage = 10
    
    trailing_stop = False
    # trailing_stop_positive = 0.002 * trade_leverage
    # trailing_stop_positive_offset = 0.005 * trade_leverage
    # trailing_only_offset_is_reached = True
    use_custom_stoploss = True
    
    stoploss = -0.02 * trade_leverage
    minimal_roi = {"0": 0.01 * trade_leverage}
    
    max_open_trades = 10
    position_adjustment_enable = True
    
    # 技术指标参数
    rsi_period = 6
    rsi_oversold = 30
    rsi_overbought = 70
    volume_ma_period = 20
    price_ma_period = 8
    long_ma_period = 50
    momentum_period = 3
    volume_multiplier = 1.1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = pta.rsi(dataframe['close'], length=self.rsi_period)
        
        # Moving Averages
        dataframe['sma'] = pta.sma(dataframe['close'], length=self.price_ma_period)
        dataframe['long_ma'] = pta.sma(dataframe['close'], length=self.long_ma_period)
        dataframe['vol_ma'] = pta.sma(dataframe['volume'], length=self.volume_ma_period)
        
        # Momentum
        dataframe['momentum'] = dataframe['close'] - dataframe['close'].shift(self.momentum_period)
        
        # Support/Resistance
        dataframe['support'] = dataframe['low'].rolling(window=10).min()
        dataframe['resistance'] = dataframe['high'].rolling(window=10).max()
        
        # 计算成交量特征
        dataframe['vol_ratio'] = dataframe['volume'] / dataframe['vol_ma']
        dataframe['high_volume'] = dataframe['vol_ratio'] > self.volume_multiplier
        
        # 价格波动
        dataframe['price_range'] = (dataframe['high'] - dataframe['low']) / dataframe['low']
        dataframe['avg_range'] = dataframe['price_range'].rolling(window=20).mean()
        dataframe['range_ratio'] = dataframe['price_range'] / dataframe['avg_range']
        
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, 
                   current_rate: float, current_profit: float, **kwargs) -> str:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        
        # RSI超买超卖退出
        if not trade.is_short and last_candle['rsi'] > self.rsi_overbought:
            return 'long_rsi_overbought'
        elif trade.is_short and last_candle['rsi'] < self.rsi_oversold:
            return 'short_rsi_oversold'
            
        # 价格突破支撑/阻力位退出
        if not trade.is_short and current_rate < last_candle['support']:
            return 'long_support_break'
        elif trade.is_short and current_rate > last_candle['resistance']:
            return 'short_resistance_break'
            
        # 获利回吐保护
        # if current_profit > 0.02 * trade.leverage:  # 2%获利
        #     if not trade.is_short and last_candle['close'] < last_candle['sma']:
        #         return 'long_profit_ma_exit'
        #     elif trade.is_short and last_candle['close'] > last_candle['sma']:
        #         return 'short_profit_ma_exit'
                
        return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        # leverage = trade.leverage
        # profit_pct = current_profit / leverage
        # if profit_pct > 0.03:
        #     return 0.02 * leverage
        # elif profit_pct > 0.02:
        #     return 0.015 * leverage
        # elif profit_pct > 0.01:
        #     return 0.01 * leverage
            
        return self.stoploss * 3

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Long Entry Conditions
        long_volume_signal = (
            dataframe['high_volume'] &  # 高成交量
            (dataframe['range_ratio'] > 1.05)  # 大于平均波动
        )
        
        long_conditions = (
            (dataframe['rsi'] < 40) &  # RSI超卖
            (dataframe['close'] > dataframe['sma']) &  # 价格在短期均线上方
            (dataframe['momentum'] > 0) &  # 动量为正
            long_volume_signal  # 成交量信号
        )
        
        dataframe.loc[long_conditions, 'enter_long'] = 1
        
        # Short Entry Conditions
        short_volume_signal = (
            dataframe['high_volume'] &
            (dataframe['range_ratio'] > 1.1)
        )
        
        short_conditions = (
            (dataframe['rsi'] > 60) &
            (dataframe['close'] < dataframe['sma']) &
            (dataframe['momentum'] < 0) &
            short_volume_signal
        )
        
        dataframe.loc[short_conditions, 'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        return dataframe
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float, 
                            min_stake: float, max_stake: float,
                            **kwargs) -> Optional[float]:
        if not self.position_adjustment_enable or current_profit <= 0:
            return None
            
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
            
        filled_entries = trade.select_filled_orders()
            
        last_entry = filled_entries[-1]
        stake_amount = last_entry.cost
        
        # 根据RSI和成交量强度调整加仓数量
        if not trade.is_short:
            signal_strength = (40 - last_candle['rsi']) / 20
        else:
            signal_strength = (last_candle['rsi'] - 60) / 20
            
        volume_factor = min(last_candle['vol_ratio'] - 1, 1)  # 将成交量因子限制在0-1之间
        signal_strength = signal_strength * (1 + volume_factor)
            
        if signal_strength > 0.5:  # 信号强度阈值
            return stake_amount * signal_strength
            
        return None
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return self.trade_leverage