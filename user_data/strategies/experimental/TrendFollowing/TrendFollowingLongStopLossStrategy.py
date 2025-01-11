from math import isnan
import numpy as np
import pandas_ta as pta
import pandas as pd
from pandas import DataFrame

from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
from functools import reduce

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_helper import stoploss_from_absolute
from freqtrade.strategy import IntParameter, DecimalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib


class TrendFollowingLongStopLossStrategy(IStrategy):
    minimal_roi = {
        "0": 100
    }
    
    buy_leverage = 2

    timeframe = '5m'

    use_custom_stoploss = True
    stoploss = -0.05 * buy_leverage  # 基础止损

    # 改用自定义的阶梯式trailing stop
    trailing_stop = False  # 关闭内置的trailing stop，改用自定义的阶梯式trailing stop
    
    # 保持原有参数设置
    can_short = True
    atr_length = 14
    atr_multiplier = 4
    short_period = 10
    mid_period = 30
    long_period = 60
    adx_length = short_period
    adx_threshold = 25
    rsi_length = short_period
    rsi_long_threshold = 60
    rsi_short_threshold = 40
    ema_short_length = short_period
    ema_mid_length = mid_period
    ema_long_length = long_period
    bbands_length = mid_period
    bbands_std = 2.0

    startup_candle_count = max(ema_long_length, long_period, atr_length, bbands_length)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 保持原有的指标计算逻辑，但添加一些用于止损判断的新指标
        dataframe = self.heikinashi(dataframe)
        
        # 原有指标计算
        dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_length)
        dataframe['smooth_high'] = dataframe['ha_high'].rolling(window=3).mean()
        dataframe['smooth_low'] = dataframe['ha_low'].rolling(window=3).mean()
        dataframe['volume_ma'] = pta.sma(dataframe['volume'], length=5)
        dataframe['adx'] = pta.adx(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.adx_length)[f'ADX_{self.adx_length}']
        
        # EMA系列
        dataframe['ema_short'] = pta.ema(dataframe['ha_close'], length=self.ema_short_length)
        dataframe['ema_mid'] = pta.ema(dataframe['ha_close'], length=self.ema_mid_length)
        dataframe['ema_long'] = pta.ema(dataframe['ha_close'], length=self.ema_long_length)
        
        # 布林带
        bbands = pta.bbands(dataframe['ha_close'], length=self.bbands_length, std=self.bbands_std)
        dataframe['bbands_lower'] = bbands[f'BBL_{self.bbands_length}_2.0']
        dataframe['bbands_upper'] = bbands[f'BBU_{self.bbands_length}_2.0']
        dataframe['bbands_mid'] = bbands[f'BBM_{self.bbands_length}_2.0']
        
        # 趋势强度指标
        dataframe['rsi'] = pta.rsi(dataframe['ha_close'], length=self.rsi_length)
        dataframe['is_bullish'] = (dataframe['ha_close'] > dataframe['ha_open']) & (dataframe['ha_close'] > dataframe['ha_close'].shift(1))
        dataframe['bullish_count'] = dataframe['is_bullish'].rolling(window=self.short_period).sum()
        dataframe['is_bearish'] = (dataframe['ha_close'] < dataframe['ha_open']) & (dataframe['ha_close'] < dataframe['ha_close'].shift(1))
        dataframe['bearish_count'] = dataframe['is_bearish'].rolling(window=self.short_period).sum()
        
        # 趋势反转信号
        dataframe['trend_reversal'] = (
            (dataframe['ema_short'] < dataframe['ema_mid']) &  # 短期均线下穿中期均线
            (dataframe['bearish_count'] >= 3) &  # 连续3根以上看跌K线
            (dataframe['volume'] > dataframe['volume_ma'])  # 放量下跌
        )
        
        # 波动率指标
        dataframe['volatility'] = (dataframe['ha_high'] - dataframe['ha_low']) / dataframe['ha_low'] * 100
        dataframe['avg_volatility'] = dataframe['volatility'].rolling(window=10).mean()
        
        return dataframe

    # 保持原有的入场逻辑不变
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        long_entry_conditions = []
        
        # 交易量条件加强
        long_entry_conditions.append(dataframe['volume'] > dataframe['volume'].shift(1) * 1.2)
        long_entry_conditions.append(dataframe['volume_ma'] > dataframe['volume_ma'].shift(1))
        
        # EMA多空排列条件
        long_entry_conditions.append(dataframe['ha_close'] >= dataframe['ema_short'])
        long_entry_conditions.append(dataframe['ema_short'] >= dataframe['ema_mid'])
        long_entry_conditions.append(dataframe['ema_mid'] >= dataframe['ema_long'])
        
        # EMA上升趋势 - 加强趋势确认
        long_entry_conditions.append(dataframe['ema_short'] > dataframe['ema_short'].shift(3))
        long_entry_conditions.append(dataframe['ema_mid'] > dataframe['ema_mid'].shift(3))
        long_entry_conditions.append(dataframe['ema_long'] > dataframe['ema_long'].shift(3))
        
        # # 突破最近高点 - 使用平滑后的价格
        # long_entry_conditions.append(dataframe['smooth_high'] >= dataframe['recent_high_mid'].shift(1))
        
        # 趋势强度条件
        long_entry_conditions.append(dataframe['adx'] >= self.adx_threshold)
        
        # RSI过滤 - 避免过度追高
        long_entry_conditions.append((dataframe['rsi'] >= self.rsi_long_threshold) & 
                                   (dataframe['rsi'] < 80))
        
        # 趋势连续性 - 要求更强的趋势
        long_entry_conditions.append(dataframe['bullish_count'] >= (int)(self.short_period * 0.7))
        
        # 布林带过滤
        long_entry_conditions.append(dataframe['ha_close'] > dataframe['bbands_mid'])
        long_entry_conditions.append(dataframe['ha_close'] <= dataframe['bbands_upper'])

        dataframe.loc[
            reduce(lambda x, y: x & y, long_entry_conditions),
            'enter_long'
        ] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        # # EMA趋势反转退出
        # dataframe.loc[
        #     (dataframe['ema_short'] < dataframe['ema_mid']) &
        #     (dataframe['ema_mid'] < dataframe['ema_long']),
        #     ['exit_long', 'exit_tag']
        # ] = (1, 'ema_exit')

        # # 强烈的熊市信号退出
        # dataframe.loc[
        #     (dataframe['bearish_count'] >= (int)(self.short_period * 0.8)) &
        #     (dataframe['ha_close'] < dataframe['bbands_mid']),
        #     ['exit_long', 'exit_tag']
        # ] = (1, 'bearish_exit')

        return dataframe
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, after_fill: bool,
                       **kwargs) -> Optional[float]:
        
        # 获取最新的分析数据
        last_analyzed_df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if last_analyzed_df.empty:
            return self.stoploss
            
        last_candle = last_analyzed_df.iloc[-1]
        
        # 1. 阶梯式追踪止损逻辑
        if current_profit > 0.05:  # 开始启用追踪止损的最低盈利要求
            # 定义三个阶梯的止损水平
            if current_profit >= 0.20:  # 第三阶梯：盈利>=20%
                trailing_stop_pct = 0.15  # 允许回撤15%
                min_profit = 0.15  # 保证至少15%的盈利
            elif current_profit >= 0.12:  # 第二阶梯：盈利>=12%
                trailing_stop_pct = 0.10  # 允许回撤10%
                min_profit = 0.08  # 保证至少8%的盈利
            else:  # 第一阶梯：盈利5%-12%
                trailing_stop_pct = 0.06  # 允许回撤6%
                min_profit = 0.03  # 保证至少3%的盈利
            
            # 计算基于最高价的追踪止损价格
            stop_price = trade.max_rate * (1 - trailing_stop_pct)
            
            # 确保止损价格至少保证最小盈利
            min_price = trade.open_rate * (1 + min_profit)
            stop_price = max(stop_price, min_price)
            
            return stoploss_from_absolute(
                stop_rate=stop_price,
                current_rate=current_rate,
                is_short=trade.is_short,
                leverage=trade.leverage
            )
        
        # 2. 盈利保护逻辑（适用于小盈利订单）
        if current_profit > 0.03:  # 有3%以上的盈利
            # 检查是否出现趋势反转信号
            if last_candle['trend_reversal']:
                return stoploss_from_absolute(
                    stop_rate=trade.open_rate * 1.01,  # 保证至少1%的盈利
                    current_rate=current_rate,
                    is_short=trade.is_short,
                    leverage=trade.leverage
                )
        
        # 3. 基于持仓时间的动态止损
        trade_duration = current_time - trade.open_date_utc
        
        # 前30分钟使用较为宽松的止损
        if trade_duration <= timedelta(minutes=30):
            return self.stoploss
            
        # 30-60分钟区间，如果没有盈利，使用更严格的止损
        if trade_duration <= timedelta(minutes=60) and current_profit < 0.02:
            return stoploss_from_absolute(
                stop_rate=current_rate * 0.98,  # 2%止损
                current_rate=current_rate,
                is_short=trade.is_short,
                leverage=trade.leverage
            )
        
        # 4. 基于ATR的动态止损（仅在有显著盈利但未达到追踪止损条件时使用）
        if 0.03 < current_profit <= 0.05:  # 3%-5%盈利区间
            atr = last_candle['atr']
            volatility = last_candle['avg_volatility']
            
            # 根据波动率调整ATR倍数
            atr_multiplier = self.atr_multiplier
            if volatility > 1.5:  # 高波动环境
                atr_multiplier *= 1.5  # 给予更大的价格波动空间
            
            if not trade.is_short:
                stop_loss_price = max(
                    trade.max_rate - (atr_multiplier * atr),  # ATR止损
                    trade.open_rate * 1.02  # 保证至少2%盈利
                )
                return stoploss_from_absolute(
                    stop_rate=stop_loss_price,
                    current_rate=current_rate,
                    is_short=trade.is_short,
                    leverage=trade.leverage
                )
    
        return self.stoploss
    
    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.mid_period,
                "trade_limit": 2,
                "stop_duration_candles": 6,  # 冷却时间6根K线
                "only_per_pair": True,
                "only_per_side": False
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return self.buy_leverage
        
    def heikinashi(self, dataframe: DataFrame) -> DataFrame:
        ha = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']
        return dataframe