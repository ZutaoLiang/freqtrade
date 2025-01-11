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


class TrendFollowingLongStrategy(IStrategy):
    minimal_roi = {
        "0": 100
    }
    
    buy_leverage = 2

    timeframe = '5m'

    use_custom_stoploss = True
    stoploss = -0.05 * buy_leverage

    # 修改追踪止损参数，提高触发门槛
    trailing_stop = True
    trailing_stop_positive = 0.08 * buy_leverage  # 提高至8%
    trailing_stop_positive_offset = 0.10 * buy_leverage  # 提高至10%
    trailing_only_offset_is_reached = True  # 改为True，必须达到offset才启动追踪止损

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

    def heikinashi(self, dataframe: DataFrame) -> DataFrame:
        ha = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.heikinashi(dataframe)
        
        # ATR指标
        dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_length)
        
        # 添加平滑处理的high/low，降低高点的敏感度
        dataframe['smooth_high'] = dataframe['ha_high'].rolling(window=3).mean()
        dataframe['smooth_low'] = dataframe['ha_low'].rolling(window=3).mean()
        
        # 移动平均成交量
        dataframe['volume_ma'] = pta.sma(dataframe['volume'], length=5)

        # ADX指标
        adx = pta.adx(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.adx_length)
        dataframe['adx'] = adx[f'ADX_{self.adx_length}']

        # EMA系列
        dataframe['ema_short'] = pta.ema(dataframe['ha_close'], length=self.ema_short_length)
        dataframe['ema_mid'] = pta.ema(dataframe['ha_close'], length=self.ema_mid_length)
        dataframe['ema_long'] = pta.ema(dataframe['ha_close'], length=self.ema_long_length)

        # 布林带指标
        bbands = pta.bbands(dataframe['ha_close'], length=self.bbands_length, std=self.bbands_std)
        dataframe['bbands_lower'] = bbands[f'BBL_{self.bbands_length}_2.0']
        dataframe['bbands_upper'] = bbands[f'BBU_{self.bbands_length}_2.0']
        dataframe['bbands_mid'] = bbands[f'BBM_{self.bbands_length}_2.0']

        # 最近中期高低价范围 - 使用平滑后的价格
        dataframe['recent_high_mid'] = dataframe['smooth_high'].rolling(window=self.mid_period).max()
        dataframe['recent_low_mid'] = dataframe['smooth_low'].rolling(window=self.mid_period).min()
        
        # RSI指标
        dataframe['rsi'] = pta.rsi(dataframe['ha_close'], length=self.rsi_length)
        
        # 趋势判断 - 使用更严格的条件
        dataframe['is_bullish'] = (dataframe['ha_close'] > dataframe['ha_open']) & \
                                 (dataframe['ha_close'] > dataframe['ha_close'].shift(1))
        dataframe['bullish_count'] = dataframe['is_bullish'].rolling(window=self.short_period).sum()

        dataframe['is_bearish'] = (dataframe['ha_close'] < dataframe['ha_open']) & \
                                 (dataframe['ha_close'] < dataframe['ha_close'].shift(1))
        dataframe['bearish_count'] = dataframe['is_bearish'].rolling(window=self.short_period).sum()
        
        return dataframe

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
        
        # 突破最近高点 - 使用平滑后的价格
        long_entry_conditions.append(dataframe['smooth_high'] >= dataframe['recent_high_mid'].shift(1))
        
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

        # EMA趋势反转退出
        dataframe.loc[
            (dataframe['ema_short'] < dataframe['ema_mid']) &
            (dataframe['ema_mid'] < dataframe['ema_long']),
            ['exit_long', 'exit_tag']
        ] = (1, 'ema_exit')

        # 强烈的熊市信号退出
        dataframe.loc[
            (dataframe['bearish_count'] >= (int)(self.short_period * 0.8)) &
            (dataframe['ha_close'] < dataframe['bbands_mid']),
            ['exit_long', 'exit_tag']
        ] = (1, 'bearish_exit')

        return dataframe
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, after_fill: bool,
                       **kwargs) -> Optional[float]:
        # 前45分钟内如果无利润，则采用更严格的止损
        if current_time - timedelta(minutes=45) > trade.open_date_utc and current_profit < 0.02:
            return stoploss_from_absolute(
                stop_rate=current_rate * 0.97,
                current_rate=current_rate,
                is_short=trade.is_short,
                leverage=trade.leverage
            )
        
        # 根据ATR动态调整止损
        if current_profit > 0.05:  # 只在有显著盈利时使用ATR止损
            last_analyzed_df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not last_analyzed_df.empty:
                last_candle = last_analyzed_df.iloc[-1]
                atr = last_candle['atr']
                
                if not trade.is_short:
                    stop_loss_price = trade.max_rate - (self.atr_multiplier * atr)
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
                "stop_duration_candles": 4,  # 增加冷却时间
                "only_per_pair": True,
                "only_per_side": False
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return self.buy_leverage
