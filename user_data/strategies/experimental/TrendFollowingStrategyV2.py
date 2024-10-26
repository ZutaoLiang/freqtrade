import pandas as pd
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta
from datetime import datetime
from typing import Optional, Tuple, Union
from freqtrade.persistence import Order, Trade

import logging
logger = logging.getLogger(__name__)


class TrendFollowingStrategyV2(IStrategy):
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.2
    
    # Stoploss:
    stoploss = -base_stop_loss * buy_leverage.value
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = base_stop_loss
    trailing_stop_positive_offset = 0.0  # Disabled

    can_short = True
 
    timeframe = '1d'
    
    breakout_long = 50
    breakout_short = 25
    atr_period = breakout_long
    
    atr_multiplier = 3
    
    ema_short_period = IntParameter(5, 100, default=10, space='buy')
    ema_long_period = IntParameter(5, 300, default=200, space='buy')
    ema_days = 5
 
    startup_candle_count = ema_long_period.value
    
    # use_custom_stoploss = False

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

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['ema_short'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_short_period.value, talib=False)
        dataframe['ema_long'] = pta.ema(close=dataframe['ohlc4'], length=self.ema_long_period.value, talib=False)
        
        dataframe['highest_long'] = dataframe['high'].rolling(window=self.breakout_long).max()
        dataframe['lowest_long'] = dataframe['low'].rolling(window=self.breakout_long).min()
        dataframe['highest_short'] = dataframe['high'].rolling(window=self.breakout_short).max()
        dataframe['lowest_short'] = dataframe['low'].rolling(window=self.breakout_short).min()

        dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_period)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        ema_up_mask_short = self.ema_up_n_days_mask(dataframe, 'ema_short', self.ema_days)
        ema_up_mask_long = self.ema_up_n_days_mask(dataframe, 'ema_long', self.ema_days)
        
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['highest_long'].shift(1))
                & ema_up_mask_short
                & ema_up_mask_long
            ),
            'enter_long'] = 1
        
        ema_down_mask_short = self.ema_down_n_days_mask(dataframe, 'ema_short', self.ema_days)
        ema_down_mask_long = self.ema_down_n_days_mask(dataframe, 'ema_long', self.ema_days)

        dataframe.loc[
            (
                (dataframe['close'] < dataframe['lowest_long'].shift(1))
                & ema_down_mask_short
                & ema_down_mask_long
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['lowest_short'].shift(1))
            ),
            'exit_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['highest_short'].shift(1))
            ),
            'exit_short'] = 1

        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs,) -> Optional[Union[str, bool]]:
        keep = None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return keep
        
        last_candle = dataframe.loc[dataframe['date'] <= current_time]
        if last_candle.empty:
            return keep
        
        last_candle = last_candle.iloc[-1]
        atr = last_candle['atr'] # 实际策略运行过程中当前K线可能不完整，对应计算的ATR可能不正确，这里采用倒数第二根K线的计算值

        if trade.is_short:
            min_rate = trade.min_rate if trade.min_rate is not None else trade.open_rate
            stop_loss_price = min_rate + (self.atr_multiplier * atr)
            # logger.info(f'Calculating short exit for {pair}, trade_id:{trade.id}, current_time:{current_time}, current_rate:{current_rate}. ' \
                    # f'min_rate:{min_rate}, atr:{atr}, stop_loss_price:{stop_loss_price}, current_profit:{current_profit}, trade={trade}')
            if current_rate > stop_loss_price:
                logger.info(f'Custom short exit for {pair}, trade_id:{trade.id}, current_time:{current_time}, current_rate:{current_rate}. ' \
                    f'min_rate:{min_rate}, atr:{atr}, stop_loss_price:{stop_loss_price}, current_profit:{current_profit}, trade={trade}')
                return 'exit_atr'
        else:
            max_rate = trade.max_rate if trade.max_rate is not None else trade.open_rate
            stop_loss_price = max_rate - (self.atr_multiplier * atr)
            # logger.info(f'Calculating long exit for {pair}, trade_id:{trade.id}, current_time:{current_time}, current_rate:{current_rate}. ' \
                    # f'max_rate:{max_rate}, atr:{atr}, stop_loss_price:{stop_loss_price}, current_profit:{current_profit}, trade={trade}')
            if current_rate < stop_loss_price:
                logger.info(f'Custom long exit for {pair}, trade_id:{trade.id}, current_time:{current_time}, current_rate:{current_rate}. ' \
                    f'max_rate:{max_rate}, atr:{atr}, stop_loss_price:{stop_loss_price}, current_profit:{current_profit}, trade={trade}')
                return 'exit_atr'
        
        return keep

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, 
                        **kwargs) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return 1

        last_candle = dataframe.loc[dataframe['date'] <= current_time]
        if last_candle.empty:
            return 1

        last_candle = last_candle.iloc[-1]
        atr = last_candle['atr'] # 实际策略运行过程中当前K线可能不完整，对应计算的ATR可能不正确，这里采用倒数第二根K线的计算值

        stop_loss_price = None

        min_rate = trade.min_rate if trade.min_rate is not None else trade.open_rate
        max_rate = trade.max_rate if trade.max_rate is not None else trade.open_rate
        
        if trade.is_short:
            stop_loss_price = min_rate + (self.atr_multiplier * atr)
        else:
            stop_loss_price = max_rate - (self.atr_multiplier * atr)

        trade.set_custom_data('atr_stop_loss', stop_loss_price)
        stop_loss_percent = (stop_loss_price - current_rate) / current_rate
        logger.info(f'Calculated stoploss for {pair}, trade_id:{trade.id}, current_time:{current_time}, current_rate:{current_rate}, stop_loss_percent:{stop_loss_percent}. ' \
                    f'after_fill:{after_fill}, min_rate:{min_rate}, max_rate:{max_rate}, atr:{atr}, stop_loss_price:{stop_loss_price}, current_profit:{current_profit}, trade={trade}')
        return stop_loss_percent
    
    @property
    def protections(self): # type: ignore
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 3
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.buy_leverage.value
