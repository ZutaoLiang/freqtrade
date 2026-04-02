from pandas import DataFrame
import pandas_ta as pta
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

from freqtrade.persistence.trade_model import Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_helper import stoploss_from_absolute, stoploss_from_open
from freqtrade.constants import Config

import logging
logger = logging.getLogger(__name__)


class SimpleTrendShortV1(IStrategy):
    
    timeframe = '1h'
    
    minimal_roi = {"0": 100}
    can_short = True
    process_only_new_candles = True
    
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
        self.base_stop_loss = self.get_config("base_stop_loss", 0.18)
        self.trade_leverage = self.get_config("trade_leverage", 5)
        self.stoploss = - float(self.base_stop_loss * self.trade_leverage)

        self.trailing_stop = self.get_config("trailing_stop", True)
        if not self.trailing_stop:
            self.custom_trailing_stop = self.get_config("custom_trailing_stop", False)
        else:
            self.trailing_stop_positive = self.get_config("base_trailing_stop", 0.12) * self.trade_leverage
            self.trailing_stop_positive_offset = self.get_config("base_trailing_stop_offset", 0.2) * self.trade_leverage
            self.trailing_only_offset_is_reached = self.get_config("trailing_only_offset_is_reached", True)
        
        self.use_custom_stoploss = self.get_config("use_custom_stoploss", False)
        
        self.atr_stop_loss_multiplier = self.get_config("atr_stop_loss_multiplier", 0)
        
        self.use_ha_candles = self.get_config("use_ha_candles", False)
        self.atr_period = self.get_config("atr_period", 21)
        self.trend_length = self.get_config("trend_length", 5)
        self.min_pct_change = self.get_config("min_pct_change", 0.15)
        self.entry_pct_change_natr_multiplier = self.get_config("entry_pct_change_natr_multiplier", 2.5)
        self.bearish_count_ratio = self.get_config("bearish_count_ratio", 0.8)
        
        self.startup_candle_count = int(max(self.atr_period, self.trend_length))
        
    def get_config(self, key: str, default):
        return self.config.get(key, default)
    
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
    
    def calc_ma(self, close, length: int):
        ma = pta.ema(close=close, length=length, talib=False)
        return ma.ffill() if ma is not None else ma

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            # haikinashi
            dataframe = self.calculate_ha(dataframe)
            
            # atr
            dataframe['atr'] = pta.atr(dataframe['ha_high'], dataframe['ha_low'], dataframe['ha_close'], length=self.atr_period)
            dataframe['natr'] = pta.natr(high=dataframe['ha_high'], low=dataframe['ha_low'], close=dataframe['ha_close'], length=self.atr_period, talib=False, scalar=1.0)
            
            dataframe['is_bearish'] = (dataframe['ha_close'] < dataframe['ha_open']).astype(int)
            
            dataframe['consecutive_bearish'] = 0
            consecutive_count = 0
            for i in range(len(dataframe)):
                if dataframe['is_bearish'].iloc[i] == 1:
                    consecutive_count += 1
                else:
                    consecutive_count = 0
                dataframe.loc[dataframe.index[i], 'consecutive_bearish'] = consecutive_count
            
            dataframe['total_bearish'] = dataframe['is_bearish'].rolling(window=self.trend_length).sum()
            
            dataframe['price_change_pct'] = (
                (dataframe['close'] - dataframe['close'].shift(self.trend_length)) / 
                dataframe['close'].shift(self.trend_length)
            )
            
            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_indicators: {e}")
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
        
        if dataframe.empty:
            return dataframe
        
        try:
            dataframe.loc[
                (
                    (dataframe['consecutive_bearish'] >= self.trend_length) | 
                    (dataframe['total_bearish'] >= self.bearish_count_ratio * self.trend_length)
                 ) &
                (dataframe['price_change_pct'] <= - self.entry_pct_change_natr_multiplier * dataframe['natr']) &
                (dataframe['price_change_pct'] <= - self.min_pct_change)
                ,
                'short_signal'
            ] = 1
            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_entry_trend: {e}")
            return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        if dataframe.empty:
            return dataframe
        
        try:
            return dataframe
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}::populate_exit_trend: {e}")
            return dataframe
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.trade_leverage

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        if not self.use_custom_stoploss:
            return None
        
        if self.atr_stop_loss_multiplier <= 0:
            return None
        
        leverage = trade.leverage
        is_short = trade.is_short
        open_rate = trade.open_rate
        _current_profit = current_profit / leverage

        last_candle = self.get_last_candle(trade)
        atr = last_candle['atr']
        natr = last_candle['natr']

        if after_fill:
            filled_orders = trade.select_filled_orders()
            count_of_orders = len(filled_orders)
            if count_of_orders == 0:
                return None
            
            if is_short:
                stop_rate_atr = open_rate + (self.atr_stop_loss_multiplier * atr) 
                stop_rate_abs = open_rate * (1 + self.base_stop_loss)
                stop_rate = min(stop_rate_atr, stop_rate_abs)
            else:
                stop_rate_atr = open_rate - (self.atr_stop_loss_multiplier * atr)
                stop_rate_abs = open_rate * (1 - self.base_stop_loss)
                stop_rate = max(stop_rate_atr, stop_rate_abs)
            
            if count_of_orders == 1:
                logger.info(f'Set {trade.pair} after fill #{count_of_orders} stoploss rate to:{stop_rate:.6f}'
                            f'(stop_rate_atr:{stop_rate_atr:.6f}, stop_rate_abs:{stop_rate_abs:.6f}), '
                            f'[new_open_rate:{open_rate:.6f}](stop/open dist:{abs(stop_rate/open_rate-1):.2%}, atr:{atr:.6f}, natr:{natr:.2%})'
                            f'current_rate:{current_rate:.6f}, '
                            f'current_profit:{current_profit:.2%}(without leverage:{_current_profit:.2%}) at {current_time}')
            else:
                logger.info(f'Set {trade.pair} after fill #{count_of_orders} stoploss rate to:{stop_rate:.6f}'
                            f'(stop_rate_atr:{stop_rate_atr:.6f}, stop_rate_abs:{stop_rate_abs:.6f}), '
                            f'[new_open_rate:{open_rate:.6f}](stop/open dist:{abs(stop_rate/open_rate-1):.2%}, atr:{atr:.6f}, natr:{natr:.2%})'
                            f'current_rate:{current_rate:.6f}, '
                            f'current_profit:{current_profit:.2%}(without leverage:{_current_profit:.2%}) at {current_time}')
            return stoploss_from_absolute(stop_rate, current_rate, is_short, leverage)
        
        open_hours = round((current_time - trade.open_date_utc).total_seconds() / 3600, 1)
        if open_hours >= 8:
            factor = np.log2(open_hours) / 2
            relative_stoploss = (0.02 * factor)
            if _current_profit > relative_stoploss:
                return stoploss_from_open(relative_stoploss * leverage, current_profit, is_short, leverage)
        
        if self.custom_trailing_stop:
            if _current_profit > self.get_config("base_trailing_stop_offset", 0.3):
                return self.get_config("base_trailing_stop", 0.12) * leverage
            
    def get_last_candle(self, trade: Trade):
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        return last_candle
