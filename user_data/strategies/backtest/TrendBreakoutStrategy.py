from freqtrade.strategy import IStrategy
from freqtrade.strategy import (DecimalParameter, IntParameter)
from freqtrade.persistence import Trade

import freqtrade.vendor.qtpylib.indicators as qtpylib

from pandas import DataFrame
import pandas_ta as pta
from datetime import datetime
from typing import Optional, Tuple, Union

import logging
logger = logging.getLogger(__name__)

class TrendBreakoutStrategy(IStrategy):
    # ROI table:
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 5, default=3, space='buy')

    base_stop_loss = 0.04
    
    # Stoploss:
    stoploss = -base_stop_loss * buy_leverage.value
    # stoploss = -base_stop_loss * buy_leverage.value

    # Trailing stop:
    trailing_stop = False
    # trailing_stop_positive = 0
    # trailing_stop_positive_offset = 0
    # trailing_only_offset_is_reached = False

    position_adjustment_enable = True
    max_entry_position_adjustment = 3

    can_short = True
    
    minutes = 5
    timeframe = f"{minutes}m"

    buy_up_period = IntParameter(3, 15, default=5, space='buy')
    buy_up_period_count = IntParameter(3, 15, default=3, space='buy')

    # buy_up_ratio = DecimalParameter(1.000, 1.010, default=1.008, space='buy', decimals=3)

    buy_short_ema = IntParameter(30, 90, default=60, space='buy')
    buy_long_ema = IntParameter(200, 500, default=240, space='buy')

    buy_breakout_ratio = DecimalParameter(1.002, 1.010, default=1.005, decimals=3, space='buy')
    buy_highest_drawdown = DecimalParameter(0.970, 0.985, default=0.97, decimals=3, space='buy')
    buy_highest_period = IntParameter(30, 90, default=30, space='buy')
        
    buy_rsi = IntParameter(30, 70, default=50, space='buy')
    buy_adx = IntParameter(20, 40, default=25, space='buy')
    buy_lookback_period = IntParameter(10, 20, default=14, space='buy')
    
    startup_candle_count = buy_long_ema.value
    
    hyper = False

    def calculate_ema(self, df: DataFrame, p):
        df[f'ema_{p}'] = pta.ema(df['close'], p, False)
        df[f'ema_{p}_2nd'] = pta.ema(df[f'ema_{p}'], p, False)
        df[f'ema_{p}_3rd'] = pta.ema(df[f'ema_{p}_2nd'], p, False)
    
    def calculate_highest_lowest(self, df: DataFrame, p):
        df['period_high'] = df[['open', 'close']].max(axis=1)
        df['period_low'] = df[['open', 'close']].min(axis=1)
        df['highest'] = df['period_high'].rolling(window=p).max()
        df['lowest'] = df['period_low'].rolling(window=p).min()
        
    def calculate_indicators_hyper(self, df: DataFrame, metadata: dict):
        for p in self.buy_short_ema.range:
            self.calculate_ema(df, p)
        for p in self.buy_long_ema.range:
            self.calculate_ema(df, p)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.buy_leverage.value
    
    @property
    def protections(self):
        return  [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            }
        ]
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Union[Optional[float], Tuple[Optional[float], Optional[str]]]:
        return None
    
        if current_entry_profit == 0:
            # Return None for no action
            return None

        # Define profit thresholds and adjustment factors
        profit_levels = [
            (0.30, 2), 
            (0.20, 1.5),
            (0.10, 1.10),
            (0.0, 1.00),
            (-0.03, 0.75),
            (-0.06, 0.5),
            (-0.09, 0.25)
        ]
        
        for profit_threshold, factor in profit_levels:
            if current_entry_profit > profit_threshold:
                new_amount = trade.stake_amount * factor
                if new_amount > max_stake:
                    new_amount = max_stake
                if min_stake and new_amount < min_stake:
                    new_amount = min_stake
                logger.info(f"Adjusting position size to {new_amount} due to profit level of {current_entry_profit}")
                return new_amount, f'profit_adjustment_{profit_threshold}'

        # Return None for no action
        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.hyper:
            self.calculate_indicators_hyper(dataframe, metadata)
        
        self.calculate_ema(dataframe, self.buy_short_ema.value)
        self.calculate_ema(dataframe, self.buy_long_ema.value)
        self.calculate_highest_lowest(dataframe, self.buy_highest_period.value)
       
        dataframe['ema_short'] = dataframe[f'ema_{self.buy_short_ema.value}']
        dataframe['ema_short_uptrend'] = dataframe['ema_short'] > dataframe['ema_short'].shift(1)

        dataframe['ema_long'] = dataframe[f'ema_{self.buy_long_ema.value}']
        dataframe['ema_long_uptrend'] = dataframe['ema_long'] > dataframe['ema_long'].shift(1)

        dataframe['rsi'] = pta.rsi(dataframe['close'], length=self.buy_lookback_period.value)
        dataframe['adx'] = pta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=self.buy_lookback_period.value)[f'ADX_{self.buy_lookback_period.value}']
        dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.buy_lookback_period.value)
        dataframe['atr_stop_loss'] = dataframe['close'].shift(1) - dataframe['atr']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_up1 = dataframe['close'] >= (dataframe['ema_short'] * self.buy_breakout_ratio.value)
        ema_up2 = dataframe['close'] >= dataframe[f'highest']
        ema_up3 = dataframe['ema_short'] > dataframe['ema_long']
        ema_up4 = dataframe['ema_short_uptrend'].rolling(window=self.buy_up_period.value).sum() >= self.buy_up_period_count.value

        rsi_filter = dataframe['rsi'] < self.buy_rsi.value  # RSI过滤条件
        adx_filter = dataframe['adx'] > self.buy_adx.value  # ADX过滤条件

        # dataframe.loc[rsi_filter & adx_filter, 'enter_long'] = 1

        dataframe.loc[:, 'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_down1 = dataframe['close'] < (dataframe['ema_short'] * self.buy_highest_drawdown.value)
        ema_down2 = dataframe['close'] < (dataframe['highest'] * self.buy_highest_drawdown.value)
        ema_down3 = dataframe['ema_short_uptrend'].rolling(window=self.buy_up_period.value).sum() <= 3
        # dataframe.loc[ema_down1 | ema_down2 | ema_down3, 'exit_long'] = 1

        # dataframe.loc[dataframe['close'] < dataframe['atr_stop_loss'], 'exit_long'] = 1

        dataframe.loc[:, 'exit_long'] = 0
        
        return dataframe

