from freqtrade.strategy import IStrategy
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter)

import pandas as pd
from pandas import DataFrame
import numpy as np
import pandas_ta as pta
from datetime import datetime
from typing import Optional


class VolatilityStrategy(IStrategy):
    # ROI table:
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=3, space='buy')

    base_stop_loss = 0.10
    
    # Stoploss:
    stoploss = -base_stop_loss * buy_leverage.value

    # Trailing stop:
    trailing_stop = False
    # trailing_stop_positive = 0
    # trailing_stop_positive_offset = 0
    # trailing_only_offset_is_reached = False

    can_short = True
    
    minutes = 5
    timeframe = f"{minutes}m"

    buy_up_period = IntParameter(3, 15, default=7, space='buy')
    buy_up_ratio = DecimalParameter(1.000, 1.010, default=1.008, space='buy', decimals=3)

    buy_ema = IntParameter(30, 90, default=60, space='buy')
    
    atr_period = 10
    atr_multiplier = 2
    
    # buy_bbands_std = DecimalParameter(1.0, 2.5, default=1.0, space='buy', decimals=1)
    
    startup_candle_count = 100

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        
        for p in self.buy_ema.range:
            df[f'ema_{p}'] = pta.ema(df['close'], p, False)
            # for s in self.buy_bbands_std.range:
            #     bbands = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=p, stds=s)
            #     bb_upper = bbands['upper']
            #     bb_upper.name = f'bb_upper_{p}_{s}'
            #     df = pd.concat([df, bb_upper], axis=1)
            #     df.fillna(0)
            #     # print(df)

        heikinashi = qtpylib.heikinashi(df)
        df['ha_open'] = heikinashi['open']
        df['ha_high'] = heikinashi['high']
        df['ha_low'] = heikinashi['low']
        df['ha_close'] = heikinashi['close']

        # df['ha_open'] = df['open']
        # df['ha_high'] = df['high']
        # df['ha_low'] = df['low']
        # df['ha_close'] = df['close']

        df['atr'] = pta.sma(close=pta.true_range(df['ha_high'], df['ha_low'], df['ha_close']), length=self.buy_ema.value, talib=False)
        
        src = (df['ha_high'] + df['ha_low']) / 2
        
        df['up'] = src - (self.atr_multiplier * df['atr'])
        df['dn'] = src + (self.atr_multiplier * df['atr'])

        df['up_final'] = df['up']
        df['dn_final'] = df['dn']
        
        for i in range(1, len(df)):
            df.loc[i, 'up_final'] = max(df.loc[i, 'up'], df.loc[i-1, 'up_final']) if df.loc[i-1, 'close'] > df.loc[i-1, 'up_final'] else df.loc[i, 'up']
            df.loc[i, 'dn_final'] = min(df.loc[i, 'dn'], df.loc[i-1, 'dn_final']) if df.loc[i-1, 'close'] < df.loc[i-1, 'dn_final'] else df.loc[i, 'dn']
                
        df['trend'] = 1
        for i in range(1, len(df)):
            if df.loc[i-1, 'trend'] == 1 and df.loc[i, 'ha_close'] < df.loc[i-1, 'up_final']:
                df.loc[i, 'trend'] = -1
            elif df.loc[i-1, 'trend'] == -1 and df.loc[i, 'ha_close'] > df.loc[i-1, 'dn_final']:
                df.loc[i, 'trend'] = 1
            else:
                df.loc[i, 'trend'] = df.loc[i-1, 'trend']
        
        df['trend_entry'] = (df['trend'] == 1) & (df['trend'].shift(1) == -1)
        df['trend_exit'] = (df['trend'] == -1) & (df['trend'].shift(1) == 1)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_up1 = (dataframe[f'ema_{self.buy_ema.value}'] >= (self.buy_up_ratio.value * dataframe[f'ema_{self.buy_ema.value}'].shift(self.buy_up_period.value)))
        
        ema_up2 = (dataframe['close'] >= dataframe[f'ema_{self.buy_ema.value}'])
        
        # n天内连涨
        ema_up_mask = (dataframe[f'ema_{self.buy_ema.value}'] > dataframe[f'ema_{self.buy_ema.value}'].shift(1))
        for i in range(2, self.buy_up_period.value):
            ema_up_mask = ema_up_mask & (dataframe[f'ema_{self.buy_ema.value}'].shift(i-1) > dataframe[f'ema_{self.buy_ema.value}'].shift(i))
            
        dataframe.loc[(ema_up1 & ema_up2 & ema_up_mask), 'enter_long'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_dn1 = (dataframe['close'] < dataframe[f'ema_{self.buy_ema.value}'])
        # ema_dn1 = (dataframe[f'ema_{self.buy_ema.value}'] < (self.buy_up_ratio.value * dataframe[f'ema_{self.buy_ema.value}'].shift(self.buy_up_period.value)))
        # ema_dn2 = (dataframe['close'] < dataframe[f'bb_upper_{self.buy_ema.value}_{self.buy_bbands_std.value}'])
        
        dataframe.loc[(ema_dn1 | dataframe['trend_exit']), 'exit_long'] = 1
        
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.buy_leverage.value
