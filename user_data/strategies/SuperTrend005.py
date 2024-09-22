from freqtrade.strategy import IStrategy
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter)

import pandas as pd
from pandas import DataFrame
import numpy as np
import pandas_ta as pta
from datetime import datetime
from typing import Optional


class SuperTrend005(IStrategy):
    # ROI table:
    minimal_roi = {"0": 100}

    buy_leverage = IntParameter(1, 3, default=1, space='buy')

    # Stoploss:
    stoploss = -0.05 * buy_leverage.value

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.10
    trailing_stop_positive_offset = 0.144
    trailing_only_offset_is_reached = False

    can_short = True

    minutes = 5
    timeframe = f"{minutes}m"

    atr_period = 10
    atr_multiplier = 2.0

    buy_ema = IntParameter(48, 96, default=90, space='buy')
    
    buy_bbands_period = IntParameter(20, 60, default=48, space='buy')
    buy_bbands_std = DecimalParameter(2, 3, default=2, space='buy', decimals=1)
    
    startup_candle_count = 100

    def typical_price(bars):
        res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
        return pd.Series(index=bars.index, data=res)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        heikinashi = qtpylib.heikinashi(df)
        df['ha_open'] = heikinashi['open']
        df['ha_high'] = heikinashi['high']
        df['ha_low'] = heikinashi['low']
        df['ha_close'] = heikinashi['close']
        
        df['atr'] = pta.sma(close=pta.true_range(df['ha_high'], df['ha_low'], df['ha_close']), length=self.atr_period, talib=False)
        
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

        # df[f'ema_{self.buy_ema.value}'] = pta.ema(df['ha_close'], self.buy_ema.value, False)
        
        for val in self.buy_ema.range:
            df[f'ema_{val}'] = pta.ema(df['ha_close'], val, False)
        
        for p in self.buy_bbands_period.range:
            for s in self.buy_bbands_std.range:
                bbands = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=p, stds=s)
                bb_band_diff = bbands['upper'] - bbands['mid']
                bb_band_diff.name = f'bb_band_diff_{p}_{s}'
                df = pd.concat([df, bb_band_diff], axis=1)
                # print(df)
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_up1 = dataframe['ha_close'] > (dataframe[f"ema_{self.buy_ema.value}"] + dataframe[f"bb_band_diff_{self.buy_bbands_period.value}_{self.buy_bbands_std.value}"])
        ema_up2 = (dataframe[f'ema_{self.buy_ema.value}'] >= dataframe[f'ema_{self.buy_ema.value}'].shift(1)) & (dataframe[f'ema_{self.buy_ema.value}'].shift(1) >= dataframe[f'ema_{self.buy_ema.value}'].shift(2))
        
        dataframe.loc[(ema_up1 & ema_up2), 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe['trend_exit']), 'exit_long'] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.buy_leverage.value
