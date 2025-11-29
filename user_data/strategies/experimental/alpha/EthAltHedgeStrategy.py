import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta, timezone
from freqtrade.strategy import IStrategy
from freqtrade.constants import Config
from freqtrade.persistence import Trade
import talib.abstract as ta
from typing import Optional


class EthAltHedgeStrategy(IStrategy):
    """
    ETH vs Altcoin Basket Hedge Strategy (Portfolio Based)
    周期: 15m
    回顾: 6h (24 candles)
    风控: 基于组合总盈亏 + 价差回归
    """

    # --------------------------------------------------------------------------
    # 策略基础设置
    # --------------------------------------------------------------------------
    
    # [关键] 禁用默认的单单止盈止损，完全由 custom_exit 接管
    minimal_roi = { "0": 100.0 }
    stoploss = -1.0 

    # 启用合约做空
    can_short = True
    
    # 基础时间周期 15分钟
    timeframe = '15m'

    # 回顾周期: 6小时 = 24 * 15m
    

    # 定义山寨币篮子权重 (总和建议为 1.0)
    BASKET_WEIGHTS = {
        'SOL/USDT:USDT': 0.30,
        'OP/USDT:USDT': 0.20,
        'SUI/USDT:USDT': 0.20,
        'DOGE/USDT:USDT': 0.15,
        'UNI/USDT:USDT': 0.15
    }
    
    ETH_PAIR = 'ETH/USDT:USDT'


    # --------------------------------------------------------------------------
    # Hyperopt 参数 (可优化)
    # --------------------------------------------------------------------------

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
        # 全局风控参数 (基于总资金)
        self.global_tp_pct = self.config.get("global_tp_pct", 0.01)
        self.global_sl_pct = self.config.get("global_sl_pct", -0.15)
        
        # 趋势判断均线: 15m
        self.trend_ma_period = self.config.get("trend_ma_period", 24)
        
        # 开仓价差阈值: 0.01 代表 1% 的背离
        self.spread_threshold = self.config.get("spread_threshold", 0.01)

        self.lookback_candles = self.config.get("lookback_candles", 24)
        
        self.trade_leverage = self.config.get("trade_leverage", 1)
        
        self.startup_candle_count = int(self.trend_ma_period * 1.2)
        
    # --------------------------------------------------------------------------
    # 1. 数据加载
    # --------------------------------------------------------------------------
    def informative_pairs(self):
        # 加载 ETH 和所有篮子代币
        pairs = [self.ETH_PAIR] + list(self.BASKET_WEIGHTS.keys())
        return [(pair, self.timeframe) for pair in pairs]

    # --------------------------------------------------------------------------
    # 2. 指标计算 (Index & Spread)
    # --------------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # --- A. 当前交易对指标 ---
        # 趋势均线
        dataframe['trend_ema'] = ta.EMA(dataframe, timeperiod=self.trend_ma_period)
        # 当前对 6小时涨跌幅
        dataframe['pct_change_6h'] = dataframe['close'].pct_change(periods=self.lookback_candles)

        if not self.dp:
            return dataframe

        # --- B. 获取 ETH 数据 ---
        eth_df = self.dp.get_pair_dataframe(pair=self.ETH_PAIR, timeframe=self.timeframe)
        eth_df['eth_pct_change'] = eth_df['close'].pct_change(periods=self.lookback_candles)
        eth_df['eth_trend_ema'] = ta.EMA(eth_df, timeperiod=self.trend_ma_period)
        
        # 合并 ETH 数据
        temp_eth = eth_df[['date', 'eth_pct_change', 'close', 'eth_trend_ema']].copy()
        temp_eth.columns = ['date', 'eth_pct_change', 'eth_close', 'eth_trend_ema']
        dataframe = dataframe.merge(temp_eth, on='date', how='left')

        # 标记 ETH 趋势状态
        dataframe['is_eth_uptrend'] = dataframe['eth_close'] > dataframe['eth_trend_ema']
        dataframe['is_eth_downtrend'] = dataframe['eth_close'] < dataframe['eth_trend_ema']

        # --- C. 计算山寨币指数 ---
        dataframe['alt_index_change'] = 0.0
        
        for pair, weight in self.BASKET_WEIGHTS.items():
            alt_df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            alt_pct = alt_df['close'].pct_change(periods=self.lookback_candles)
            
            temp_alt = alt_df[['date']].copy()
            temp_alt[f'pct_{pair}'] = alt_pct
            
            dataframe = dataframe.merge(temp_alt, on='date', how='left')
            # 累加指数涨跌幅
            dataframe['alt_index_change'] += dataframe[f'pct_{pair}'].fillna(0) * weight

        # --- D. 计算核心价差 (Spread) ---
        # Spread > 0: ETH 强
        # Spread < 0: 山寨 强
        # 增加 EMA 平滑防止信号闪烁
        dataframe['spread_raw'] = dataframe['eth_pct_change'] - dataframe['alt_index_change']
        dataframe['spread'] = ta.EMA(dataframe['spread_raw'], timeperiod=3) 

        return dataframe

    # --------------------------------------------------------------------------
    # 3. 进场逻辑
    # --------------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        
        # 逻辑 A: 避险/抗跌 (ETH 下行趋势，但 Spread 显示 ETH 更强/跌得少)
        # 行为: 做多 ETH，做空 山寨
        condition_crash_hedge = (
            (dataframe['is_eth_downtrend']) & 
            (dataframe['spread'] > self.spread_threshold)
        )

        # 逻辑 B: 山寨季/补涨 (ETH 上行趋势，但 Spread 显示 山寨 更强/涨得多)
        # 行为: 做空 ETH，做多 山寨
        condition_altseason = (
            (dataframe['is_eth_uptrend']) &
            (dataframe['spread'] < -self.spread_threshold)
        )

        if pair == self.ETH_PAIR:
            dataframe.loc[condition_crash_hedge, 'enter_long'] = 1
            # dataframe.loc[condition_altseason, 'enter_short'] = 1
        
        elif pair in self.BASKET_WEIGHTS:
            dataframe.loc[condition_crash_hedge, 'enter_short'] = 1
            # dataframe.loc[condition_altseason, 'enter_long'] = 1

        return dataframe

    # --------------------------------------------------------------------------
    # 4. 出场逻辑 (不使用 populate_exit_trend，完全由 custom_exit 接管)
    # --------------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 这里留空，逻辑全部在 custom_exit 处理
        return dataframe

    # --------------------------------------------------------------------------
    # 5. [核心] 组合级风控与离场
    # --------------------------------------------------------------------------
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        
        # --- A. 检查“单腿裸奔”风险 (Legging Risk) ---
        # 如果只有 ETH 没有山寨，或者只有山寨没有 ETH，且持续时间超过 30分钟，强制平仓
        unhedged_reason = self.check_unhedged_risk()
        if unhedged_reason:
            return unhedged_reason

        # --- B. 计算组合总盈亏 (Portfolio PnL) ---
        portfolio_pnl_pct = self.get_portfolio_pnl_pct(current_rate, trade)
        
        # 全局止盈
        if portfolio_pnl_pct >= self.global_tp_pct:
            return f"global_tp_{portfolio_pnl_pct:.2%}"
        
        # 全局止损
        if portfolio_pnl_pct <= self.global_sl_pct:
            return f"global_sl_{portfolio_pnl_pct:.2%}"

        # --- C. 价差回归离场 (Alpha Logic) ---
        # 获取当前 K 线数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) > 0:
            current_spread = dataframe.iloc[-1]['spread']
            
            # 逻辑：当 Spread 回归到 0 轴附近，说明套利空间消失
            
            if pair == self.ETH_PAIR:
                # 多 ETH：Spread 不再大于 0
                if not trade.is_short and current_spread <= 0:
                    return "spread_reverted_eth_long"
                # 空 ETH：Spread 不再小于 0
                if trade.is_short and current_spread >= 0:
                    return "spread_reverted_eth_short"
            
            elif pair in self.BASKET_WEIGHTS:
                # 空 山寨：Spread 不再大于 0
                if trade.is_short and current_spread <= 0:
                    return "spread_reverted_alt_short"
                # 多 山寨：Spread 不再小于 0
                if not trade.is_short and current_spread >= 0:
                    return "spread_reverted_alt_long"

        return None

    # --------------------------------------------------------------------------
    # 辅助函数：计算组合盈亏
    # --------------------------------------------------------------------------
    def get_portfolio_pnl_pct(self, current_rate, current_trade):
        if self.dp.runmode.value in ('backtest', 'plot', 'hyperopt'):
            return 0.0
        
        trades = Trade.get_trades([Trade.is_open.is_(True)]).all()
        if not trades:
            return 0.0

        total_profit_abs = 0.0
        total_stake = 0.0

        for t in trades:
            # 如果是当前触发回调的单子，用最新推送的价格
            rate = current_rate if t.id == current_trade.id else t.current_rate
            if rate is None: continue
            
            # 计算盈亏金额
            profit_ratio = t.calc_profit_ratio(rate)
            total_profit_abs += profit_ratio * t.stake_amount
            total_stake += t.stake_amount

        if total_stake == 0: return 0.0
        return total_profit_abs / total_stake

    # --------------------------------------------------------------------------
    # 辅助函数：检查裸奔风险
    # --------------------------------------------------------------------------
    def check_unhedged_risk(self):
        if self.dp.runmode.value in ('backtest', 'plot', 'hyperopt'):
            return None
        
        trades = Trade.get_trades([Trade.is_open.is_(True)]).all()
        if not trades: return None

        has_eth = any(t.pair == self.ETH_PAIR for t in trades)
        has_alts = any(t.pair in self.BASKET_WEIGHTS for t in trades)

        # 正常：都有 或 都没有
        if (has_eth and has_alts) or (not has_eth and not has_alts):
            return None

        # 异常：只有一边
        # 检查最早开仓时间
        oldest_trade = min(trades, key=lambda t: t.open_date_utc)
        # 获取当前 UTC 时间
        now_utc = datetime.now(timezone.utc)
        
        # 如果开仓超过 45分钟 (3根K线) 依然没有配对成功，强制平仓
        if (now_utc - oldest_trade.open_date_utc).total_seconds() > 45 * 60:
            return "emergency_exit_unhedged"
        
        return None

    # --------------------------------------------------------------------------
    # 6. 动态仓位管理
    # --------------------------------------------------------------------------
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        
        total_wallet = self.wallets.get_total_stake_amount()
        
        # 目标仓位: ETH 50%, 山寨总共 40% (留 10% 缓冲)
        TARGET_ETH_RATIO = 0.50
        TARGET_ALT_RATIO = 0.50

        if pair == self.ETH_PAIR:
            return total_wallet * TARGET_ETH_RATIO
        
        elif pair in self.BASKET_WEIGHTS:
            weight = self.BASKET_WEIGHTS.get(pair, 0)
            return total_wallet * TARGET_ALT_RATIO * weight

        return proposed_stake
    
    @property
    def protections(self): # type: ignore
        protections = []
        
        cooldown_candles = self.config.get("cooldown_candles", 0)
        if cooldown_candles > 0:
            protections.append(
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": cooldown_candles,
                }
            )
            
        return protections
    
    def leverage(
        self, 
        pair: str, 
        current_time: datetime, 
        current_rate: float, 
        proposed_leverage: float, 
        max_leverage: float, 
        entry_tag: Optional[str], 
        side: str, 
        **kwargs
    ) -> float:
        return self.trade_leverage
    