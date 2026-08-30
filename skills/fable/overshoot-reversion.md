# 过冲回归策略(Freqtrade 实现)

信号来源:薄订单簿被市价单/清算打穿,价格过冲后回归。**只做 maker 接单,禁止 taker 追入**——
taker 进出的成本(价差+双边费率)在冷门 pair 上就是 edge 本身的量级。
回测纪律见主 SKILL.md §2;本策略对其中"成交保守化"的依赖最重,单独强化见 §4。

## 1. 触发条件(populate_indicators / populate_entry_trend)

三个条件同时满足才算过冲(缺一即是普通波动或信息性移动):

```python
timeframe = "1m"
can_short = True
startup_candle_count = 300

@informative("1m", "BTC/USDT:USDT")   # leader 用于 β 过滤
def populate_indicators_btc(self, dataframe, metadata):
    dataframe["ret_1"] = dataframe["close"].pct_change()
    dataframe["sigma"] = dataframe["ret_1"].rolling(240).std()
    return dataframe

def populate_indicators(self, dataframe, metadata):
    df = dataframe
    df["ret_1"] = df["close"].pct_change()
    df["sigma"] = df["ret_1"].rolling(240).std()
    df["vol_z"] = (df["volume"] - df["volume"].rolling(240).mean()) / df["volume"].rolling(240).std()
    df["atr"] = ta.ATR(df, timeperiod=14)
    return df

def populate_entry_trend(self, dataframe, metadata):
    move_z = dataframe["ret_1"] / dataframe["sigma"]
    leader_z = dataframe["btc_usdt_usdt_1m_ret_1"] / dataframe["btc_usdt_usdt_1m_sigma"]
    idio = (move_z.abs() > self.m_sigma.value) \
         & (dataframe["vol_z"] > self.vol_z_min.value) \
         & (leader_z.abs() < self.leader_veto.value)      # β 移动否决:leader 同期大动不接
    dataframe.loc[idio & (move_z < 0), ["enter_long", "enter_tag"]] = (1, "os_long")
    dataframe.loc[idio & (move_z > 0), ["enter_short", "enter_tag"]] = (1, "os_short")
    return dataframe
```

- `m_sigma` 起点 4–6(过冲必须是极端事件,阈值低了全是噪音),`leader_veto` 起点 2。
- 有清算数据源时(交易所强平推送/第三方聚合),离线合并成事件列替代 vol_z,精度更高;
  没有就用量能尖峰近似,不要阻塞在数据源上。

## 2. Maker 接单实现

```python
order_types = {"entry": "limit", "exit": "limit",
               "stoploss": "market", "stoploss_on_exchange": False}
order_time_in_force = {"entry": "PO", "exit": "GTC"}   # 交易所支持 post-only 时
unfilledtimeout = {"entry": 3, "exit": 5, "unit": "minutes"}

def custom_entry_price(self, pair, trade, current_time, proposed_rate, entry_tag, side, **kwargs):
    df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    last = df.iloc[-1]
    k = self.entry_atr_k.value            # 起点 0.5–1.0
    if side == "long":
        return last["close"] - k * last["atr"]
    return last["close"] + k * last["atr"]
```

- 挂在过冲方向的**更深处**(close ∓ k×ATR):要么被继续过冲的 flow 打到(拿到更好价),
  要么不成交超时作废。绝不挂在回归方向追价。
- 分层接单:Freqtrade 单入场价,分层用 `position_adjustment_enable = True` +
  `adjust_trade_position` 在价格继续不利时按预定层级加仓(最多 2–3 层,每层预算固定,
  总敞口 = 初始 × ≤2)。层数与间距来自筛选期过冲深度分布的分位数,不是拍脑袋。
- 实盘 `confirm_trade_entry` 里查实时盘口(`dp.orderbook`):spread 异常放大或对手侧深度
  消失时放弃本次(盘口在逃跑,接单是接刀)。此检查回测不存在,属额外安全层。

## 3. 出场与风控

```python
stoploss = -0.03                      # 硬底,按 3× 门槛收益校准
use_custom_stoploss = True

def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
    held_min = (current_time - trade.open_date_utc).total_seconds() / 60
    return max(self.stoploss, -0.03 + 0.005 * (held_min // 5))   # 时间衰减容忍度

def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
    if (current_time - trade.open_date_utc).total_seconds() > self.t_max.value * 60:
        return "timeout"
    df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    if abs(df.iloc[-1]["ret_1"] / df.iloc[-1]["sigma"]) < 1.0:   # 波动归常≈回归完成
        return "reverted"
    return None
```

- 止损**用 market**:止损时刻付 taker 成本是正确的(此时你就是要立刻离场的一方)。
- 回归目标不要设过冲前原价:冷门 pair 常只回归 50–70%,出场目标用筛选期回归幅度分位数。

## 4. 回测成交保守化(本策略的生死线)

Freqtrade 回测对限价单是"K 线区间触及挂价即全额成交",没有排队——过冲瞬间你排在真实队列
最后,触及≠成交。强制两套跑法:

1. **触及版**:`custom_entry_price` 原样(close ∓ k×ATR)。这是上界。
2. **穿价版**:挂价额外让 `fill_buffer` bp(如 5–10bp,取该 pair 一个 tick 与
   spread 中位数的较大者),近似"价格必须穿过挂价才成交"。这是接近真实的下界。

两版结果都报告;**决策以穿价版为准**。若穿价版不过门槛(主 SKILL.md §2.3),该配置否决。
另:入场当根即触及止损的交易(entry 与 stop 同根)在 1m 回测中歧义最大,
统计其占比,>10% 时结果不可信,需重设层级/止损间距。

## 5. 陷阱清单

- 连续过冲(瀑布行情)逐层接满后继续下行:总敞口上限与硬止损是唯一防线,
  `adjust_trade_position` 里不允许突破预算的"摊平"逻辑。
- PO 单在实盘被拒(挂价越过对手价)→ Freqtrade 按超时/拒单流程作废,属正常路径;
  监控拒单率,过高说明 ATR 偏移量太小。
- `stoploss_on_exchange` 在冷门 pair 上易被插针扫掉,默认关闭,用引擎内止损;
  代价是断线风险,用全局最大回撤保护(protections)兜底。
- 同方向信号在相邻 K 线重复触发 → 用 `enter_tag` + cooldown protection 去重。
