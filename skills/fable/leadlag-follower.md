# Lead-lag 跟随策略(Freqtrade 实现)

信号来源:leader(BTC/ETH 或同 symbol 的主导交易所)已发生的变动,目标冷门 pair 尚未跟随。
这是三类策略中信号最强、最先实现的一个。回测纪律见主 SKILL.md §2,此处不重复。

## 0. 适用性检查(先于一切代码)

- 筛选阶段测得的 lead-lag 峰值滞后必须 ≥ ~30–60 秒且逐周稳定。1m 收盘粒度下,
  滞后 < 30s 的关系在信号产生时已被吃掉,Freqtrade 做不了,直接放弃或转 Nautilus。
- leader 与目标 pair 需在**同一交易所配置**内可订阅(informative pairs 的限制)。
  跨所 leader 见 §5 进阶。

## 1. 数据与特征(populate_indicators)

```python
from freqtrade.strategy import IStrategy, informative

class LeadLagFollower(IStrategy):
    timeframe = "1m"
    can_short = True                      # futures 模式,双向
    startup_candle_count = 400            # 覆盖 σ 滚动窗 + 特征滞后
    process_only_new_candles = True

    @informative("1m", "BTC/USDT:USDT")   # leader;合约对写法以交易所为准
    def populate_indicators_btc(self, dataframe, metadata):
        dataframe["ret_1"] = dataframe["close"].pct_change()
        dataframe["ret_k"] = dataframe["close"].pct_change(self.lead_window.value)
        dataframe["sigma"] = dataframe["ret_1"].rolling(240).std()
        return dataframe

    def populate_indicators(self, dataframe, metadata):
        # informative 合并后 leader 列带后缀:btc_usdt_usdt_1m_ret_k 等
        df = dataframe
        df["t_ret_k"] = df["close"].pct_change(self.lead_window.value)
        # 跟随度:同窗口内 target 已实现变动占 leader 变动的比例
        df["follow_ratio"] = df["t_ret_k"] / df["btc_usdt_usdt_1m_ret_k"].replace(0, float("nan"))
        return df
```

要点:
- leader 触发强度用 **z-score**(`ret_k / sigma`)而非绝对阈值,跨市况稳定。
- `lead_window`(K 线数)来自筛选阶段测得的滞后,不是自由参数;hyperopt 只允许 ±1–2 根微调。
- β 校准:若目标 pair 对 leader 的 β 明显偏离 1,把 follow_ratio 换成
  `t_ret_k / (beta * leader_ret_k)`,β 用筛选期回归值,固定不滚动(避免引入前视)。

## 2. 入场(populate_entry_trend)

```python
def populate_entry_trend(self, dataframe, metadata):
    lead_z = dataframe["btc_usdt_usdt_1m_ret_k"] / dataframe["btc_usdt_usdt_1m_sigma"]
    not_followed = dataframe["follow_ratio"].fillna(0) < self.follow_max.value  # 如 0.3
    dataframe.loc[(lead_z > self.k_sigma.value) & not_followed, ["enter_long", "enter_tag"]] = (1, "ll_long")
    dataframe.loc[(lead_z < -self.k_sigma.value) & not_followed, ["enter_short", "enter_tag"]] = (1, "ll_short")
    return dataframe
```

- `k_sigma` 起点 2.5–3.5;由"该阈值下历史触发次数 × 单次门槛收益"反推,保证有足够样本又不稀释。
- 入场价:默认下根开盘 taker。**先验证 taker 版本**——若 taker 成本下无 edge,再切 maker:
  `order_types = {"entry": "limit", ...}` + `custom_entry_price` 返回当前 close(或 close 略优),
  `unfilledtimeout.entry` 设 1–2 分钟,超时即弃(信号是时效性的,不追)。
  交易所支持时 `order_time_in_force = {"entry": "PO", "exit": "GTC"}`。

## 3. 出场

跟随完成或超时,二者取先:

```python
def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
    if (current_time - trade.open_date_utc).total_seconds() > self.t_max.value * 60:
        return "timeout"
    df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    last = df.iloc[-1]
    if last["follow_ratio"] >= self.follow_done.value:   # 如 0.8:跟随基本完成
        return "converged"
    return None
```

- `minimal_roi` 关闭(`{"0": 100}`),出场逻辑全部显式;`stoploss` 设为
  2–3× 单次门槛收益的硬止损,`custom_stoploss` 可做时间衰减(持仓越久容忍越低)。
- `t_max` 起点 = 测得滞后的 2–3 倍,通常 2–5 分钟。

## 4. 回测与验证要点(在主 SKILL.md §2 之上)

- Freqtrade 的内生延迟(收盘出信号→下根开盘成交)对本策略是**特性不是缺陷**:
  它强制 edge 至少存活一根 K 线。若回测收益全部来自"信号当根",说明滞后测量有误,重测。
- 分 `enter_tag` / 分事件强度(lead_z 分桶)看收益分布:健康形态是 z 越大单笔越好;
  若倒挂,说明大 z 事件是信息性移动、冷门 pair 同步定价了,该 pair 的滞后关系已失效。
- leader 数据缺口(informative 缓存不足)会让 follow_ratio 为 NaN 静默不触发:
  回测后核对触发次数与筛选阶段预估的量级一致。

## 5. 进阶:跨所 leader(producer/consumer)

主导所与交易所不同时:起一个 producer 实例连主导所只产出分析后的 dataframe,
consumer 实例(实际交易)经 websocket 消费其列(`ExternalMessageConsumer` 配置,
文档 Producer/Consumer mode 章节)。注意:
- 回测无法复现 producer 链路,只能用"把主导所历史 K 线离线合并进数据"的方式近似,
  实盘与回测的数据路径不同,dry-run 对账(主 SKILL.md §2.6)在此场景是硬门槛。
- 两实例间延迟(秒级)直接吃掉滞后预算,测得滞后 < 2 分钟时不建议用此方案。

## 6. 陷阱清单

- informative 列名后缀拼错 → 全 NaN → 永不触发且无报错;开发时先 print 列名。
- `follow_ratio` 在 leader 微小变动时爆炸:必须配合 lead_z 阈值使用,单独判跟随度无意义。
- 双向合约下 short 的资金费率与借贷成本进 PnL(futures 模式回测已含 funding,现货杠杆另算)。
- 同一 leader 事件在多个目标 pair 同时触发 → 仓位相关性 1:`max_open_trades` 收紧,
  或按事件 id 去重(protections/自定义确认里限制同窗口开仓数)。
