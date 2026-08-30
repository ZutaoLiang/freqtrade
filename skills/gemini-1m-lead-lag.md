# Skill 02: BTC Lead-Lag（领头羊滞后动能）策略

> **英文名称**：Cross-Asset High-Frequency Lead-Lag Momentum & Orderbook Transmission Engine  
> **策略类型**：跨资产毫秒/秒级动能传导（Cross-Asset Momentum / Lead-Lag）  
> **时间周期**：实时微观成交流（WebSocket `aggTrade`），秒级响应，1m 内完成平仓  
> **版本编号**：v1.0.0 Pro  

---

## 1. 策略核心概述与设计原理

比特币（BTC）作为加密货币市场的绝对流动性中心，当其发生巨鲸脉冲扫盘或订单簿剧烈波动时，做市商与高频算法会优先修正 BTC 的盘口。而高 Beta 山寨币（如 SOL、DOGE、SUI、PEPE 等）的订单簿在微观结构上存在秒级到十秒级的“价格吸收与传导延迟”。

本策略利用 WebSocket 实时监听 BTC 永续合约的 `aggTrade` 逐笔成交流，检测真金白银的量能脉冲冲击。一旦确认 BTC 发生强动能突破且目标山寨币处于**滞后未跟涨/跟跌状态**时，策略以毫秒级速度抢跑下单，捕获山寨币后续补涨/补跌的微观动能溢价。

> **💡 关键优势 (Core Edge)**  
> 避开 BTC 本身高精尖的 HFT 盘口争夺，利用山寨币微观流动性传导的滞后窗口（Lag Window），实现跨资产的极短持仓高频获利。

---

## 2. 信号捕捉：BTC 主动成交冲击 (Signal Engine)

策略不等待 1m K 线收盘，必须在 1m K 线内部实时监听 Binance BTC 永续合约的 `aggTrade` WebSocket 数据流，建立滑动时间窗口（如 3 秒滑动窗口）：

### 2.1 信号触发条件 (Trigger Conditions)

1. **主动买/卖量脉冲（Taker Volume Spike）**：
   $$\text{TakerBuyVol}_{3s} > 5 \times \overline{\text{TakerBuyVol}}_{1h}$$
2. **微观价格动能（Micro-Momentum）**：
   $$\Delta P_{3s}(\text{BTC}) \ge +0.15\% \quad (\text{做多}) \quad \text{或} \quad \Delta P_{3s}(\text{BTC}) \le -0.15\% \quad (\text{做空})$$
3. **CVD 累积买卖差真伪验证（Volume Delta Confirmation）**：
   $$\text{CVD}_{3s} = \sum \text{Vol}_{\text{buyer\_is\_taker}} - \sum \text{Vol}_{\text{seller\_is\_taker}}$$
   * 必须满足 $\text{CVD}_{3s} > 0$（做多）或 $\text{CVD}_{3s} < 0$（做空），排除因盘口撤单或流动性抽干引发的虚假价格脉冲。

---

## 3. 目标资产池与残差过滤 (Universe Filtering)

### 3.1 动态资产池筛选 (Universe Screening)

为了确保传导效应成立，目标山寨币池必须满足以下动态过滤门槛：

| 维度 | 指标 / 筛选方法 | 准入门槛 / 参数配置 | 设计意图 |
| :--- | :--- | :--- | :--- |
| **相关性** | 1 小时 Rolling 收益率相关系数 $R$ | $R(\text{ALT}, \text{BTC}) > 0.85$ | 确保标的与大盘存在强联动性。 |
| **弹性放大** | 1 小时 Rolling Beta 系数 $\beta$ | $\beta_{\text{alt}} > 1.5$ | 优先选择价格弹性大、补涨幅更高的标的。 |
| **流动性底线** | 1m 成交额中位数 | $> 20,000 \text{ USDT}$ | 保证市价单冲击时不会产生恶性滑点。 |

### 3.2 滞后窗口确认 (Lag Readiness Test)

> ⚠️ **实证警告（残差化检验）**  
> 仅监控山寨币原始跟随收益（如 BTC +0.4% 后山寨币跟随 +0.44%）大部分属于 Beta 共动，扣除 Taker 成本后收益不可变现。必须对山寨币进行残差化处理：

$$\text{Residual\_Return}_{\text{alt}} = \Delta P_{\text{forward}}(\text{ALT}) - \beta_{\text{alt}} \times \Delta P_{\text{forward}}(\text{BTC})$$

**滞后成立条件**：当 BTC 3秒内已拉升 $+0.15\%$ 时，目标山寨币在过去 3 秒内的涨幅必须 $< +0.03\%$，确认其处于明显的**传导滞后断层**状态。

---

## 4. 交易执行与出场逻辑