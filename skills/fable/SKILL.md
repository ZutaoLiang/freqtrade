---
name: freqtrade-niche-pair
description: Implement and backtest niche-pair (low-competition crypto) strategies in Freqtrade at 1m candle granularity — lead-lag following via informative pairs, overshoot reversion with maker limit entries, and event-driven (funding/listing) plays. Use this skill whenever the user builds, backtests, or dry-runs these strategies in Freqtrade, configures pairlists for a screened universe, injects realistic costs into backtests, or asks about fill assumptions, custom_entry_price, informative pairs, producer/consumer mode, or FreqAI features for lead-lag — including Chinese phrasings like 冷门交易对、领先滞后、过冲回归、资金费率、回测撮合、挂单成交假设.
---

# 冷门交易对策略:Freqtrade 实现与回测

本 skill 把冷门交易对的三类分钟级策略落到 Freqtrade 上,并规定回测必须遵守的成本与成交假设。
筛选方法学(三阶段 pipeline、统计验证)沿用 `niche-pair-trading` skill 的定义;本 skill 只负责
**Freqtrade 侧的落地**:筛选结果如何进 pairlist、策略如何写、回测如何做才不自欺。

## 0. 能力边界(先想清楚 Freqtrade 版本做的是什么)

Freqtrade 是 K 线驱动框架:信号在收盘价上计算,实盘主循环秒级轮询,回测按 OHLCV 撮合。
因此这里实现的是三类策略的 **1 分钟收盘粒度近似版**,定位是:验证 edge 在 1m 粒度 + 保守成本下
是否存活,并快速迭代;不是微观结构级实现。

| 能做 | 不能做(去 NautilusTrader / `niche-pair-trading` skill) |
|---|---|
| 1m 收盘信号、下根开盘/限价成交 | 亚秒级 lead-lag、tick 级触发 |
| informative pairs 引入 leader(同所) | 订单簿级回测撮合、maker 排队建模 |
| 限价/post-only 入场 + 超时撤单 | 跨所双腿对冲套利 |
| futures 模式含资金费率回测 | 盘口指标(刷新率/深度恢复)的回测复现 |
| producer/consumer 跨实例喂数据(进阶) | 真实逆向选择成本估计(只能实盘 dry-run 对账) |

**一条判断规则**:若 §2 统计验证测得的 lead-lag 峰值滞后 < ~30 秒,或策略必须依赖盘口状态,
Freqtrade 版本不会有结果,不要硬做,直接建议走 Nautilus 路线。

## 1. 筛选结果 → pairlist 集成

筛选 pipeline 本身在 Freqtrade 外运行(离线 polars 批处理)。集成方式:

- **实盘/dry-run**:外部筛选器每周产出 JSON(`{"pairs": [...], "refresh_period": 86400}`),
  bot 用 `RemotePairList` 指向该文件/URL 作为首位 handler;或直接改 `StaticPairList` 的
  `pair_whitelist`。粗筛(Stage 1)可叠加内置 handler 兜底:`VolumePairList`(成交量带,
  min/max 参数以当前版本文档为准)→ `AgeFilter`(`min_days_listed: 7`)。
- **回测**:一律 `StaticPairList` + 显式 whitelist,保证可复现。ticker 类动态 handler 回测不可用;
  candle-lookback 模式的 VolumePairList 虽可用,但结果随取数窗口漂移,不作为正式回测配置。
- 交易所黑名单(`pair_blacklist`)维护:已知刷量/控盘 pair、杠杆代币 `.*(UP|DOWN|BULL|BEAR)/.*`。

## 2. 共享回测纪律(所有策略文档都引用本节,违反即结果无效)

Freqtrade 回测按 K 线价格撮合、限价单**触及即成交**、无排队、无价差。冷门 pair 价差 10–40bp,
这些默认假设会系统性高估结果。强制规则:

1. **成本注入**:回测 `--fee` 设为 `真实单边费率 + 半价差估计`(半价差取筛选期该 pair
   spread 中位数的一半;分层不便时取池内 75 分位做统一保守值)。理由:Freqtrade 无 spread 模型,
   把它折进 fee 是唯一全局手段。报告中注明注入值。
2. **成交保守化**:所有限价入场按各策略文档的"穿价规则"打折(挂价再让 X bp),
   并同时跑"触及成交"与"穿价成交"两套,真实结果按靠近后者解读。
3. **门槛判定**:平均单笔毛收益 < 3×(注入后单边成本×2) 的配置,直接判不可行,
   不进入 hyperopt。hyperopt 只允许微调已由测量确定的参数(±小范围),禁止大网格搜索——
   冷门 pair 样本少,大网格必然过拟合。
4. **切分验证**:按时间 walk-forward(如 3 个月训练窗滚动 1 个月),各窗口独立报告;
   `--breakdown month` 检查收益是否集中在个别事件。
5. **timeframe 细化**:基准 `timeframe = "1m"`。若策略用更高 timeframe 聚合信号,
   回测必须加 `--timeframe-detail 1m` 还原盘中路径。
6. **dry-run 对账(上线前必做)**:dry-run ≥2 周,对比每笔成交的
   markout(1s/5s/30s/5min)与回测隐含预期;实盘 markout 显著更差 = 成交假设仍太乐观,
   回到第 2 条加码,而不是调策略参数。
7. 版本确认:动手前 `freqtrade --version`,API 以 https://www.freqtrade.io/en/stable/ 为准
   (月度发版,callback 签名偶有变更);本 skill 代码骨架只表达结构意图。

## 3. 策略路由(不同策略读不同文档)

| 用户在做什么 | 读 |
|---|---|
| leader(BTC/ETH/主导所)带动、目标 pair 滞后跟随 | `references/leadlag-follower.md` |
| 大单/清算打穿后的回归、maker 接单 | `references/overshoot-reversion.md` |
| 资金费率结算、上新、价差极值等结构性时点 | `references/event-driven.md` |
| 资金费极值的**已实测**分支:站在付费方做动量 | `references/funding-skew-momentum.md` |

只读当前任务相关的文档;各文档共享本文件 §2 的回测纪律,不再重复。

`funding-skew-momentum.md` 与前三份性质不同:它记录的是在本地 1m perp 数据上**实测出来的**
结果,包含前三份中两份的否决记录(lead-lag 滞后不足、过冲回归训练段为负)与
资金费 carry 的否决记录。**动手做本 skill 任何一条策略线之前先读它的 §0**,避免重做已否决的工作;
它的 §6 是四条实测补充的回测纪律,与本文件 §2 同等强制;§10 是 dry-run 运行手册。

## 4. 触发本 skill 时的执行清单

1. 判断任务属于哪份策略文档,读取后再写代码;涉及回测先核对 §2 全部 7 条。
2. 用户给出回测结果时:先问/查 fee 注入值与 fill 规则,未做成本注入的结果不解读收益,
   只指出需要重跑。
3. 信号逻辑改动后,提醒 `startup_candle_count` 与 informative 数据窗口是否够长。
4. 任何需要盘口/亚秒能力的需求,明确说 Freqtrade 不适合并指回 `niche-pair-trading` skill,
   不要用 hack 绕。
