---
name: high-volume-trend-research
description: 高成交量趋势跟踪（HighVolumeFourMtfV1 / MainWave V20-V23）研究的既有结论、复现口径和证伪记录。当任务涉及这一策略族的回测、调参、币池轮动、锚点相位、仓位倍率拟合或"能否上实盘"判断时使用。Covers the high-volume trend-following backtest research: what has already been tested and falsified, how to reproduce it, and which conclusions must not be re-derived.
---

# 高成交量趋势跟踪研究

## 一句话结论

**这条策略线在 2026 年样本内的高收益不可迁移。** 2025 年同口径全年 `-18.98%`、PF `0.83`；2026 年 `+345.86%` 的收益 97% 来自两个信号档、45 笔交易，而这两档正是 2025 年亏损最大的两档。等风险基线和独立事件研究都显示底层突破信号没有可用 edge。**不要在这条信号线上继续加参数。**

## 使用方式

- 有人要求"再调一版倍率/再加一个档位/再换一个锚点"时，先读本文件的[证伪清单](#证伪清单)。已经跑过的不要重跑。
- 需要具体数字、表格、逐档盈亏时读 `reference/2026-08-review-findings.md`。
- 需要重跑统计检验时用 `scripts/analyze_breakout_signal_event_study.py`（不依赖回测框架，几分钟出结果）。

## 证伪清单

按结论强度排序。每条都已实测，勿重复。

### 1. 收益集中在两个档位，且这两档跨年反号

2026 年 1 日锚点 `+1729 USDT` 总盈利里：

| 档位 | 2026 笔数 | 2026 盈亏 | 2025 笔数 | 2025 盈亏 |
|---|---:|---:|---:|---:|
| `1h_highvolume_mainwave_breakout_short` | 25 | **+1010.08** | 15 | **-72.57** |
| `1h_persistent_mainwave_breakout_long` | 20 | **+672.99** | 11 | **-53.23** |

这两档就是被 `refit_parameter_names` 拟合了 stake 倍率的两档。45 笔交易撑起 97% 的收益，2025 年同样两档是亏损榜前二。这不是策略，是两年一次的样本。

### 2. 仓位倍率拟合结果跨年完全反向

同一个 9 点网格（`pretrend_short_minimum_rvol` × `persistent_mainwave_long_multiplier` × `strong_short_stake_multiplier`）：

- 2026 最优在网格**最大角**（`off_l200_s175`，`+473.57%`），网格跨度 111 个百分点；
- 2025 最优在网格**最小角**（`off_l150_s125`，`-18.36%`），网格跨度 4.25 个百分点，**九个点全负**；
- 两年 9 个点的 Spearman 相关 `-0.067`。

拟合面在 2025 年是平的且全负——没有信号可拟合，只有噪声可放大。

### 3. scout 档不是"无用交易"，是隐式风险闸门

最小仓位 scout 占 79% 的槽位分钟数、盈亏近零，直觉上应该删掉。实测删掉后（`ScoutFreeV23`）四锚点平均最大回撤从 `16.05%` 涨到 `27.90%`，PF 从 1.73 掉到 1.33。

同一机制解释了另外三个实验为什么都变差：六槽（`SixSlotV23`）、市值排除放宽到 top-50、以及任何"释放槽位"的改动。**当前的实际风险水平是槽位争用的副产品，不是规则。**

把闸门显式化（`RiskSlots{1,2,3}V23` 限制并发风险仓数量）后：

- R=3 与基线**逐笔完全一致**——上限从不触发；
- R=2 几乎一致；
- R=1 把 1 日锚点从 `+345.86%` 砍到 `+99.29%`，但把最差锚点的回撤从 20.20% 降到 11.20%。

即并发风险仓 93% 以上时间只有 0 或 1 个。四槽这个数字本身没有意义。

### 4. 锚点相位不是过拟合，但"全正"只在样本内成立

跑满全部 28 个刷新锚点：

- 全期收益中位数 `141.48%`，原来的 4 锚点样本中位数 `141.76%` —— 4 锚点抽样**无偏**；
- 全期 **0/28 亏损**；
- 但切到 2026-06 之后的样本外段：**9/28 亏损**，训练段与测试段收益的 Pearson `0.092`、Spearman `-0.171`。

锚点相位不携带任何可预测信息。选锚点等于抽签。

### 5. 等风险基线 = 没有 edge

去掉全部档位倍率、每笔都用同一个 ATR 风险公式（`FlatRiskV23`）：

| | 2025 | 2026 |
|---|---:|---:|
| 收益 | `-19.29%` | `+50.79%` |
| PF | `0.96` | `1.10` |
| 最大回撤 | `47.70%` | `25.72%` |

底层突破信号本身的 PF 在 1.0 附近抖动。所有超额收益都来自"给恰好赚钱的两档加杠杆"。

### 6. 独立事件研究：多头信号方向是错的

不用回测框架，直接统计 `1h Donchian-20 突破 + rvol ≥ 1 + 4h EMA 同向` 之后的前向收益（`scripts/analyze_breakout_signal_event_study.py`）：

- **多头信号两年都显著为负**：2025 H24 `-1.434%`（t=-3.51）、H48 `-1.745%`（t=-3.36）；2026 H48 `-4.673%`（t=-3.65）、H72 `-6.915%`（t=-4.89）。胜率 36–45%。
  但相对**同趋势对照组**的超额在 ±1% 内且符号不稳定——亏的是 4h 多头状态下这个币池的 beta，不是信号本身。
- **空头信号只有 2026 H72 显著**（`+3.137%`，超额 `+1.299%`，block bootstrap p=0.0165），而 2025 同一格超额是 `-1.096%`，符号相反。
- **唯一跨年稳定的效应**是 H1/H4 的突破后动量 `+0.05% ~ +0.4%`，**小于 0.14% 的双边手续费**。

这解释了为什么 `scout_breakout_long` 是唯一两年都亏的档位。

### 7. 已排除的伪信号来源（这些不是问题）

- **同根 K 线移动止损**：回测里 `Trade.adjust_min_max_rates` 先用当根 high/low 更新极值再判断止损，理论上乐观。用上一根极值重算（`LaggedTrailV23`）后差异 < 3 个百分点，非收益来源。
- **合成补齐 K 线**：关掉 fill-up（`nofillup`）后 1 日锚点逐笔完全一致。
- **幸存者偏差**：`download_all_binance_usdt_perp_1m.py` 原本只枚举 `exchangeInfo` 的在架合约。已修复为并入 Binance 归档中的退市合约（币种全集 650 → 679）。修复后实测在本窗口**零影响**：只有 4 个退市合约有 2026 数据，峰值 24h 成交额 23M，而 top-20 门槛是 102–312M。
  代码缺陷是真的、影响是零——两件事都要说。

## 复现口径

```bash
python3 -m freqtrade backtesting \
  --config user_data/config_high_volume_mainwave_v18_2026_02_08.json \
  --config <锚点 config> \
  --datadir user_data/data/binance-monthly-incremental-top2-28anchors-2026-30m-windowed \
  --timerange 20260201-20260814 \
  --max-open-trades 100 --dry-run-wallet 500 --fee 0.0007 \
  --cache none --export trades --breakdown month \
  --backtest-directory <out> --pairs <该锚点币池>
```

要点：

- `--max-open-trades 100` 只是绕开下单前的档位查询限制，实际并发由策略里的 `portfolio_position_slots = 4` 硬限；审计确认最大并发确为 4。
- 币池是**逐期点位化**的，必须按期切 pairs，不能给全集。
- 数据目录是按选币窗口切过的（warmup 31 天 / carry 31 天），这是内存能跑下去的前提。
- 2025 年口径：纯月度池，无周度增量叠加，排除集为 `{"PAXG", "XAUT"}`。

## 已知代码缺陷与修复

| 文件 | 问题 | 状态 |
|---|---|---|
| `scripts/download_all_binance_usdt_perp_1m.py` | 只枚举在架合约，退市合约缺失（幸存者偏差） | 已修复：并入归档 S3 前缀列出的退市合约，扣除 `BLUEBIRD/DOTECO/FOOTBALL` 等非币标的 |
| `scripts/build_monthly_incremental_candidate_pools.py` | 叠加层只在"刷新日恰好也是周度快照日"时重置；锚点 1/8/15/22 恰好都满足，换成锚点 2 就 `AssertionError: incremental candidates overlap the active core` | 已修复：核心刷新时无条件重置。对原 4 锚点逐期零差异 |
| `freqtrade/exchange/binance_leverage_tiers.json` | 缺 GRVT/INX/KAT/MANTRA/XAUT/EOS，回测直接退出 2 | 已从 `https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets` 取真实档位补入（688 → 694） |

**无法修复的边界**：SXP 和 EOS 已退市且 Binance 不再提供历史 market metadata（最小名义、精度），`ValueError: Can't get market information for symbol SXP/USDT:USDT`。选币可以纳入退市合约，回测不能。这两个占 116 个池位中的 2 个（1.7%），必须在结论里披露。

## 资源纪律

本机约 1.6 GiB 内存。见仓库根的 `AGENTS.md`。实操要点：

- 回测**串行**跑，`concurrency = 1`。
- 启动前门控 `MemAvailable ≥ 512 MiB`；等待用轻量 bash 循环，不要用 python 父进程等（pandas 常驻约 100 MB，自己就把门槛顶穿了）。
- 环境变量 `MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072`，`nice -n 10`。
- 用 `setsid nohup ... & disown` 起长任务；scratchpad 会在会话重启时被清掉，作业清单要落到 `user_data/` 下。
- `pkill -f <pattern>` 会匹配到自己的 bash 命令行——按 PID 杀。

## 产物地图

全部在 `user_data/analysis/` 下（被 `.gitignore` 的 `user_data/*` 忽略，仅本地）：

| 目录 | 内容 |
|---|---|
| `_repro-riskbudget2pct-2026/` | 提交版基线复现（逐笔一致） |
| `_repro-variant-{scoutfree,sixslot,laggedtrail,nofillup,top50mcap,riskslots1,riskslots2,riskslots3}-2026/` | 各变体 |
| `_repro-28anchors-2026/` | 28 个锚点全跑 |
| `_repro-2025-monthly/` | 2025 全年月度 |
| `_refit2025/`、`_refit2026monthly/` | 9 点倍率网格，两年各一份 |
| `_flatrisk/` | 等风险基线，2025/2026 × flatRisk/multOff |
| `monthly-refresh-offset-28anchors-2026/`、`monthly-incremental-top2-28anchors-2026/`、`monthly-rotation-2025/` | 点位化选币产物 |

数据目录：`binance-2025/`（54M）、`binance-monthly-rotation-2025-30m-windowed/`（30M）、`binance-monthly-incremental-top2-28anchors-2026-30m-windowed/`（117M）。

变体策略类在 `user_data/strategies/ReviewRiskBudget2PctVariants.py`。

## 如果还要往下做

不要再调这条线的参数。有意义的方向只剩：

1. 换标的宇宙或换信号族，重新做第 6 项的事件研究**再**决定要不要写策略——统计检验几分钟，回测几小时。
2. 如果坚持用这个信号，把它当成**动量 1–4 小时**的东西做，并先解决 0.14% 手续费大于信号幅度这个问题。
3. 任何新结论回填本文件。
