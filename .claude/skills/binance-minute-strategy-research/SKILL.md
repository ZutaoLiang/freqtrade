---
name: binance-minute-strategy-research
description: 在任意服务器上从零做 Binance USDT-M 永续分钟级策略研究的完整流程——用 data.binance.vision 下载原始数据、转换成 freqtrade 可回测的 feather（含 1h 资金费与标记价）、校验数据、用任意技术指标构建策略、按预注册 + 三段切分 + 真实成本的纪律循环迭代，直到出现满足验收标准（含交易频率下限）的盈利策略或满 50 轮为止。Use when downloading Binance futures data for freqtrade, converting Binance Vision archives to freqtrade format, backtesting minute-level (1m/5m/15m) crypto strategies, or running an iterate-until-profitable strategy search; also for 分钟级策略、币安数据下载、数据转换、freqtrade 回测、策略迭代、技术指标策略.
---

# Binance 分钟级策略研究（数据 → 回测 → 迭代到盈利或 50 轮）

本 skill 让另一台服务器能完整复现这条研究线：下载 → 转换 → 校验 → 构建策略 → 严格回测 →
循环迭代。脚本在 `scripts/`，模板在 `templates/`，细节在 `reference/`。
**动手前先读 `reference/prior-results.md`**：已经被证伪的想法不要重做。

## 0. 目录

| 文件 | 作用 |
|---|---|
| `scripts/download_vision.py` | 从 data.binance.vision 串行、可续传、SHA-256 校验下载 klines / fundingRate / markPriceKlines |
| `scripts/vision_to_freqtrade.py` | 原始 zip → freqtrade feather（1m 及重采样周期、1h 资金费、1h 标记价） |
| `scripts/validate_datadir.py` | 找出 freqtrade 不会报错的静默数据问题（资金费覆盖、join、缺口、dtype） |
| `scripts/make_whitelist.py` | 只用 TRAIN 窗口按流动性排名选交易对，剔除并列出缺杠杆档位的交易对 |
| `scripts/harness.py` | numba 快速预筛：与 freqtrade 同样的成交约定，按日聚类统计 |
| `scripts/bt_runner.py` | freqtrade 串行分块回测 + 汇总（IS/OOS、月度、资金费检查、峰值内存） |
| `templates/TemplateIndicatorStrategy.py` | 策略模板（多周期、收盘价止损、时间出场、指标库回落） |
| `templates/backtest-config.example.json` | futures 回测配置模板 |
| `templates/iteration_log.md` | 迭代日志模板 |
| `reference/data-pipeline.md` | 数据来源、归档结构、格式陷阱、freqtrade 文件布局 |
| `reference/indicators.md` | 可用指标库与指标族、不前视写法、lookahead/recursive 分析 |
| `reference/prior-results.md` | 已测结论（20 轮分钟级扫描 + 本仓库更早的策略线） |

所有命令从 freqtrade 仓库根目录执行；`K=.claude/skills/binance-minute-strategy-research/scripts`。
全流程已在 freqtrade 2026.1 上端到端跑通（下载 → 转换 → 校验 → 回测，资金费非零）。

## 1. 环境与资源纪律（先做）

```bash
python3 -m freqtrade --version          # 以本机版本的文档为准，API 按月变
free -h; nproc; df -h .                  # 内存、核数、磁盘
python3 -c "import numba, pandas, pyarrow"
```

- 本机若有 `AGENTS.md` 的资源规则，优先遵守。默认：**重任务串行**（下载、转换、回测、hyperopt 一次一个）；
  先测单个任务的峰值 RSS 再决定能否并行；`MemAvailable` 至少留物理内存的 30%（1.6GB 主机 ≥512MB）；
  看到 swap 增长就停止派发新任务。
- 实测参考：freqtrade 自身基线约 400MB；1m 回测 ~50 对 × 3 天约 520MB，~235 对 × 3 天约 700MB；
  1m 全宇宙全时段在 94GB 机器上是 46–49GB——小机器只能切块或切片（§5.4）。
- 长时间回测用后台任务 + 内存守护（`MemAvailable` 低于阈值就杀进程）。守护循环里用 `pgrep -f`
  时，模式会匹配到守护脚本自己的命令行，永远等不到结束——按 PID 等。

## 2. 数据：下载 → 转换 → 校验

```bash
python3 $K/download_vision.py --list-symbols > /tmp/symbols.txt
python3 $K/download_vision.py --symbols-file /tmp/symbols.txt --start 2025-01 --end 2026-08   # 或 --symbols BTCUSDT,ETHUSDT
python3 $K/vision_to_freqtrade.py --raw user_data/data/binance-vision --datadir user_data/data/binance --resample 5m,15m,1h
python3 $K/validate_datadir.py --datadir user_data/data/binance --csv /tmp/audit.csv      # 退出码 1 = 不许回测
```

细节、归档结构、每条格式陷阱见 `reference/data-pipeline.md`。REST 可达且只要少量交易对时也可以用
`python3 -m freqtrade download-data --trading-mode futures --candle-types futures mark funding_rate -t 1m ...`，
但同样要跑 `validate_datadir.py`。

## 3. 切分、宇宙与成本（每个项目开始时定一次，写进迭代日志）

- **三段切分**（50 轮迭代会把单一 OOS 用成 IS）：TRAIN（~50%）用于构思与调参；VALID（~25%）每个
  幸存候选读一次，用于在候选之间取舍；HOLDOUT（最后 ~25%，至少 3 个月）**只在宣布最终策略时读一次**。
  例：2025-01~2025-09 / 2025-10~2026-02 / 2026-03~2026-08。
- 宇宙：`make_whitelist.py --rank-start <TRAIN 起> --rank-end <TRAIN 止>`，只用 TRAIN 窗口的流动性排名
  （用全样本排名会把"后来变大的币"选进来）。它会剔除并列出 freqtrade 缺杠杆档位的交易对——最新上线的
  币往往就在其中，这是必须写进报告的幸存者偏差。
- 成本（按边）：BTC/ETH 等主流 7.5bp（taker 5 + 滑点 2.5），其他 10bp；冷门币再加半价差。freqtrade 的
  `fee` 就填这个数。资金费一律真实计入（freqtrade 阶段）。
- 成交：信号用收盘 K 线，下一根开盘成交；市价单；限价/maker 假设在 1m OHLC 上无法可靠验证，
  不作为结论依据。

## 4. 验收标准（"可以回测盈利"的定义，必须全部满足）

频率是硬门槛——一年只触发几次的策略无论赚多少都不算（样本太少，不可信）。

| # | 条件 |
|---|---|
| A | freqtrade 引擎回测（不是 harness），成本按 §3，真实资金费；`validate_datadir.py` 无 ERROR；`lookahead-analysis` 无前视 |
| B | **频率**：VALID 与 HOLDOUT 各自 ≥ 300 笔、≥ 60 个有交易的日子、组合平均 ≥ 1 笔/天 |
| C | VALID 与 HOLDOUT 净利润都 > 0，PF ≥ 1.2；TRAIN PF ≥ 1.1（三段同号） |
| D | 按日聚类：VALID+HOLDOUT 的日收益 t ≥ 2.0 |
| E | 稳定：VALID+HOLDOUT 中 ≥ 60% 的月份为正；单一交易对贡献 ≤ 30% 利润；单一日子贡献 ≤ 20% 利润 |
| F | 成本压力：fee × 1.5 时 VALID+HOLDOUT 仍净盈利 |
| G | 平台：每个主要参数 ±20% 扰动，VALID PF 仍 ≥ 1.0（只做检验，不再据此选参） |

达不到 B 的候选可以记录为"低频事件策略"，但不能宣布为本流程的结果。

**事件型策略的可选口径（2026-09-24 用户批准，只能由用户选择启用）**：B 改为"组合平均 ≥ 0.4 笔/天、每段笔数 ≥ 0.4 × 天数"；
E 的单币/单日集中度上限放宽（事件策略天然集中在少数暴跌币）。其余 A、C、D、F、G 不变。启用时在迭代日志开头写明。

## 5. 策略构建

### 5.1 指标

可以使用**市面上任何能找到的技术指标**：TA-Lib、pandas_ta、freqtrade vendor qtpylib、technical、
TradingView 公开脚本、论文与博客里的自定义指标、freqtrade 社区策略。清单、库用法、不前视写法见
`reference/indicators.md`。多周期用 `merge_informative_pair`；外部特征（资金费、持仓量、溢价指数）
按 `date` 合并前对齐时区并断言命中率。

### 5.2 模板

复制 `templates/TemplateIndicatorStrategy.py` 到 `user_data/strategies/<名字>.py`，改指标与规则；
复制 `templates/backtest-config.example.json` 并填 whitelist。模板已内置：`use_exit_signal = True`
（否则 freqtrade 2026.x 不调用 `custom_exit`）、收盘价止损（1m 插针会扫掉最终盈利的单子）、时间出场、
杠杆 1。

### 5.3 先 harness 预筛，再上 freqtrade

```python
import sys; sys.path.insert(0, ".claude/skills/binance-minute-strategy-research/scripts")
import numpy as np, harness as H
d = H.load("user_data/data/binance", "BTC", "1m")
sig = np.where(<条件>)[0]
tr = H.run(d, sig, side=+1, hold=60, sl=0.01, cost_bps=7.5)
print(H.report({"BTC": tr}, oos_start="<VALID 起点>"))     # TRAIN 行才用于决策
```

harness 与 freqtrade 同样的成交约定（下一根开盘成交、同 bar 止损优先、算术收益），但**不计资金费**。
freqtrade 复核时每笔差异中位数应在几十 bp 以内；差很多先查实现，不要调参。

### 5.4 小内存主机上的 freqtrade 回测

- 连续型策略：少量交易对 + `bt_runner.py --chunk month|quarter` 串行分块（每块钱包重置、跨块持仓被
  强平，持有期要远短于块长）。
- 事件型策略：做"切片数据目录"——每个交易对只保留事件窗口前后若干天的 1m K 线（及对应 1h 资金费/
  标记价），只放可能触发信号的交易对，按事件日或月份串行跑。
- `--backtest-directory` 必须事先建好（`bt_runner.py` 已处理），否则导出被静默跳过。

## 6. 迭代循环（直到满足 §4，或满 50 轮）

每一轮 = 一个想法从预注册走到判定。按下面的顺序，每一步写进迭代日志（`templates/iteration_log.md`）：

1. **找想法**：联网搜索论文、量化博客、TradingView、freqtrade 社区策略，或用 `reference/indicators.md`
   的指标族组合。先对照 `reference/prior-results.md` 与本项目日志去重。记录来源 URL 和原文规则。
2. **预注册**：跑之前写下周期、宇宙、入场、出场、止损、持有上限、成本、参数网格（每轮 ≤ 24 组）。
3. **TRAIN 预筛**（harness）：门槛——净收益 > 0、按日聚类 t ≥ 1.5、平均 ≥ 1 笔/天；毛收益 < 1× 往返成本
   的直接否决。选参取平台中心，不取最大值。
4. **freqtrade 复核 TRAIN**：确认引擎结果与 harness 一致；跑 `lookahead-analysis`。
5. **VALID 读一次**：按 §4 的 B–F 判定。失败就否决，写原因。
6. **不许用 VALID 调参**。从 VALID 失败中得到的新想法可以作为新的一轮，但日志里标"受 VALID 污染"，
   它的 VALID 结果证据等级降低；最终只认 HOLDOUT。
7. **宣布候选**：第一个在 VALID 上满足 §4（除 HOLDOUT 项）的策略，跑 G 的平台检验，然后读 HOLDOUT
   一次。HOLDOUT 通过 → 结束，交付；不通过 → 记为失败的一轮，继续（HOLDOUT 已被读过，之后若再宣布
   候选，需要等新数据形成新的 HOLDOUT，或在报告里明确降级）。
8. **计数与停止**：轮次到 50 仍无通过者 → 停止，交付"未找到"报告。不要为了凑结果放宽 §4。

方向建议（来自已测结论）：主流币的纯价格指标在分钟级基本打不过 taker 成本。优先考虑能把单笔毛收益
做大或把换手做小的结构——更高周期定方向 + 分钟级择时、结构性资金流（资金费、持仓量、上新、清算）、
横截面排序降低换手、过滤掉低波动时段；同时守住频率下限。

在 Claude Code 里可以用 `/goal <目标>` 或 `/loop` 驱动多轮迭代；每轮结束更新日志与计数。

## 7. 交付物

- 迭代日志（总表 + 每轮细节）、所有脚本、候选策略文件与配置。
- 最终报告：通过的策略（或"50 轮未找到"），三段结果表、§4 每一项的数值、成本压力与平台检验、
  已知偏差（被剔除的交易对、数据缺口、资金费覆盖）、复现命令。
- 反例与否决原因同样是交付物，写进 `reference/prior-results.md` 供下一台服务器使用。

## 8. 陷阱清单（每条都在真实研究中产生过错误结论）

1. 资金费文件起点晚于 K 线 → 早期交易资金费静默为 0（`validate_datadir.py` 查）。
2. 资金费 `calc_time` 未取整到小时 → 与标记价 join 失败被丢弃。
3. freqtrade 2026.x 资金费/标记价只读 1h 文件；其他周期的文件静默不被读取。
4. freqtrade 2026.x 只有 `use_exit_signal = True` 时才调用 `custom_exit`（否则持仓到数据结束）。
5. 缺杠杆档位的交易对会让整个 futures 回测中止；它们多是最新上线的币 → 幸存者偏差。
6. 下架交易对需要 `StaticPairList` 的 `"allow_inactive": true`；完全不在 ccxt markets 里的会报错，要剔除。
7. `--backtest-directory` 不存在时导出被静默跳过；配置里的 `datadir` 键不生效。
8. 山寨币事件结果必须按日聚类；暴跌日一天能贡献几百个"独立"样本。
9. 外部序列按 `date` 映射前对齐时区，否则静默全 NaN。
10. 1m 插针扫掉引擎 `stoploss`，被扫的往往是最终盈利的单子 → 用收盘价止损。
11. 同一交易对同时只能持一仓：持有期长于事件间隔时，后续信号被静默吞掉，先量化吞掉的部分。
12. 做空收益用算术收益，对数收益会高估空头端。
13. 同一个 OOS 反复用于挑选就不再是 OOS；新变体只能用新数据验证。
14. 调仓/检查时点本身会制造"盈利"（例如只有 00:00 调仓为正）——检验所有偏移量取平均。
15. 用全样本流动性选宇宙是前视；只用 TRAIN 窗口。
16. 股票/商品挂钩永续（NVDA、MSTR、CL、PAXG、XAUT 等）不是加密资产，按策略机制决定是否剔除，
    并且在看结果之前决定。
17. **事件锚定策略的 1h 近似会把事件瞬间的跳价算成收益**：资金费结算后第一分钟平均跳 +86~+120bp，1h 口径（T 开盘入场）
    让资金费偏斜的 PF 从 ~1.3 虚增到 ~1.8。任何事件策略都必须用 1m 数据、按"信号后下一根"入场复核（见 prior-results C2）。
