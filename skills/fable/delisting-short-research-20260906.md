# 合约退市公告做空：事件研究（2026-09-06）

承接 `strategy-candidates-20260906.md` 的 C1。机制：Binance 公告"将于 T 结算并下架 X 永续"后，
多头必须在 T 之前平仓，空头没有对手盘约束，价格在公告到结算之间持续下行。
事件研究、敏感性、安慰剂在前半部分；原生引擎实现、回测对账与 dry-run 运行方式在后半部分（§原生实现、§原生引擎回测结果）。**dry-run 尚未开始。**

## 研究前冻结的设计

- 事件：Binance CMS 退市目录（catalogId=161）中标题含 Futures/Delist/Perpetual 的公告，
  公告发布时间精确到秒，逐条解析"automatic settlement"段落里的合约与结算时间。
  只取 USDT 永续；COIN-M、公告延期通知不计。
- 入场：公告发布 + 延迟 d 分钟后第一根 1m K 线的开盘价做空，d ∈ {0, 1, 5, 15, 60}。
  d = 0 是不可达的参照，不是可交易臂。
- 出场：结算前 60 分钟强制平仓（公告规定最后一小时保险基金不介入、只允许 IOC 清算），
  或 1m 收盘价高于入场价 15% 后下一根开盘止损，先到者为准。
- 成本：单边 0.10%（手续费 + 摩擦假设，非实测盘口），另报 0.20%。
- 资金费：持仓期间每次实际结算按 `费率 × 当时 mark / 入场价` 计入空头收益（空头在正费率时收）。
- 收益口径：空头按入场名义本金的算术收益 `1 − 出场价 / 入场价`，上限 100%。
  第一版脚本误用 `入场/出场 − 1`，把 NEIROETH 一笔算成 +452%，已修正；这是 §6.3 陷阱的又一次实例。
- 数据：`binance_public/klines_1m` 的归档在退市后用零成交量行填充，事件只在该币非零成交量区间内有效。
- 对照：同窗口 BTC 做空；安慰剂 = 同币、同持仓长度、同止损，入场提前 7 / 14 / 21 天。
- 分期按公告日期：2025 = 训练，2026-01～05 = 验证，2026-06 起 = 保留。
- 稳健性：按公告聚类 bootstrap、剔除最好 3 笔、成本翻倍。

## 样本

2025-01 以来 25 条 USDT 永续退市公告、59 个合约事件。可用 53 个（23 条公告）：
PORT3 与 AIA 是公告后 30 分钟即结算的同日退市，入场晚于强制出场时点，排除；
AERGO 本地 K 线归档只到 2026-06-30，公告在 7 月，排除。
分期：训练 27、验证 25、保留 1（IP，亏损）。**保留段只有 1 笔，本轮没有时间外证据。**
公告到结算的间隔中位数约 91 小时，范围 45～144 小时。

## 主结果（延迟 5 分钟入场，15% 收盘止损，结算前 60 分钟平仓）

| | 全部 | 2025 训练 | 2026-01～05 验证 | 保留 |
|---|---:|---:|---:|---:|
| 笔数 | 53 | 27 | 25 | 1 |
| 单笔净均值 | **+14.2%** | +16.4% | +13.1% | −15.3% |
| 中位数 | +11.3% | +16.2% | +7.7% | |
| 胜率 | 62.3% | 66.7% | 60.0% | |

- 按公告聚类 bootstrap 95% 区间 [+6.9%, +22.0%]，P(均值 ≤ 0) < 0.0005。
- 剔除最好 3 笔：+10.6%。成本翻倍到 0.20%/边：+14.0%。资金费贡献均值 +0.9%。
- BTC 同窗口做空：+0.4%，市场方向解释不了。
- 18 笔触发 15% 止损，止损笔亏损集中在 −13%～−17%；35 笔持有到结算前 60 分钟。
- 固定金额账本：150 USDT/笔、53 笔共 +1130 USDT；23 条公告中 15 条为正；
  最大同时持仓 8 笔（同一公告常含 3～6 个合约）。

## 安慰剂：这些币本来就在跌吗

同币、同持仓长度、同止损、同成本，入场提前 7 / 14 / 21 天：

| 入场 | 单笔净均值 | 中位数 | 胜率 |
|---|---:|---:|---:|
| 公告后 5 分钟（真实） | +14.2% | +11.3% | 62.3% |
| 提前 7 天 | +3.0% | +2.5% | 64.2% |
| 提前 14 天 | +2.5% | +0.6% | 54.7% |
| 提前 21 天 | +1.5% | +1.4% | 63.5% |

配对差（真实 − 安慰剂）分别为 +11.2% / +11.7% / +13.3%，聚类 bootstrap 区间下限 +4.0% 以上，
P(差 ≤ 0) ≈ 0.001～0.002。这些币确有下行漂移（安慰剂为正），但公告本身贡献了绝大部分。

## 敏感性（全表报告，不选最优）

延迟 × 止损 × 出场的 80 个组合完整表在 `user_data/delist_short_20260906/sens.log`。结构性读数：

1. **延迟 5 与 15 分钟几乎无差别**（+14.2 对 +13.7）；延迟 60 分钟降到 +7.0%、中位数 −0.9%；
   延迟 4 小时 +6.3%，延迟 24 小时 +3.1%。**edge 的一半在公告后第一小时内**，
   但另一半是随后三四天的漂移：延迟 60 分钟、不止损、持有到结算前 60 分钟仍有 +16.9%。
2. **不止损的臂均值最高**（+20.5%，胜率 79%），但最大不利波动达入场价的 6 倍，
   1x 逐仓在那笔上会被清算；不止损不可实现。10% / 15% / 25% 止损结果相近（+13.5 / +14.2 / +13.8）。
3. **持有到结算前 60 分钟优于固定 24h / 48h**（+14.2 对 +10.6 / +13.1）。按距结算小时数看，
   平均累计收益从结算前 24h 的 14% 升到前 6h 的 18%、前 3h 的 21%；最后一段仍在跌。
   结算前 6 小时平仓比前 60 分钟少约 1.6 个百分点。
4. 止损臂的盘中最大不利波动 32%～44%（1m 最高价口径）：15% 收盘止损不封顶损失，
   跳空与逼空都可能放大。

## 与本仓库其他结论的关系

- 与上市首日做空（`listing-decay-research-20260905.md`）机制相反方向的同类：那条靠供应释放，
  这条靠强制平仓；两者都是"多数币下跌但均值靠少数大跌"的分布，这里剔除最好 3 笔仍 +10.6%。
- 资金费策略 §6.3 的算术/对数陷阱在这里以另一种形式重现：空头收益必须用 `1 − 出场/入场`。

## 尚未做、上线前必须做（写于原生复核之前；第 1、2、3 条已在下文完成）

1. **Freqtrade 原生引擎复核**：53 个合约的杠杆档位都在静态表里；但 3 个已从 ccxt 市场消失（见下文）。
2. **公告监听是实盘的前提**。Freqtrade 没有事件输入；需要一个外部轮询进程每 30～60 秒读
   CMS 接口，把 `(symbol, settle_time)` 写入本地 JSON，策略在 `populate_entry_trend` 里读它。
   延迟 15 分钟内 edge 完整，60 分钟只剩一半，轮询间隔按此定。
   还要先确认公告后交易所是否立即限制新开空单（公告写的是结算前 30 分钟禁止非只减仓单）。
3. 同一公告 3～6 个合约同时入场：`max_open_trades` 至少 8，或每条公告限 N 个。
4. 流动性：这些币 30 日日成交额中位数多在 50 万～400 万 USDT，150 USDT 市价单可以，
   0.1%/边摩擦对这类币偏乐观，实盘 dry-run 对账 markout 是硬门槛。
5. 保留段只有 1 笔；2026-06 之后的新公告是唯一的干净样本外，继续累积后再评。

## 复现

```bash
.venv/bin/python scripts/research_delist_short.py          # 主事件研究（含公告解析结果 events_parsed.csv）
.venv/bin/python scripts/research_delist_short_sens.py     # 80 组敏感性 + 路径 + 账本
.venv/bin/python scripts/research_delist_short_placebo.py  # 安慰剂
```

产物在 `user_data/delist_short_20260906/`：`announcements_raw.json`（公告原文）、
`events_parsed.csv`、`trades_all_delays.parquet`、`sensitivity_grid.parquet`、`placebo.csv`、
`cum_path_minutes.csv`、`exclusions.csv`、各 `*.log`。
公告接口：`https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&catalogId=161`，
正文用 `.../article/detail/query?articleCode=`，本机需经代理 `127.0.0.1:10811`。

## 原生实现（2026-09-06 下午）

三个件，公告接入、动态选币、策略：

| 件 | 文件 | 作用 |
|---|---|---|
| 公告轮询器 | `scripts/delist_announcement_poller.py` | 读 Binance CMS 退市目录，解析每条合约退市公告的 `(symbol, 发布时间, 结算时间)`，写 `user_data/delist/events.json`；同时把"现在处于 [发布, 结算−60m) 内"的 pair 写成 `user_data/delist/pairlist.json` |
| 动态 pairlist | `RemotePairList` + `file:///root/freqtrade/user_data/delist/pairlist.json` | 只有在退市窗口内的合约进白名单；`refresh_period: 60` |
| 策略 | `user_data/strategies/DelistShort1m.py` | 读 `events.json`（每次 `bot_loop_start` 按 mtime 重载）；公告 + 5 分钟起 15 分钟入场窗内做空；`custom_exit` 按 K 线开盘价止损 15%、结算前 60 分钟强平；`confirm_trade_entry` 保证每个事件只交易一次 |

回测用 `--from-research` 把 `events_parsed.csv` 转成同一份 `events.json`，datadir 由
`scripts/build_delist_datadir.py` 生成（1m 裁掉零成交量填充行，funding/mark 按 1h 网格），
配置 `config-delist-backtest.json`（58 个静态 pair，150 USDT/笔，最多 10 仓，fee 0.001）。

```bash
# 回测
.venv/bin/python scripts/delist_announcement_poller.py --from-research user_data/delist_short_20260906/events_parsed.csv
.venv/bin/python scripts/build_delist_datadir.py
.venv/bin/freqtrade backtesting -c config-delist-backtest.json --datadir user_data/data/delist \
  --strategy DelistShort1m --timerange 20250101-20260817 --cache none --export trades \
  --backtest-directory user_data/delist/backtest_results --breakdown month
.venv/bin/python scripts/report_delist_backtest.py      # 与事件研究逐笔对账

# dry-run：两个进程
.venv/bin/python scripts/delist_announcement_poller.py --loop --interval 30 --proxy http://127.0.0.1:10811
.venv/bin/freqtrade trade -c config-delist-dryrun.json --strategy DelistShort1m
```

dry-run 配置 `config-delist-dryrun.json`（副本 `skills/fable/delist-dryrun-config.example.json`）：
钱包 100 USDT、`stake_amount: unlimited`、`max_open_trades: 8`（同一公告常带 3～6 个合约），
`process_throttle_secs: 5`。轮询 30 秒 + 5 秒主循环，从公告发布到入场约 5～6 分钟，落在事件研究的
"延迟 5～15 分钟 edge 完整"区间内。

注意事项：
- 没有公告时 `pairlist.json` 的 `pairs` 为空，bot 空转，这是正常状态，不是故障。
- 轮询器失败时保留上一份文件（`keep_pairlist_on_failure`），策略侧不会因此误开仓；但轮询器
  长时间挂掉会错过事件，日志要监控 `ERROR` 行。
- 公告延期会再发一条通知，轮询器按同一 symbol 取最新一条的结算时间。
- 结算前 30 分钟交易所禁止非只减仓单，策略在结算前 60 分钟已经平仓；若 bot 在那之后才启动，
  `populate_entry_trend` 的入场窗早已过去，不会开仓。

## 原生引擎回测结果与对账（2026-09-06）

`config-delist-backtest.json`，55 个 pair，2025-01-01～2026-08-17，1m，逐仓 1x，150 USDT/笔、1000 USDT 钱包、最多 10 仓，fee 0.001/边，真实 1h 网格 funding/mark。
排除 3 个已从 ccxt 市场消失、无法取得精度与下单限制的合约（AERGO、BDXN、SXP，见 `user_data/delist/engine_exclusions.json`），
不伪造市场信息；其中 BDXN（研究 +33.5%）与 SXP（−19.5%）是两个可交易事件，合计影响 +14%。

| 指标 | 原生引擎 | 事件研究（同 51 笔） |
|---|---:|---:|
| 笔数 | 51 | 51 |
| 单笔净均值 | **+14.56%** | +14.50% |
| 净利润 | +1112.33 USDT（**+111.2%**） | |
| Profit factor | 3.58 | |
| 账户最大回撤 | 5.01% / 107.95 USDT | |
| 资金费合计 | +70.11 USDT（空头收） | |
| 退出 | settle_exit 34、close_stop 17 | force 34、close_stop 17 |
| 拒单 / 超时 | 0 / 0 | |

逐笔对账：51 笔入场时间全部与事件研究一致（偏差 0 分钟）；单笔收益差均值 +5.6bps，最大 137.5bps（ZRC）。
ZRC 的差异来自止损判定口径：引擎按当根开盘价 ≥ 入场 × 1.15 判定并按该开盘价成交，研究按上一根收盘价判定；
跳空开高时引擎早 9 分钟止损。其余 50 笔退出时间完全相同。引擎的 funding 按结算时 mark 价与实际数量入账，
与研究的"费率 × mark / 入场价"一致到 0.1% 以内。

月度：18 个月里 4 个月为负（2025-01 −26、2025-08 −23、2026-04 −20、2026-06 −23 USDT），
2025-11 与 2025-12 各 +280 USDT 以上。4 月的 12 笔 5 胜 7 负是唯一"多事件的负月"。
这是固定金额、无复利的账户；+111% 中约 55% 来自 2025-11～2026-01 三个月。

### 实现中踩到的三个坑（已修）

1. **`custom_exit` 只在 `use_exit_signal = True` 时被调用**（`freqtrade/strategy/interface.py:1460`）。
   第一版设为 False，止损与结算前平仓都没触发，仓位挂到数据末尾被 force_exit，锁死了 900 USDT，
   之后所有事件因资金不足都没开仓，报表却显示"拒单 0"。任何只靠 `custom_exit` 退出的策略都要检查这一条。
2. **策略在 `bot_loop_start` 重载事件文件**：回测跑到一半时我在同一台机器上测试了实时轮询器，
   它覆盖了 `events.json`，回测随即失去所有事件。现在回测/hyperopt 模式下只在 `bot_start` 读一次。
3. **信号根与成交根差一根**：freqtrade 在信号 K 线的下一根开盘成交。研究定义"公告 + 5 分钟后第一根开盘"，
   对应策略 `delay_min = 4`（信号根 = ceil(公告 + 4 分钟)，成交 = 下一根开盘 = ceil(公告 + 5 分钟)）。
   设成 5 会晚 1 分钟。

### 判定

原生引擎复现了事件研究的每一笔，机制在 Freqtrade 的撮合与记账下成立。上线前仍缺：
dry-run 对账（真实成交 markout，这些币日成交额 50 万～400 万 USDT，0.1%/边是假设）；
2026-06 之后的新公告作为唯一干净样本外。

## 回溯 2024：edge 不是全时期稳定的（2026-09-06 晚）

把公告目录翻到 2024-01：全年 9 条合约退市公告，7 条可解析，21 个 USDT 合约事件
（3 月 FOOTBALL/BLUEBIRD、ANT/DGB/CTK；5 月 STPT/SNT/MBL/RAD/CVX、IDEX/SLP/GLMR/MDT/AUDIO；
11～12 月 XEM/ORBS/LOOM、MAVIA/OMG/BOND）。OMG 后来两次延期，按首条公告的结算时间处理。
数据从 data.binance.vision 按月补齐（`scripts/fetch_delist_2024.py`：1m K 线、1m mark、funding，
写成 `user_data/delist2024/public` 与 freqtrade datadir `user_data/delist2024/data`），
事件研究用同一份冻结代码回放（`scripts/research_delist_short_2024.py`）。

延迟 5 分钟、15% 止损、结算前 60 分钟平仓，2024 全年：

| | 2024 | 2025-26 |
|---|---:|---:|
| 笔数 | 21 | 53 |
| 单笔净均值 | **+0.9%** | +14.2% |
| 中位数 | +3.7% | +11.3% |
| 胜率 | 57% | 62% |
| 剔除最好 3 笔 | −3.4% | +10.6% |
| 公告后 60 分钟价格反应 | −1.5% | +5.3% |
| 公告到结算中位 | 171h | 91h |
| 币的 30 日日成交额中位 | 500 万～1800 万 | 50 万～400 万 |

按季度看全部 74 笔：

| 季度 | n | 净均值 | 胜率 |
|---|---:|---:|---:|
| 2024Q1 | 5 | −12.3% | 0% |
| 2024Q2 | 10 | +4.5% | 90% |
| 2024Q4 | 6 | +6.1% | 50% |
| 2025Q1 | 1 | −17.6% | 0% |
| 2025Q3 | 8 | +14.8% | 50% |
| 2025Q4 | 18 | +19.0% | 78% |
| 2026Q1 | 13 | +26.1% | 77% |
| **2026Q2** | 13 | **−2.2%** | 38% |

- 强 edge 集中在 **2025Q3～2026Q1 的 39 笔**；2024 全年接近零，2026Q2 也回到零附近。
- 按季度做块 bootstrap（比按公告聚类更诚实）：53 笔区间 [+0.7%, +22.7%]，74 笔 [+0.1%, +18.1%]，
  P(均值 ≤ 0) 分别 0.008 和 0.023。按公告聚类的 [+6.9, +22.0] 低估了时间聚集。
- 2024 的公告形态不同：标题多为"Delist and Update the Leverage & Margin Tiers"，提前 7 天，
  对象是老牌低波动币，公告后第一小时价格几乎不动（−1.5%）。2025 下半年起对象变成上市不久、
  流动性差的新合约，提前 4 天，公告后立刻跳水。**edge 依附于交易所的退市对象选择规则，
  这个规则会变，2026Q2 已经出现了回落。**

判定修正：这是一条"当前有效但不保证持续"的事件策略，不是稳定的结构性 edge。
dry-run 的意义不只是对账成交，更是逐季检查 edge 是否还在；连续两个季度均值 ≤ 0 就停。

2024 的原生引擎复核（`config-delist-backtest-2024.json`，16 个仍在 ccxt 市场里的合约，
2024-02-01～2025-01-01，`DELIST_EVENTS_PATH=user_data/delist2024/events.json`）：
14 笔（AUDIO/MBL/ANT/FOOTBALL/BLUEBIRD 已从市场消失，不能装载），净 +21.21 USDT（**+2.1%**），
PF 1.16，最大回撤 10.1%，settle_exit 8、close_stop 6。14 笔与回放逐笔一致：入场偏差 0 分钟，
单笔收益差均值 1.2bps，最大 19.8bps；LOOM 的引擎止损晚 35 分钟（同样是开盘价对收盘价的判定口径）。
产物：`user_data/delist2024/`（`events.json`、`event_study.log`、`backtest.log`、`backtest_results/`）。

### 公告形态能否区分（事后切分，74 笔，不是预注册）

- **"Delist and Update the Leverage & Margin Tiers" 标题形态**：2024 年 15 笔（−1.1%），2025 年只有 1 月的 REEF（−17.6%），
  2026 年 0 笔。正文带杠杆档位表的还有 2025-10 的 SLERF（−18.6%）。带档位语言的 17 笔均值约 −3%，
  没有一笔落在强势期。这种形态自 2025-02 起基本消失，**用标题可以直接识别并排除**。
- **公告到结算的间隔**（公告里两个时间戳相减，完全可观测）：

| 间隔 | n | 净均值 | 胜率 |
|---|---:|---:|---:|
| ≤ 50h | 5 | +20.0% | 80% |
| 50～100h | 27 | +20.7% | 74% |
| 100～150h | 20 | +5.6% | 45% |
| > 150h | 22 | +0.1% | 55% |

  2024 全部 > 150h（提前 7～10 天）；2025Q3～2026Q1 多为 45～100h。但它**不能解释 2026Q2**：
  该季 3 笔 ≤ 100h 的也是 −3.5%，10 笔 > 100h 是 −1.8%。
- "periodically reviews" 模板句、同批合约数、币的成交额都没有更强的区分力。

结论：标题里的档位更新形态可以作为硬排除规则（有机制依据：那是对老币的风控调整，不是对新合约的清退）；
间隔 ≤ 100h 只作为观察指标，不作为过滤器写进策略，因为它是看过结果后才划的线，且对最近一季无效。

## 其他统计规律（2026-09-06 晚，全部事后探索，未预注册）

样本有限，所以顺着已有 74 + 53 个事件找了几条可测的规律。每条都标了能不能用。

1. **晚入场臂（不需要公告监听）**：合约退市事件，从结算前 H 小时做空到结算前 60 分钟，15% 止损：

   | H | n | 净均值 | 中位 | 胜率 |
   |---|---:|---:|---:|---:|
   | 72h | 45 | +2.4% | −0.3% | 47% |
   | **48h** | 47 | +7.8% | +5.3% | 64% |
   | 24h | 52 | +3.8% | +4.7% | 62% |
   | 12h | 52 | +4.9% | +3.1% | 58% |
   | 6h | 52 | +2.8% | +2.6% | 65% |
   | 3h | 52 | −1.4% | +0.7% | 58% |

   只需每日轮询即可实现，但 edge 只有公告后 5 分钟入场的一半。作为监听失败时的备用臂可以。
2. **止损之后币继续跌**：18 笔被 15% 止损的交易，从止损出场到结算前 60 分钟的空头收益均值 +17.3%、
   中位 +13.6%、胜率 72%。但止损后的进一步最大反弹中位 18.6%，所以再入场规则难写：
   价格跌回原入场价时再入场（13 笔）均值 −7.7%、9 笔再次止损；止损后等 24 小时再入、25% 止损（16 笔）
   均值 +11.2%、胜率 69%、仅 1 笔再止损。合并第一腿后每个止损事件仍是 −6.3%，
   即再入场只能挽回约三分之二的止损损失。样本 16～18 笔，不写进策略。
3. **公告前 24 小时已经大跌的事件没有 edge**（合约退市 2025-26，52 笔）：

   | 公告前 24h 收益 | n | 净均值 | 胜率 |
   |---|---:|---:|---:|
   | < −15% | 11 | **−9.0%** | 18% |
   | −15%～−5% | 16 | +24.4% | 81% |
   | −5%～0 | 10 | +16.5% | 80% |
   | 0～+5% | 6 | +31.6% | 83% |
   | > +5% | 9 | +14.0% | 56% |

   Spearman(公告前 24h 收益, 净收益) = +0.29，p = 0.04。机制说得通：币已经先崩过（市场整体下跌、
   或现货已先公告），被迫平仓的多头早就走了。2026Q2 的 13 笔公告前 24h 均值 −11.7%，
   是所有季度里最深的，这解释了一部分该季的回落。2024 样本无此关系（p = 0.55）。
   **可作为候选过滤器（公告前 24h 收益 > −15% 才入场）**，在新样本上验证后再启用。
4. **现货退市公告 → 做空永续**：独立事件源，53 笔，延迟 15 分钟、25% 止损 +14.6%，
   2026-06 起为负；详见 `spot-delist-perp-research-20260906.md`。不止损会遇到 ALPACA 式的 46 倍拉升。

## 生产版（2026-09-06 深夜）：完整代码与运行方式

| 件 | 文件 | 说明 |
|---|---|---|
| 公告轮询器 | `scripts/delist_announcement_poller.py --loop --interval 60` | 每 60 秒读 CMS 退市目录（最近 14 天），解析合约退市公告与现货退市公告（含其 Futures 段的合约结算时间），原子写 `user_data/delist/events.json` 与 `pairlist.json`（`updated` 时间戳作心跳） |
| 动态 pairlist | `RemotePairList`，`file:///.../pairlist.json`，`refresh_period: 60` | 只含处于 [公告, 结算 − 60 分钟) 窗口内、且形态为 plain 的 pair；空列表 = 无事件，正常 |
| 策略 | `user_data/strategies/DelistShort1m.py` | 合约事件 +5 分钟入场 / 15% 止损；现货事件 +15 分钟 / 25% 止损；结算前 60 分钟平仓，结算前 20 分钟应急平仓；每事件一笔；结算时间写入 trade custom data，事件文件丢失也能退出；轮询器 10 分钟没心跳则告警 |
| 引擎止损 | `stoploss = -0.50`，live 配置 `stoploss_on_exchange: true` | 灾难底线，样本内最大不利波动 32% 不会触发；交易所侧止损在 bot 掉线时仍保护仓位 |
| 启停 | `scripts/delist_start.sh [config]` / `scripts/delist_stop.sh` | 轮询器带自动重启循环，两个进程各有 pid 文件，日志在 `user_data/delist/logs/` |
| 演练 | `scripts/delist_pipeline_drill.py --pair DOGE/USDT:USDT --kind futures --settle-in-min 72` | 注入假事件，dry-run 十几分钟内走完"加 pair → 开空 → 结算前平仓"整条链；`--restore` 还原 |
| 配置 | `config-delist-dryrun.json`、`config-delist-live.json`（副本在 `skills/fable/delist-*-config.example.json`） | 100 USDT、`stake_amount: unlimited`、`max_open_trades: 8`、逐仓 1x、`process_throttle_secs: 5`；live 版只差 `dry_run: false` 与 API key |

```bash
scripts/delist_start.sh config-delist-dryrun.json      # 或 config-delist-live.json
tail -f user_data/delist/logs/freqtrade.log user_data/delist/logs/poller.log
scripts/delist_stop.sh
```

时延预算：公告发布 → 轮询器最多 60 秒 → pairlist 刷新最多 60 秒 → freqtrade 拉该 pair 的 1m K 线 →
信号根 ceil(公告 + 4 分钟) 收盘 → 下一根开盘市价开空 ≈ 公告后 5～6 分钟，落在事件研究"延迟 5～15 分钟 edge 完整"的区间。
现货事件同理，≈ 公告后 15～16 分钟。

live 与回测的三处已知差异：止损在 live 按 ticker 价（回测按 K 线开盘价）判定；成交摩擦按 0.1%/边假设，
这些币盘口薄，首批事件要手动核对成交价；freqtrade 每 pair 同时只允许一笔，同一币先现货后合约两次公告时只会交易一次。
