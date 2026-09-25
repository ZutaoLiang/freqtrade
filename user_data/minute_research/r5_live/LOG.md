# r5_live — 实盘帖子/社区策略的实现与回测，方案 C 复核，PROFITABLE_STRATEGIES 复核（2026-09-25）

> **重建说明**：原文件 `user_data/minute_research/r5/LOG.md`（未跟踪、被忽略）在拉取 6d8f60daf 时被另一会话同名文件覆盖。
> 本文件按本会话逐条写入的记录重建。脚本在 `scripts/minute_research/r5/`，产物在 `user_data/minute_research/r5/`
> （nfi/、e0v1e/、kst_bt/、p7_*、recheck_schemeC_*.csv、l4_grid_*.csv 等未被覆盖）。r4 日志在 `user_data/minute_research/r4/LOG.md`（未冲突）。

## 实盘帖子来源（用户指示：先实现再回测，有潜力就深挖）
| # | 来源 | 结果 | 判定 |
|---|---|---|---|
| L2 | 做多 BTC / 做空等权山寨篮子（"巨鲸一篮子做空 21 个山寨"） | 2021-22 V0 −59%/年、回撤 −77%；V1（相对趋势过滤）−1~−4%；2023-24 +9~+29%（t ≤ 0.71）；2025 TRAIN V1 N60 +71% t1.21；2025 做空山寨付资金费约 −12% | 有结构、不显著 |
| L2b | 空头腿换新币（上线 30-365 天） | 不优于 L2 | 否 |
| L1 | NostalgiaForInfinity | 当前 X7 在 2025 PF 87（样本内拟合）；历史版本→其发布后数据：滚动 2023-07..2026-08 合计 +3.5%（年化 +1.2%，t 0.40），2024 两窗口 PF<1 | 否（≈打平） |
| L3 | leader_squeeze（龙头跟随，实盘中） | 作者声明不可回测 | 跳过 |
| L4 | 合约网格（中性/做多/做空） | 5 主流 2025 平均为负 | 否 |
| L5 | E0V1E（称币安带单） | 2025-03 版样本外约 −21% | 否 |
- NFI 数据修正：1d/4h 需长历史（startup 800 根日线），新建 r5fut / r5spot2（1h/4h/1d 向前延长至 2022-11）；兼容垫片：pandas_ta 布林带列名、enter_long 初始化类型。
- NFI 滚动样本外：2023H2 X3 +0.8%；2024H1 X4@2023-12 −0.4%；2024H2 X4@2024-06 −1.4%；2025H1 X5 +0.1%；2025-07..2026-08 X6 +4.9%。

## 方案 C 复核（TRAIN 2025-01..09 / 隔离 2025-10..11 / VALID 2025-12..2026-02 / HOLDOUT 2026-03..）
| 候选 | 旧 VALID | VALID-C | 判定 |
|---|---|---|---|
| r3-10 f10 BB 吞没 4h | +160bp t1.91 | +143bp PF1.48 t1.41 单日 35% | 否 |
| r3-34 KST 日线 | −494bp | +135bp t0.65 | 否 |
| r3-42 KST 日线 + BTC 低波动 sl1.5_tp3 | +945bp t2.05 | n101 +1740bp PF9.33 t3.11（全在 2026-01） | 事件口径形式通过 → HOLDOUT |
| r4 YoungListingFadeUp | +191bp t1.95 | +144bp t0.97 | 否 |
| NFI X6 | — | +4.0% t2.43（n32） | HOLDOUT −1.4% → 否 |
| L2 V1 N60 | — | +10.9% t0.91 | 否 |
| E0V1E | — | −28% | 否 |
| r3 35 族 TRAIN 最佳格（参考） | — | 11 族净正、0 族 t≥2 | — |

## KST 日线 + BTC 低波动：HOLDOUT（用户批准一次读取）与 freqtrade 引擎复核
- harness HOLDOUT：n308 +255bp PF1.48 t2.04，V+HO t3.41。
- freqtrade 引擎（`user_data/strategies/KSTLowVolDaily.py`，数据 user_data/data/r5kst，1h 明细）：TRAIN n482 **−46bp PF0.95**；VALID n100 +1783bp；HOLDOUT n306 +215bp PF1.39 t1.78。
- 原因：harness 的 BTC 日线自 2025-01 起，180 日中位数历史不足，2025-01/02 低波动期被漏判（真实 30/27 天 vs 0）。**否决**（C：TRAIN PF < 1.1）。

## PROFITABLE_STRATEGIES.md 七策略独立复核（U160、Vision 1h 原始 K 线含 taker_buy、真实资金费；p7_panel.py / p7_strategies.py）
| 策略 | 文档 VALID | 复核 VALID-C | 复核 TRAIN | 结论 |
|---|---|---|---|---|
| R291 | +103bp t1.44 | n61 +94bp PF1.83 t1.20 | n54 +106bp t1.53 | 复现，t<2 |
| R24 | +177~203bp t2.3-2.6 | n134 +223bp t3.30，ARC 占 83% | n1206 −12bp | VALID 单币；TRAIN 不复现 |
| R102 | +527bp t1.32 | +552bp t1.34，单日 71% | +42bp t0.22 | 复现，t<2 |
| R145 | +175bp t1.39 | +143bp t1.20 | +191bp t1.47 | 复现，t<2 |
| R362 / R105 | +110 / +107bp | n6 / n4 | n0 / n15 | 不可复现（taker 占比 >0.66 仅 0.8% 的小时） |
| R234 | +230bp t2.58 | 液 30 有 metrics：n0 | n25 +78bp | 数据不全 |
