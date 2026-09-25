# r5 — 50 轮网络实战思路迭代（2026-09-25）

协议：`.claude/skills/binance-minute-strategy-research/SKILL.md` §3/§4。
数据：binance_public panels 860 币 2025-01-01..2026-08-16（5m/15m/1h/4h/1d，含 funding/mark/OI/toptrader/taker）。
新鲜阶段：binance-hist 1h feathers（2022-11..2025-12，608 合约含下架）。
切分：TRAIN 2025-01..09 / VALID 2025-10..2026-02 / HOLDOUT 2026-03..08（**未读**）。
成本：7.5bp（BTC/ETH）/ 10bp 每边；事件型口径（2026-09-24 用户批准）备而未用。
引擎：`eng5.py`（次根开盘成交、SL 优先、一币一仓、日聚类 t、集中度）。

## 来源（网络搜索，curl + duckduckgo html，2026-09-25）

- https://pineify.app/resources/blog/master-funding-rate-strategies-for-consistent-crypto-trading-profits
- https://medium.com/@DolphinDB_Inc/profiting-from-perpetuals-implementing-a-funding-rate-arbitrage-strategy-with-backtesting-e8b9b8766ac1
- https://quantengines.com/blog/perpetual-futures-funding-rate
- https://pruviq.com/blog/funding-rate-arbitrage-practical-guide/
- https://backtrex.com/en/blog/crypto-trading-strategy-backtesting-guide
- https://formion.ai/blog/how-to-backtest-crypto-strategy-without-overfitting
- WebSearch/WebFetch 工具 403 不可用；DDG html 两次后限流。prior-results E/F 节已扫过 Robot Wealth /
  VertoxQuant / Quant Galore / SetupAlpha / ICT / Bookmap。本轮 = 资金费/微观结构/日历事件/上新结构方向，
  与 prior-results 100+ 族价格指标去重（价格指标族不再扫）。

## 结果总表（TRAIN 判定：净>0、day_t≥1.5、≥1 笔/天；幸存者再过 2023-24 新鲜阶段门）

| 轮 | 想法 | 结论 | 关键数字 |
|---|---|---|---|
| R1 | 24h 累计资金费 top-3 空头篮子（R24 的 E 修补） | 否 | 最好 t −0.00；netting 后无边际 |
| R2 | 负费枯竭做多（镜像，首次） | 否 | 全组 −4~−30bp，t ≤ 0.28 |
| R3 | 枯竭空 + OI 上升确认 | 否 | 最好 t −0.06 |
| R4 | 结算后 T+5h 入场持 20h（避跳价） | 否 | t ≤ −0.45 |
| R5/R48 | 结算间隔变更事件 | 否 | 面板内每币间隔几乎不变（3 个事件）→ 无法测 |
| R6 | 宽结算 K 反转 | 否 | t ≤ −0.22 |
| R7 | 资金费 7d 极值 + mark 溢价同向（U60） | 弱过 | 最好 t 1.51 → 由 R47 变体取代 |
| R9-R11,R13-R15 | close/mark 失衡反转/延续/放量/中盘（15m、5m） | 否 | 全组 −13~−21bp，t −2.9~−17（两方向皆负=无边际） |
| R12 | 失衡横截面日频 LS | 否 | mean +10bp 但 med −2.5bp，t 0.89 |
| R17-R19 | Deribit 周/月到期、CME 季度到期窗口 | 否 | 周 n78 t 0.82；月/CME n≤18 t ≤ 0.69 |
| R20/R22/R23 | CPI / NFP 事件窗动量反转 | 否 | 全组 t ≤ 0.05 |
| R21 | FOMC 后做多 BTC/ETH | 否（新鲜门） | TRAIN +82bp t 2.90 但 n=12；2023-24 16 事件 −50bp t −0.83 |
| R24 | 月初前 3 天做多 | 否 | t ≤ −0.38 |
| R25 | 美股收盘反转 | 否 | t ≤ 0 |
| R26 | 亚盘开盘动量 | 否 | t ≤ 0 |
| R27 | 周内季节性（周二全宇宙做多） | 否（新鲜门） | TRAIN +129bp t 1.96；2023-24 t 0.98 / 0.46；周内基线 7 天里 Tue/Sat/Sun 都>1 → 噪声 |
| R29 | 大户持仓比极值衰减 | 否 | t ≤ 0.40 |
| R30 | 持仓比-户数比发散 | 否 | t ≤ 0 |
| R31 | 量枯竭 + OI 上升突破 | 否 | t ≤ 0.29 |
| R32/R33 | taker 占比期限结构 / 极值反转 | 否 | t ≤ 0.33 |
| R34 | OI 速度 + 价格同向/反向 | 否 | t ≤ 0.74 |
| R35 | 已实现波动比扩张跟随 | 否 | t ≤ 0 |
| R36 | 平均单笔规模 z 跟随 | 否 | t ≤ 0 |
| R37 | 全市场 OI 变化择时山寨 | 否 | t ≤ 0.36 |
| R38 | OI 95 分位 + 价格 5 分位做多 | 否 | t ≤ 0.29 |
| R39 | 资金费极值 × 高 OI 衰减 | 否 | t ≤ 0.42 |
| R41 | 新上市 2/6/12h 动量做多 | 否 | t ≤ 0.63 |
| R42 | 新上市 6h 后做空持 48h | 否（新鲜门） | TRAIN +248bp t 1.77；2023H1 +175bp t 1.36 但 2024H2 −356bp（上市空头部反弹=轧空） |
| R43 | 上市 30-90 天动量 | 否 | t 1.36，coin_share 0.70 |
| R44 | 休眠币放量苏醒跟随 | 否 | t ≤ −0.92 |
| R45 | 下架前 14 天做空 | 薄 | n=6，不可判 |
| **R47** | **资金费 72h 累计 ≥0.12% + mark 溢价>0 做空（镜像做多），中盘 31-90，持 96h** | **否（新鲜门）** | TRAIN：n 4635、+132bp、PF 1.32、**day_t 3.09**、17 笔/天、coin 0.09、hold 48→96 单调（t 2.37→3.09）、72h 优于 168h。**2023-24 门：−71/−103bp（t −0.90/−0.76）与 −57/−156bp（t −0.06/0.44）全负** |
| R49 | 小时收益偏度横截面 LS | 否 | t ≤ −2.48 |
| R50 | R27 锚点稳健性（陷阱 14） | 否 | hour0=avg24h（同值），Wed −112bp → 周二效应是 2025 专属季节性 |
| R51-R56 | R47 阈值/相位网格、尾部宇宙、资金费消融、溢价消融、β=1 对冲分解、屏障出场 | 否（R47 已被门否决；细化仅记录稳健性） | 阈值 0.06-0.18% × 相位 t 2.07-2.88（平台非尖峰）；消融：只资金费 t 1.88、只溢价 t 0.24（两条件都必要）；β=1 对冲残差 +72.6bp（PF 1.18，非 beta，2025 个币特质）；屏障：tp15% t 2.46、加 sl 反而降（震荡扫损）。R52 首跑有 universe bug（build 写死 u_mid），重跑 u61-150：n 7556、+97bp、t 2.55——宇宙间稳健，但 R47 门否决不变（尾部宇宙属同一 2025 形态） |

## 结论

50 个想法轮全部完成。唯一强 TRAIN 幸存者 R47（拥挤资金费 + 溢价同向衰减）在两个独立历史阶段
（2023-01..06、2024-06..12，含下架合约）全部为负——2025 山寨拥挤多头平仓潮的专属形态，与
prior-results D 节"2025→2023-24 翻号"模式一致。**没有策略通过 §4；VALID/HOLDOUT 未读。**

## 陷阱新增

- binance_public panels 的 `funding_interval_hours` 每币在面板内几乎恒定 → 间隔变更事件在面板数据上不可研究（需原始 archive）。
- `binance-hist` futures feather 无 `quote_volume` 列（只有 base `volume`）；funding 费率在 `open` 列、`calc_time` 即 `date`。
- 日历事件（FOMC/CPI/NFP/到期）在 TRAIN 里 t 可达 2-3 但 n≤40，新鲜阶段必然翻号或频率必挂 B——不再作为主攻方向。

## 复现

```
cd user_data/minute_research/r5
python3 batchA.py   # R1-R7   funding 族
python3 batchB.py   # R9-R15  mark/last 失衡族
python3 batchC.py   # R17-R27 日历/宏观事件
python3 batchD.py   # R29-R39 流向/持仓
python3 batchE.py   # R41-R45 上新/结构
python3 batchF.py   # R47-R50 资金费+溢价 及检验
python3 batchG.py   # R51-R56 R47 细化/消融
python3 freshgate.py   # R21/R27/R42 新鲜阶段门
python3 r47_gate.py    # R47 新鲜阶段门（binance-hist）
```
