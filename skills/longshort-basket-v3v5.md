# Skill 02: LongShortV3/V5 多空组合策略 — 标的篮子与参数实验

> **英文名称**：LongShortV3/V5 ETH-Anchored Portfolio Fade Strategy — Basket & Parameter Study
> **策略类型**：趋势锚定反向组合（ETH 主锚 + 山寨反向 + 组合级止盈/追踪止盈）
> **时间周期**：5m（信号 EMA 288/1152 ≈ 24h/4d）
> **实验日期**：2026-08-29 ~ 2026-08-30
> **回测环境**：freqtrade 2026.1，binance perp，1m 数据重采样 5m，5x 杠杆，cross 保证金

---

## 1. 策略机制回顾

- **信号**：ETH 收盘价 vs EMA(288)/EMA(1152)，且 EMA 连续 N 根上行/下行（trend_length=6）。
- **方向**：ETH 主锚在上行 regime 做多；其余山寨全部**反向**（上行 regime 做空，下行 regime 做多）。
- **退出**：单笔无独立止盈（`portfolio_exit_only=true`）；组合级止盈 +10 USDT（V5 加时间衰减：持仓每 72h 止盈阈值减半衰减）、组合追踪止盈（激活 3 USDT，回撤 0.35~0.4）、单笔止损 -500%（5x，实际由爆仓约束）。
- **仓位**：总 200 USDT，多空各半；山寨腿等权或 1/NATR 动态加权。

策略盈利前提：山寨与 ETH 同涨同跌（β≈1），偏离后回归到组合止盈/追踪止盈出场。

---

## 2. 篮子筛选实验（LongShortV3）

### 2.1 统计筛选全部失败

| 方法 | 结论 |
| :--- | :--- |
| Engle-Granger 协整（864 币，1h） | **无真实 ETH/alt 协整**；两个窗口仅 2/185 平凡存活（平行会话 Johansen 独立验证一致） |
| 高相关 + 低特异波动 + β 稳定性（1h log return，IS=2025.01-10 / OOS=2025.11-2026.08） | 指标 OOS 持续，但**月度重选跑输静态篮子** |
| Regime-fade 代理打分 | 代理回测 −32.8%~−98.4% vs 原白名单 +16.7% —— 代理忽略手续费/资金费/组合出场，**只可做粗筛负过滤** |
| 月度 walk-forward 重选（14 个月，2 组独立实现） | 重选 +54.3% < 静态 R3b +161.8% < 原白名单 +229.4%（月度强平链接法会美化长持仓尾部风险） |

**教训**：统计筛选（协整/IC/代理）在这类组合退出机制下全部不可迁移；静态篮子 + 等权最稳。

### 2.2 R3b 与其参数尖峰

- R3b = ETH + LINK UNI SKY SOL DOGE（前期人工搜索产物，存在 OOS 污染）。
- V3 口径 2025（1-9 月）+131.97% / Sharpe 2.24 / DD 26.0%；2026（至 8/17）+41.48%。
- **参数敏感性扫描（14 个单因子变体 × 两窗口）**：
  - 2026：全部变体盈利（+21.4% ~ +46.2%）——机制在顺风 regime 里不挑参数；
  - 2025：仅 4/14 盈利，BASE 邻居直接 −21% ~ −28%，TP/回撤任何改动回撤恶化到 80-96%；
  - **结论：BASE 参数是尖峰不是平台，2025 年收益含相当参数运气成分。**

### 2.3 β 加权仓位被否决

β 加权（滚动 OLS β，clip 后）所有窗口均不优于等权。1/NATR 动态仓位过度集中于高波动/β 破裂山寨（SUI、SKY）。**等权是唯一幸存方案。**

---

## 3. LongShortV5 实验（V3 结论迁移验证）

V5 相对 V3 新增：止盈时间衰减（decay 72h）、组合级止损（配置 -300 ≈ 关闭）、追踪止盈 3/0.35、1/NATR 动态仓位默认开。

| 篮子 | 窗口 | V5 dynamic ON | V5 等权 (dyn OFF) | V3 参考 |
| :--- | :--- | :--- | :--- | :--- |
| E0 (ETH+LINK SUI UNI ADA SKY) | 2025.01-10.01 | +54.06% / 0.70 / DD 60.9% | **+60.50% / 1.03 / DD 35.3%** | — |
| E0 | 2026.01-08.17 | +35.09% / 0.97 / DD 29.3% | **+59.57% / 1.98 / DD 20.1%** | +38.29% |
| R3b | 2025.01-10.01 | +48.90% / 0.45 / DD 80.0% | — | +131.97% |
| R3b | 2026.01-08.17 | +22.38% / 0.53 / DD 21.5% | — | +41.48% |

### 3.1 核心结论

1. **R3b 优势不能迁移到 V5**（+131.97% → +48.90%）：R3b 的 V3 收益是"参数尖峰 × 组合机制"的乘积，机制一变即失效。决策：**放弃优化 R3b，聚焦 E0**。
2. **`enable_dynamic_stake=false`（等权）在 V5 下双窗口全面占优**：收益更高、Sharpe 接近翻倍（2026：0.97→1.98）、回撤全面下降。1/NATR 加权再次被数据否决。
3. **当前最优组合：V5 + E0 白名单 + 等权仓位 + main_pairs=ETH**；2026 年 +59.57% / Sharpe 1.98 / DD 20.1% 为全部实验最佳。
4. 回测全过程中 2025 年所有 V5/E0 组合无爆仓。

---

## 4. 落地配置

`user_data/config/config-LongShortV5-E0.json`（含本机 WSL 代理设置，**含 API 密钥，勿入库**）：

- `main_pairs`: `ETH/USDT:USDT`；白名单：ETH + LINK SUI UNI ADA SKY
- `enable_dynamic_stake: false`；total_stake 200；`trade_leverage: 5`；`base_stop_loss: 1`
- 组合退出：TP 10 / decay 72 / 追踪 3 + 0.35 / 组合止损 -300
- 信号：5m，EMA 288/1152，trend_length 6，`use_ha_candles: false`

启动：`freqtrade trade --strategy LongShortV5 --config user_data/config/config-LongShortV5-E0.json`

注意：V5 在 dry-run 会在工作目录写 `next_entry_time_LongShortV5.json`（组合退出冷却状态）；残留旧文件会错误延迟进场。

---

## 5. 数据与脚本索引

| 文件 | 内容 |
| :--- | :--- |
| `scripts/ls3_corr_idio_screen.py` | 864 币高相关+低特异波动筛选（IS/OOS） |
| `scripts/ls3_coint_screen.py` | Engle-Granger 协整筛选 |
| `scripts/ls3_walkforward_select.py` / `ls3_walkforward.py` / `ls3_wf_ethmain.py` | 月度 walk-forward（两套独立实现） |
| `scripts/ls3_betaw_test.py` | β 加权仓位验证 |
| `scripts/ls3_sweep_run.sh` / `ls3_sweep_rerun.sh` / `ls3_sweep_y25cut.sh` | 参数敏感性扫描 runner（4~8 并发） |
| `scripts/ls3_basket_exp.py` | 篮子组合批量回测 |
| `user_data/basket_exp/sweep_results.csv` | V3 参数扫描结果（y25 全年口径） |
| `user_data/basket_exp/sweep_y25cut.csv` | V3 参数扫描结果（2025.01-10.01 口径） |
| `user_data/basket_exp/corr_idio_results.csv` | 相关性/特异波动筛选全量结果 |
| `user_data/basket_exp/v5_*.json`, `v5_*.log` | V5 实验配置与全量回测日志 |

**回测口径备注**：freqtrade 回测结果 zip 内嵌完整 config 与 timerange，可按 `pair_whitelist`+参数+窗口唯一定位一次回测，解析 zip 比解析控制台日志可靠。
