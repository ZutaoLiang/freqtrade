# AllWeatherRegimeAdaptive dry-run / 实盘运行手册

全天候宏观自适应多策略系统（融合双挤压动量、持仓量断崖出清抄底、多结算周期资金费衰竭做空三大正交 Alpha）：
1. **牛市与震荡动量引擎（Dual Squeeze）**：4H 与 1H 双挤压蓄势，仅在 BTC 宏观多头（`BTC 1D > SMA50 & BTC 4H > EMA50`）突破肯特纳上轨时做多；在 BTC 宏观大熊市中**强制关停破位追空**，杜绝被暴力轧空扫止损。
2. **反脆弱暴跌缓冲器（OI Flushout 爆仓出清抄底）**：12 小时价格跌幅 $>8\%$、1H RSI $<30$ 出现首根放量阳线反转时做多，全周期常驻，吃杠杆踩踏出清后的报复性反弹。
3. **衍生品现金流做空（R24 Funding Exhaustion）**：连续 3 期结算资金费率 $\ge +0.03\%$（年化 $> +32.8\%$ APR）时做空，持仓 8 小时吃妖币多头失血与情绪瓦解红利。

连接 Binance USD-M 正式市场的公共行情，由 Freqtrade 在本地模拟成交；不要把 `dry_run` 改成 `false`，除非用户另行明确授权实盘。

---

## 一、生产件与固定口径

| 文件 | 作用 |
|---|---|
| `user_data/strategies/AllWeatherRegimeAdaptive.py` | 全天候策略（1h，多空双向，自适应风控） |
| `config-allweather-dryrun.json` | 生产 dry-run 配置（无代理，供服务器直连部署） |
| `user_data/tradesv3.allweather_regime_adaptive_dryrun.sqlite` | 模拟交易数据库，首次启动生成，不提交 |

- **资金口径**：初始钱包 **100 USDT**，`stake_amount: "unlimited"` + `max_open_trades: 10`（每仓约 9.9 USDT，全自动动态复利）。
- **杠杆与费用**：逐仓 1.5 倍杠杆（Isolated Futures），模拟单边手续费 0.06%（万六 Taker）。
- **交易宇宙**：`VolumePairList` 自动筛选 30 天成交额排名前 120 的主流活跃合约（过滤稳定币、杠杆代币与极低价代币）。
- **风控硬规则**：
  - Flushout 抄底多：固定 -6% 止损，12h 最大持仓，+10% 止盈。
  - Dual Squeeze 趋势：固定 -7% 止损，18h 最大持仓，+14% 止盈。
  - R24 资金费做空：固定 -50% 灾难性硬止损，8h 完整结算周期，+15% 止盈。
  - 策略级兜底止损：`stoploss = -0.50`。

---

## 二、回测实证依据（Freqtrade 引擎，真实资金费，100 USDT 复利口径）

| 分段 | 跨度 | 笔数 | 胜率 | 净利润 | PF | 夏普 (Sharpe) | 最大回撤 | 备注 |
|---|---|---|---|---|---|---|---|---|
| **TRAIN** (2025-01..10) | 273天 | 1,133 | 53.9% | **+138.7%** (+138.7 U) | 1.27 | 7.79 | 24.9% | 结构牛市，多头贡献 +94.3 U，空头 +44.4 U |
| **VALID-C** (2025-12..2026-02) | 90天 | 323 | 58.2% | **+65.0%** (+65.0 U) | **1.62** | **12.80** | **14.7%** | 大盘大跌 -36.5%，策略逆势创历史新高 |
| **HOLDOUT** (2026-03..08) | 184天 | 768 | 47.0% | **+13.1%** (+13.1 U) | 1.06 | 1.76 | 29.5% | 2026 单边熊市，多头 Flushout 贡献 +21.2 U |
| **全周期贯通** (2025-01..2026-08) | **603天** | **2,517** | **52.7%** | **+439.2%** (+439.2 U) | **1.21** | **5.27** | **29.8%** | **本金 100 U 增长至 539.2 U (5.39倍)**，多空收益比 1.02:1.00 |

---

## 三、启动前检查

API 用户名、密码、JWT 密钥和 WebSocket token 不得提交，启动前通过环境变量注入：

```bash
export FREQTRADE__API_SERVER__USERNAME='<本地用户名>'
export FREQTRADE__API_SERVER__PASSWORD='<本地强密码>'
export FREQTRADE__API_SERVER__JWT_SECRET_KEY='<本地随机密钥>'
export FREQTRADE__API_SERVER__WS_TOKEN='<本地随机 token>'
```

仓库配置不含机器专属代理。需要代理的本地开发机使用 git-ignored 的 `config-funding-exhaustion-local.json` 覆盖：

```bash
freqtrade show-config -c config-allweather-dryrun.json -c config-funding-exhaustion-local.json
```

确认参数：
- `dry_run=true`
- `dry_run_wallet=100`
- `stake_amount=unlimited`
- `max_open_trades=10`
- `trading_mode=futures`
- `margin_mode=isolated`
- `strategy=AllWeatherRegimeAdaptive`

---

## 四、启动与守护

```bash
mkdir -p user_data/logs
freqtrade trade \
  -c config-allweather-dryrun.json -c config-funding-exhaustion-local.json \
  --strategy AllWeatherRegimeAdaptive \
  --dry-run \
  --logfile user_data/logs/allweather-dryrun.log
```

启动日志须确认：
1. `Whitelist with 1xx pairs`
2. `Funding cache seeded for N pairs`
3. 状态变为 `state='RUNNING'`

---

## 五、观察与核对

```bash
tail -f user_data/logs/allweather-dryrun.log
freqtrade show-trades --db-url sqlite:///user_data/tradesv3.allweather_regime_adaptive_dryrun.sqlite
```

核对清单：
1. **多头入场标签**：
   - `dual_sq_long`：确认发生时 BTC 1D > SMA50 且 4H > EMA50。
   - `flushout_long`：确认 12h 跌幅 $>8\%$ 且 1H RSI $<30$ 出现首根放量阳线。
2. **空头入场标签**：
   - `exhaust_short`：确认连续 3 次结算资金费 $\ge 0.03\%$。
   - `dual_sq_short`：确认仅在中性震荡市进场，大熊市中严禁出现该标签。
3. **出场标签**：
   - 资金费空单：`time_exhaust_8h` 或 `tp_exhaust_15pct` 或 `sl_exhaust_50pct`。
   - 抄底多单：`time_flushout_12h` 或 `tp_flushout_10pct` 或 `sl_flushout_6pct`。
   - 挤压趋势：`time_dualsq_18h` 或 `tp_dualsq_14pct` 或 `sl_dualsq_7pct`。
