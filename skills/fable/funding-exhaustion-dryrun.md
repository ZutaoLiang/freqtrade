# FundingExhaustionShort5m dry-run 运行手册

资金费衰竭做空（r3 研究第 R24 轮）：某合约连续 3 次结算的资金费率都 ≥ 0.03%（多头拥挤、持续付费）时，
在第 3 次结算后的 5 分钟做空，持有 8 小时后市价平仓。连接 Binance USD-M 正式市场的公共行情，
由 Freqtrade 在本地模拟成交；不要把 `dry_run` 改成 `false`，除非用户另行明确授权实盘。

## 生产件与固定口径

| 文件 | 作用 |
|---|---|
| `user_data/strategies/FundingExhaustionShort5m.py` | 策略（5m，只做空，止损 −50%，市价单） |
| `config-funding-exhaustion-dryrun.json` | dry-run 配置 |
| `user_data/tradesv3.funding_exhaustion_dryrun.sqlite` | 模拟交易数据库，首次启动生成，不提交 |

口径：初始钱包 100 USDT，`stake_amount: "unlimited"` + `max_open_trades: 5`（每仓约 19.8 USDT，复利），
逐仓 1 倍杠杆，模拟单边费率 0.05%。白名单是 162 个合约（TRAIN 期 2025-01..09 日成交额中位数 ≥ 1000 万 USDT，
剔除股票/商品挂钩合约，`user_data/minute_research/r3/universe_rank.csv`）；已下架的由 StaticPairList 自动忽略
（2026-09-25 启动时有效 154 个）。修改资金、仓位数、阈值、持有期或白名单都视为新的前向测试口径并记录。

实盘/dry-run 下资金费来自策略内缓存：启动时逐个合约拉最近 5 次结算，此后每个整点后的前 6 分钟每分钟
调用一次 `/fapi/v1/fundingRate`（全市场一次请求）。日志中应能看到 `funding cache seeded for N pairs`，
以及每个整点后的 `funding refresh: N whitelisted settlements`。

## 回测依据（freqtrade 引擎，真实资金费，100 USDT 口径）

| 段 | 笔数 | 收益 | PF | 日聚类 t | 最大回撤 | 备注 |
|---|---|---|---|---|---|---|
| TRAIN 2025-01..09 | 244 | +60.8% | 1.56 | 2.49 | 16.0% | 止损触发 1 次 |
| VALID 2025-12..2026-02 | 99 | +46.6% | 2.40 | 2.30 | 4.9% | ARC 占利润 68% |
| HOLDOUT 2026-03..08 | 97 | +22.2% | 1.55 | 1.01 | 9.6% | ARC 占利润 52%，统计不显著 |

已知风险：利润集中在少数资金费长期偏高的币（ARC、SWARMS、BTCDOM 等）；HOLDOUT 盈利但不显著；VALID
区间（剔除 2025-10..11）是在已知这两个月不利之后定义的。−20% 止损在每一段都降低收益，不要收紧。
详细记录见 `user_data/minute_research/r3c/RESULTS_schemeC.md` 与 `.claude/skills/binance-minute-strategy-research/PROFITABLE_STRATEGIES.md`。

## 启动前检查

API 用户名、密码、JWT 密钥和 WebSocket token 不得提交，启动前通过环境变量注入：

```bash
export FREQTRADE__API_SERVER__USERNAME='<本地用户名>'
export FREQTRADE__API_SERVER__PASSWORD='<本地强密码>'
export FREQTRADE__API_SERVER__JWT_SECRET_KEY='<本地随机密钥>'
export FREQTRADE__API_SERVER__WS_TOKEN='<本地随机 token>'
```

API 监听 `0.0.0.0:18082`（18081 与 8080 已被其他 dry-run 占用），只通过防火墙、VPN 或 SSH 隧道访问。

仓库配置不含代理。需要代理的机器在被 Git 忽略的 `config-funding-exhaustion-local.json` 中配置
（本机 Binance REST 需要代理；ccxt 同步与异步都要配，不要同时写 httpProxy 和 httpsProxy）：

```json
{
  "exchange": {
    "ccxt_config": {"httpsProxy": "<本机代理 URL>", "wsProxy": "<本机代理 URL>"},
    "ccxt_async_config": {"httpsProxy": "<本机代理 URL>", "wsProxy": "<本机代理 URL>"}
  }
}
```

```bash
.venv/bin/freqtrade show-config -c config-funding-exhaustion-dryrun.json -c config-funding-exhaustion-local.json
```

确认：`dry_run=true`、`dry_run_wallet=100`、`stake_amount=unlimited`、`max_open_trades=5`、
`trading_mode=futures`、`margin_mode=isolated`、策略 `FundingExhaustionShort5m`。

## 启动与恢复

```bash
mkdir -p user_data/logs
.venv/bin/freqtrade trade \
  -c config-funding-exhaustion-dryrun.json -c config-funding-exhaustion-local.json \
  --strategy FundingExhaustionShort5m \
  --dry-run \
  --logfile user_data/logs/funding-exhaustion-dryrun.log
```

命令行 `--dry-run` 是第二重保护。启动日志须出现 `Whitelist with 1xx pairs`、`funding cache seeded`
和 `state='RUNNING'`。正常停止发一次 `Ctrl-C`；恢复执行同一条命令、使用同一数据库，缓存会在启动时重新拉取。
不要并发启动两个使用同一数据库的进程。

## 观察与核对

```bash
tail -f user_data/logs/funding-exhaustion-dryrun.log
.venv/bin/freqtrade show-trades --db-url sqlite:///user_data/tradesv3.funding_exhaustion_dryrun.sqlite
```

每笔核对：入场是否在结算后第一根 5m K 线收盘后（T+5 分钟）、触发时的三次资金费率、成交价与 T+5 开盘价的偏差、
8 小时持仓期内收到的资金费、退出原因。回测平均约 0.5–1.1 笔/天，且信号成簇出现；
至少积累一个季度的独立结算事件再评估，不要根据几周的盈亏调整参数。
