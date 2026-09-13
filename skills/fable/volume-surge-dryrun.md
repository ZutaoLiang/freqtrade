# VolumeSurgeTrend1m dry-run 运行手册

本手册用于仓库中的通用异常放量多空策略。它连接 Binance USD-M 正式市场的公共实时
行情，但由 Freqtrade 在本地模拟订单、余额和持仓；不要设置 Testnet，也不要把
`dry_run` 改为 `false`，除非用户另行明确授权实盘。

## 生产件与固定风险口径

| 文件 | 作用 |
|---|---|
| `user_data/strategies/VolumeSurgeTrend1m.py` | 1m 异常放量突破做多、极端冲高确认反转做空 |
| `config-volume-surge-dryrun.json` | Binance 永续 dry-run 配置 |
| `user_data/tradesv3.volume_surge_dryrun.sqlite` | 独立模拟交易数据库，首次启动后生成，不提交 |

配置固定为 100 USDT 初始钱包、每笔 25 USDT、最多同时 4 仓、逐仓 1 倍杠杆；静态
白名单为 LSK、CVC、STEEM、ARK。入场、退出和止损均用市价单，模拟单边费率为
0.05%。修改资金、仓位数、白名单或费用后，应视为新的 forward-test 口径并明确记录。

移动止损为 4×ATR，距离限制在 3%–18%。策略把持仓极值、绝对止损价、方向和最后处理
的已收盘 K 线写入 Trade 自定义数据 `atr_chandelier_v1`。重启必须继续使用同一个数据库，
不要在持仓期间删除、替换或与其他 bot 共用该数据库。

## 启动前检查

API Server 已启用并监听 `0.0.0.0:18081`。用户名、密码、JWT 密钥和 WebSocket token
不得提交到仓库，启动前通过环境变量注入：

```bash
export FREQTRADE__API_SERVER__USERNAME='<本地用户名>'
export FREQTRADE__API_SERVER__PASSWORD='<本地强密码>'
export FREQTRADE__API_SERVER__JWT_SECRET_KEY='<本地随机密钥>'
export FREQTRADE__API_SERVER__WS_TOKEN='<本地随机 token>'
```

端口绑定到全部网卡，但 Freqtrade API 本身不提供 HTTPS。只应通过防火墙、VPN 或 SSH
隧道访问，不要把 18081 直接暴露到公网。

随后在仓库根目录 `/root/freqtrade` 执行：

```bash
.venv/bin/freqtrade --version
.venv/bin/freqtrade show-config -c config-volume-surge-dryrun.json
.venv/bin/freqtrade test-pairlist -c config-volume-surge-dryrun.json
```

必须确认解析结果包含：`dry_run=true`、`dry_run_wallet=100`、`stake_amount=25`、
`max_open_trades=4`、`trading_mode=futures`、`margin_mode=isolated` 和策略
`VolumeSurgeTrend1m`。配置不含私钥；若只做 dry-run，不要加入交易权限 API key。

仓库配置不包含任何代理。某台机器需要代理时，在被 Git 忽略的本地覆盖文件中配置，
例如 `config-volume-surge-local.json`：

```json
{
  "exchange": {
    "ccxt_async_config": {
      "aiohttp_proxy": "<本机代理 URL>"
    }
  }
}
```

然后在下列检查和启动命令中额外添加
`-c config-volume-surge-local.json`；不需要代理的环境不要创建该文件。

## 前台启动与恢复

```bash
mkdir -p user_data/logs
.venv/bin/freqtrade trade \
  -c config-volume-surge-dryrun.json \
  --strategy VolumeSurgeTrend1m \
  --dry-run \
  --logfile user_data/logs/volume-surge-dryrun.log
```

命令行的 `--dry-run` 是配置之外的第二重保护。启动日志必须出现解析到
`VolumeSurgeTrend1m`、dry-run 已启用和四个白名单合约；任一项不符就停止排查。
API 健康检查地址为 `http://<运行主机>:18081/api/v1/ping`。

正常停止时发送一次 `Ctrl-C`，等待 Freqtrade 完成退出。恢复时执行同一条命令；同一
数据库会恢复模拟持仓，策略再从 Trade 自定义数据恢复并推进 ATR 移动止损。不要并发启动
两个使用同一数据库的进程。

## 观察与核对

```bash
tail -f user_data/logs/volume-surge-dryrun.log
.venv/bin/freqtrade show-trades \
  --db-url sqlite:///user_data/tradesv3.volume_surge_dryrun.sqlite
```

至少记录每笔信号时间、模拟成交价、盘口价差、方向、退出原因和实际持仓时长。重点比较
实时 dry-run 与回测的追涨滑点、快速假突破次数，以及 `trailing_stop_loss` 是否按上一根
已收盘 K 线单向推进。短期 PnL 不足以证明策略有效；该策略的历史收益集中于少数异常
事件，先积累独立事件样本再决定是否扩大币种或资金。
