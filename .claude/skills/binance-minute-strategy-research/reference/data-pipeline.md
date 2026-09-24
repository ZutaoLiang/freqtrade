# Binance 数据下载与 freqtrade 格式转换

本文件是 SKILL.md §2 的细节。所有命令从 freqtrade 仓库根目录执行。

## 1. 两条获取路径

| 路径 | 什么时候用 | 限制 |
|---|---|---|
| **A. data.binance.vision 公开归档**（首选） | 任何主机；REST 被地区屏蔽时唯一可行 | 月度包在月末后几天才生成；资金费只有月度包 |
| B. `freqtrade download-data` | REST 可达、只要少量交易对的近期数据 | 走交易所 API（地区屏蔽即失败）；动态 pairlist 不解析；资金费 limit 过大可能 403（配置 `"_ft_has_params": {"funding_fee_candle_limit": 200}`） |

先测可达性：

```bash
curl -s -o /dev/null -w "vision %{http_code}\n" https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-01.zip.CHECKSUM
curl -s -o /dev/null -w "fapi %{http_code}\n" --max-time 10 https://fapi.binance.com/fapi/v1/time
```

## 2. Vision 归档结构（USDT-M 永续）

根：`https://data.binance.vision/data/futures/um/{monthly|daily}/<dataset>/<SYMBOL>/...`，每个 zip
都有同名 `.CHECKSUM`（SHA-256）。目录列表用 S3 接口：
`https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/monthly/klines/`
（v1 分页：`marker` / `NextMarker`）。

| dataset | 粒度 | CSV 列 | 用途 |
|---|---|---|---|
| `klines/<SYM>/<interval>/` | 1m…1d | open_time, open, high, low, close, volume, close_time, quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore | 主 K 线 |
| `fundingRate/<SYM>/`（仅 monthly） | 每次结算 | calc_time, funding_interval_hours, last_funding_rate | 资金费 |
| `markPriceKlines/<SYM>/1h/` | 1h | 同 klines | freqtrade 资金费计算用的标记价 |
| `premiumIndexKlines/<SYM>/<iv>/` | 1m… | 同 klines | 基差 / 溢价指数特征 |
| `indexPriceKlines/<SYM>/<iv>/` | 1m… | 同 klines | 现货指数价 |
| `metrics/<SYM>/`（仅 daily） | 5m | create_time, symbol, sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio | 持仓量、多空比特征 |
| `aggTrades`, `trades`, `bookTicker`, `bookDepth` | tick | — | 微观结构（体量大，1.6GB 主机慎用） |

freqtrade 只读 OHLCV；`taker_buy_volume`、`metrics`、`premiumIndex` 这类额外特征要单独存成
parquet，在策略里按 `date` 合并（先对齐时区，断言命中率，见 SKILL.md 陷阱 #9）。

## 3. 下载（scripts/download_vision.py）

```bash
K=.claude/skills/binance-minute-strategy-research/scripts
python3 $K/download_vision.py --list-symbols > /tmp/symbols.txt          # 全部 USDT-M 永续
python3 $K/download_vision.py --symbols BTCUSDT,ETHUSDT --start 2025-01 --end 2026-08
python3 $K/download_vision.py --symbols-file /tmp/symbols.txt --start 2025-01 --end 2026-08   # 全宇宙
```

- 串行、可续传：已存在且校验通过的文件直接跳过；校验失败的文件丢弃重下。
- 当前月 / 上月的月度包还没打出来时，K 线和标记价自动回落到日度包；资金费没有日度包，
  自动走 REST `/fapi/v1/fundingRate`（limit=200 分页），存到 `rest/fundingRate/`。
- 404 = 该时段没有数据（上线前 / 下架后），不是错误。
- 实测：2 个交易对 × 3 个月 × 三类数据，8 秒，峰值 38MB。全宇宙 650 对 × 20 个月 1m K 线约 5GB 压缩包，
  下载前先 `df -h` 看磁盘。

## 4. 转换（scripts/vision_to_freqtrade.py）

```bash
python3 $K/vision_to_freqtrade.py --raw user_data/data/binance-vision --datadir user_data/data/binance \
        --resample 5m,15m,1h [--symbols BTCUSDT,ETHUSDT] [--mark-fallback]
```

产出（freqtrade 2026.x feather 布局）：

```
user_data/data/binance/futures/BTC_USDT_USDT-1m-futures.feather      # 主 K 线
user_data/data/binance/futures/BTC_USDT_USDT-5m-futures.feather      # --resample 生成
user_data/data/binance/futures/BTC_USDT_USDT-1h-funding_rate.feather # 费率在 open 列，其余列 0
user_data/data/binance/futures/BTC_USDT_USDT-1h-mark.feather         # 1h 标记价 OHLC，volume 0
```

Schema：`date` = `timestamp[ms, UTC]`；OHLCV = float64；按 date 升序、无重复。
交易对命名：`<BASE>/USDT:USDT` ↔ 文件名 `<BASE>_USDT_USDT`（BASE 可以是 `1000PEPE`、中文名）。

转换器处理掉的格式问题（每一条都出过错）：

1. 新 CSV 有表头、旧 CSV 没有 —— 按首字节判断。
2. 部分归档的 open_time 是微秒 —— 大于 1e14 的除以 1000。
3. 月度与日度包重叠 —— 按 date 去重保留最后一条。
4. 资金费 `calc_time` 带毫秒抖动（如 `1759280400015`）—— 必须四舍五入到整点，否则 freqtrade
   按 `date` inner join 资金费与标记价时这一行被静默丢弃，资金费 = 0。
5. freqtrade 2026.x 的 `funding_fee_timeframe` 与 `mark_ohlcv_timeframe` 都是 **1h**
   （`freqtrade/exchange/exchange.py`）。存成 8h 的文件不会报错，只是永远不被读取。
6. 月度标记价包偶尔缺一天 —— 转换器报 `mark_missing_hours`；`--mark-fallback` 用最新价 1h
   K 线补齐并计数（标记价近似，影响的是那几小时的资金费金额）。

内存：一次一个交易对，20 个月 1m 峰值约 300MB。

## 5. 校验（scripts/validate_datadir.py）—— 回测前必跑

```bash
python3 $K/validate_datadir.py --datadir user_data/data/binance --csv /tmp/datadir_audit.csv
```

退出码 1 = 有 ERROR，不许回测。它检查的都是 freqtrade **不会报错**的问题：

- 资金费文件起点晚于 K 线起点 → 之前的交易资金费全为 0。
  （真实案例：本仓库 `binance/futures` 下很多交易对的 1h 资金费从 2026-01-01 开始，2025 年
  K 线上的回测资金费静默为 0。）
- 资金费行在标记价里找不到同一 `date` → inner join 丢弃。
- 资金费时间戳不在整点。
- 标记价缺失小时数；K 线缺口数与最大缺口。

回测后再复核一次：导出结果里持仓跨结算的交易 `funding_fees` 不应全为 0；若全为 0，必须说清是
"持仓从不跨结算"还是"数据没被读到"。

## 6. 回测时间窗与切块

- 1m 全宇宙回测内存巨大（94GB 机器上 204 对 × 561 天实测 46–49GB RSS）。1.6GB 主机只能：
  少量交易对；或按月/季度切块串行（`scripts/bt_runner.py`）；或事件型策略用"切片数据目录"
  （只保留事件窗口内的 K 线，见 SKILL.md §5.4）。
- `--timeframe-detail 1m`：信号在 5m/15m 上算、盘中路径用 1m 还原，比纯 1m 省内存。
- `--backtest-directory` 目录必须事先存在，否则导出被静默跳过。
- 配置文件里的 `datadir` 键不生效，必须用 `--datadir` 参数。
