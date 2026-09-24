# 技术指标：可以用市面上任何能找到的指标

原则：**指标来源不设限**——TA-Lib、pandas_ta、freqtrade-technical、TradingView 公开脚本、论文、
量化博客、GitHub 上的 freqtrade 社区策略（freqtrade/freqtrade-strategies）都可以拿来。限制只在
三件事上：不能前视、必须能在 freqtrade 里按收盘 K 线复现、每轮参数数量受 SKILL.md §6 的预算约束。

## 1. 库（先查本机装了什么）

```bash
for m in talib pandas_ta technical; do python3 -c "import $m; print('$m ok')" 2>/dev/null || echo "$m missing"; done
```

| 库 | 规模 | 用法 | 备注 |
|---|---|---|---|
| TA-Lib | ~160 个函数 | `import talib.abstract as ta; ta.RSI(df, timeperiod=14)` | 官方 freqtrade docker 镜像自带；宿主机常缺（需要 C 库） |
| pandas_ta | ~200 个指标 | `import pandas_ta as pta; pta.supertrend(df.high, df.low, df.close, 10, 3)` | 纯 Python，最容易装 |
| freqtrade vendor qtpylib | 常用工具 | `from freqtrade.vendor.qtpylib import indicators as qtpylib` → `crossed_above`, `bollinger_bands`, `vwap`, `rolling_vwap`, `typical_price` | 随 freqtrade 一起 |
| technical（freqtrade-technical） | 补充指标 | `from technical import indicators as ti` → ichimoku, vfi, madrid_sqz 等 | 需单独 pip |
| 手写 pandas / numpy / numba | 任意 | 论文里的自定义指标 | 最灵活；大计算用 numba |

模板 `templates/TemplateIndicatorStrategy.py` 演示了 TA-Lib → pandas_ta → 纯 pandas 的回落写法。

## 2. 指标族（选题清单，不是推荐）

| 族 | 例子 | 典型用法 |
|---|---|---|
| 趋势 | EMA/SMA/HMA/KAMA 交叉、SuperTrend、Donchian、ADX/DMI、Ichimoku、Parabolic SAR、线性回归斜率 | 突破跟随、方向过滤 |
| 动量 / 振荡 | RSI、Stoch/StochRSI、MACD、CCI、Williams %R、ROC、TSI、Connors RSI | 超买超卖回归、背离 |
| 波动率 | ATR、布林带 / %B / 带宽、Keltner、挤压（BB in KC）、历史波动率、Garman-Klass | 止损尺度、挤压突破、波动率 regime |
| 成交量 | VWAP（日内锚定 / 滚动）、OBV、MFI、CMF、VWMA、成交量 z-score、主动买入占比（klines 的 taker_buy_volume） | 放量突破、偏离 VWAP 回归 |
| 价格结构 | 开盘区间、前日高低、枢轴点、分形、K 线形态（TA-Lib CDL*） | ORB、区间交易 |
| 时间 | 小时/星期季节性、美股开盘、资金费结算时刻、周末 | 过滤器或事件锚点 |
| 衍生品特有 | 资金费率及其连续性、溢价指数（premiumIndexKlines）、持仓量与多空比（metrics） | 拥挤度、挤压 |
| 横截面 | 相对强弱排名、相对 BTC 的残差收益、板块动量 | 多空篮子（需自写组合层） |
| 多周期 | 用 `merge_informative_pair` 引入 5m/15m/1h/4h 指标 | 高周期定方向、低周期择时 |

## 3. 不前视的写法（强制）

- 只用 `shift(+n)`，永远不用 `shift(-n)`；rolling 不用 `center=True`。
- 不做全样本归一化（z-score、分位数、min-max）——用 rolling 或 expanding。
- 高周期数据只能通过 `merge_informative_pair(..., ffill=True)` 合并，它会把 15m 值推迟到该根
  15m 收盘后才可见；手工 merge 会前视一根高周期 K 线。
- 外部序列（资金费、metrics、自算的全市场指标）按 `date` 合并前：两边都是 tz-aware UTC，
  并断言命中率（否则 `Series.map` 静默全 NaN）。
- 每个进入 freqtrade 阶段的策略都跑：

```bash
python3 -m freqtrade lookahead-analysis -c <config> --strategy <S> --datadir <dd> --timerange <IS 片段>
python3 -m freqtrade recursive-analysis -c <config> --strategy <S> --datadir <dd> --timerange <IS 片段>
```

  前者报出有前视的信号列；后者检查 `startup_candle_count` 是否足够（指标值随起点变化即不足）。

## 4. 从网上找思路时的记录要求

每轮在迭代日志里写：来源 URL、原文规则（参数、周期、出场）、原文声称的业绩与成本假设、你做的
任何改动。原文没给参数的（例如"放量 3 倍"但没说窗口），把你的取值当作本轮的预注册参数。
