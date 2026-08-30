# 事件驱动策略(Freqtrade 实现)

结构性时点的收益分布可测,不依赖连续预测。每类事件独立成子策略、独立验证;
样本少是常态,**先做描述统计(事件后收益分布、最优持有窗),分布不显著就不写策略代码**。
回测纪律见主 SKILL.md §2。

## 1. 资金费率极值(最适合 Freqtrade 的一类)

前提:`trading_mode = "futures"`,`margin_mode = "isolated"`。futures 模式下
`download-data` 会一并拉取 funding rate,回测 PnL 自动计入资金费。

信号:funding 极值本身 + 结算时点前后的 flow 不对称。

```python
def populate_indicators(self, dataframe, metadata):
    df = dataframe
    # funding rate 列:futures 模式下经 dp 获取或离线合并(列名/获取方式以当前版本文档为准)
    df["fr"] = self._merge_funding(df, metadata["pair"])
    df["fr_z"] = (df["fr"] - df["fr"].rolling(30 * 3).mean()) / df["fr"].rolling(30 * 3).std()
    df["min_to_settle"] = df["date"].apply(self._minutes_to_next_settlement)  # 8h 周期,所依交易所而定
    return df

def populate_entry_trend(self, dataframe, metadata):
    df = dataframe
    window = df["min_to_settle"] < self.pre_window.value          # 结算前窗口,如 30 分钟
    # 极端负 funding:空头付费拥挤 → 结算前后常见挤压向上;正极值反向
    df.loc[window & (df["fr_z"] < -self.fr_k.value), ["enter_long", "enter_tag"]] = (1, "fr_squeeze_l")
    df.loc[window & (df["fr_z"] > self.fr_k.value), ["enter_short", "enter_tag"]] = (1, "fr_squeeze_s")
    return df
```

- 出场:持有窗覆盖结算时点后 N 分钟(`custom_exit` 超时),N 来自事件研究的最优窗。
- 冷门 pair 的 funding 极值常伴随控盘拉砸:与主 SKILL.md §1 黑名单联动,
  fr_z 极端但成交量/持仓集中度异常的 pair 不做。
- 注意区分方向逻辑:是"收 funding 的 carry"(持有跨结算)还是"挤压方向的价格移动"
  (结算前后短持有)。两者持有窗与风控完全不同,不要混在一个 enter_tag 里。

## 2. 跨所价差极值(降级版:单边信号交易)

Freqtrade 单实例单交易所,**做不了真正的双腿对冲套利**——那是 `niche-pair-trading`
(Nautilus)skill 的范围,用户要做对冲套利时明确指过去,不要在 Freqtrade 里 hack。

Freqtrade 能做的降级版:把"本所价格显著偏离参照所"当作均值回归信号,单边在本所交易。

- 参照价来源:producer/consumer(另一实例连参照所,消费其 close 列;链路延迟秒级,
  只适用于半衰期 ≥ 数分钟的价差)。回测用离线合并的参照所 K 线近似,
  实盘-回测数据路径不同 → dry-run 对账是硬门槛(主 SKILL.md §2.6)。
- 信号:`(mid_local - mid_ref) / mid_ref` 的 z-score 超阈值且半衰期窗口内;
  出场:价差回归 或 超时。**必须**先在筛选层确认该 pair 价差确有均值回归半衰期,
  单边交易承担的是净方向敞口,价差趋势性走扩时止损离场,没有对冲腿兜底。

## 3. 上新初期(独立实例,风险自担声明)

价格发现期波动与费前 edge 都大,但无历史数据、操纵密集,与主池逻辑冲突,
**必须独立 bot 实例 + 独立预算**运行:

- pairlist:`VolumePairList` 或交易所公告驱动的 RemotePairList + 反向 `AgeFilter` 思路
  (只留新上架;当前版本 AgeFilter 参数是否支持 max_days 以文档为准,不支持则外部筛选器出名单)。
- 策略形态:上新首日不做方向预测,只做过冲回归的收紧版(`overshoot-reversion.md`
  参数整体×1.5 的阈值,仓位减半)。
- 回测基本不可行(无先例数据),以 dry-run 收集 2–4 周事件样本后再决定是否放实钱。

## 4. 事件类策略的验证方式(替代常规回测指标)

- 按事件切片:`--breakdown day/week` + 按 `enter_tag` 分组统计;
  单事件类型样本 < 30 时只报告分布与置信区间,不报告年化。
- 收益集中度检查:剔除最好的 3 笔后是否仍过成本门槛;过不了 = 靠运气分布,否决。
- 事件定义漂移:交易所改结算周期/费率公式、上新规则变化,事件定义要有版本号,
  跨定义版本的样本不合并统计。

## 5. 陷阱清单

- funding/结算时间是交易所本地规则(多为 UTC 00/08/16,但有例外与动态周期),
  `_minutes_to_next_settlement` 按所硬编码前先查该所文档;错 8 小时整个信号反向。
- funding 数据与 K 线合并的时间对齐:funding 是离散点,前向填充时确保不引入未来值(前视)。
- producer/consumer 断连时 consumer 拿旧数据静默继续:对参照价加陈旧度检查,
  超过阈值(如 3 根 K 线)自动停开新仓。
- 事件窗内 `unfilledtimeout` 要短于窗口本身,否则挂单跨过事件时点成交,交易的已不是该事件。
