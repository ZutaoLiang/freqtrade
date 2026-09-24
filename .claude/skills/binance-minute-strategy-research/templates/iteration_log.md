# 分钟级策略迭代日志

- 数据：<datadir>，<起止日期>，validate_datadir.py 结果：<pairs / ERROR 数>
- 宇宙：<make_whitelist.py 参数与结果，含因缺杠杆档位被剔除的交易对>
- 三段切分：TRAIN <起止> / VALID <起止> / HOLDOUT <起止>（HOLDOUT 只在宣布最终候选时读一次）
- 成本：主流 <x>bp/边，其他 <y>bp/边；资金费：真实（freqtrade 阶段）
- 验收标准：SKILL.md §4（频率下限：VALID ≥ 1 笔/天且 ≥ 300 笔）

## 总表

| # | 想法（来源 URL） | 预注册规则摘要 | 试过的配置数 | TRAIN（harness） | VALID（freqtrade） | 判定 / 原因 |
|---|---|---|---|---|---|---|
| 01 | | | | n= mean= bp, 日聚类 t= , 笔/天= | n= PF= 利润= , 笔/天= | |

## 每轮细节（按轮追加）

### 第 NN 轮 — <名字>

- 来源：<URL，原文规则与原文业绩>
- 预注册（跑之前写下）：周期 / 宇宙 / 入场 / 出场 / 止损 / 持有上限 / 成本 / 参数网格（≤24 组）
- TRAIN 结果：<表>；选中配置：<为什么是平台中心而不是最大值>
- freqtrade 复核（TRAIN）：与 harness 的差异 <bp>
- VALID 结果：<一次读取>
- 结论：通过 / 否决（原因）/ 需要什么新证据
- 用掉的 VALID 读取次数累计：<k>
