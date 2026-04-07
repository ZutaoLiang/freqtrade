# freqtrade策略开发约定

## 创建策略
- 根据指定要求创建freqtrade策略
- 创建的策略文件放在user_data/strategies的子目录下，可根据策略特点放到适合的子目录下
- 策略文件命名格式为：strategy_name_V1.py，其中V1为策略版本号，如：strategy_demo_V1.py，后续相同的策略优化使用相同前缀，不断升级版本号

## 策略配置
- 策略配置文件放到user_data/config目录下，文件名为config-strategy.json，每个策略均对应独立的config-xxx.json文件

## 策略回测
- 策略写完后，均需要进行回测，回测在freqtrade的根目录下运行，调用根目录下的backtest.sh，参数1为timeframe，参数2为timerange，参数3为strategy名称；回测时间默认为2026年1月1日开始；
- 策略如果回测异常，则根据运行输出自动进行一次调整，重新回测；
- 策略运行正常后，根据输出结果进行策略总结，如果运行效果不理想，先提出优化思路，但不直接调整策略代码，等待人工确认后继续优化；
