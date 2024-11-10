import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import pwlf

# 假设您的数据为 x 和 y
# 这里以示例数据为例
x = np.linspace(0, 10, 500)
y = np.piecewise(
    x,
    [x < 3, (x >= 3) & (x < 7), x >= 7],
    [
        lambda x: 2 * x + 1,
        lambda x: -1 * x + 20,
        lambda x: 0.5 * x + 5
    ]
) + np.random.normal(0, 1, 500)

# 将 y 作为信号进行变化点检测
signal = y

# 使用 Pelt 算法进行变化点检测
model = "l2"
algo = rpt.Pelt(model=model).fit(signal)

# 尝试增大惩罚项 pen 的值
pen = 100  # 根据需要调整，增大 pen 值
result = algo.predict(pen=pen)

# 输出检测到的变化点位置
print("检测到的变化点位置：", result)
print("检测到的变化点数量：", len(result) - 1)

# 绘制变化点检测结果
rpt.display(signal, [], result)
plt.title(f'变化点检测结果（pen = {pen}）')
plt.show()

# 将变化点索引转换为 x 的断点位置
bkps = result[:-1]  # 去除最后的信号长度值
breaks = [min(x)] + [x[bkp] for bkp in bkps] + [max(x)]

# 使用 pwlf 进行分段线性拟合
model = pwlf.PiecewiseLinFit(x, y)
res = model.fit_with_breaks(breaks)

# 绘制拟合结果
x_hat = np.linspace(min(x), max(x), num=1000)
y_hat = model.predict(x_hat)

plt.figure(figsize=(12, 6))
plt.plot(x, y, 'o', markersize=2, label='原始数据')
plt.plot(x_hat, y_hat, '-', label='分段线性拟合')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'分段线性拟合结果（pen = {pen}）')
plt.legend()
plt.show()

