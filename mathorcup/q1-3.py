import numpy as np

def entropy_weight_method(data):
    """
    熵权法计算指标权重
    :param data: 输入的数据矩阵，每列代表一个指标，每行代表一个样本
    :return: 各指标的权重
    """
    # 数据标准化（归一化）
    # 找到每列的最小值和最大值
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    # 避免分母为 0 的情况
    diff = max_val - min_val
    diff[diff == 0] = 1e-8
    # 进行标准化
    standardized_data = (data - min_val) / diff

    # 计算第 j 项指标下第 i 个样本值占该指标的比重
    p = standardized_data / np.sum(standardized_data, axis=0)

    # 避免对数计算时出现 log(0) 的情况
    p[p == 0] = 1e-8
    # 计算第 j 项指标的熵值
    entropy = -np.sum(p * np.log(p), axis=0) / np.log(len(data))

    # 计算第 j 项指标的差异系数
    coefficient = 1 - entropy

    # 计算各指标的权重
    weights = coefficient / np.sum(coefficient)

    return weights

# 示例数据
# 假设这里有 5 个样本，4 个指标
data = np.array([
    [10, 20, 30, 40],
    [15, 25, 35, 45],
    [20, 30, 40, 50],
    [25, 35, 45, 55],
    [30, 40, 50, 60]
])

# 调用熵权法函数计算权重
weights = entropy_weight_method(data)

# 输出结果
print("各指标的权重:", weights)
#熵权法