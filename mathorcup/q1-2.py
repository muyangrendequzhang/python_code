import numpy as np

def ahp(matrix):
    # 计算矩阵的行数，即指标的数量
    n = matrix.shape[0]
    # 计算每列的和
    column_sum = np.sum(matrix, axis=0)
    # 归一化矩阵，将矩阵的每个元素除以对应列的和
    normalized_matrix = matrix / column_sum
    # 计算每个指标的权重，即归一化矩阵每行的平均值
    weights = np.mean(normalized_matrix, axis=1)
    # 计算最大特征值
    max_eigenvalue = np.max(np.linalg.eigvals(matrix).real)
    # 计算一致性指标 CI
    ci = (max_eigenvalue - n) / (n - 1)
    # 随机一致性指标 RI，这里给出了不同阶数的RI值
    ri_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    ri = ri_dict[n]
    # 计算一致性比率 CR
    cr = ci / ri
    return weights, max_eigenvalue, ci, cr

# 示例判断矩阵，这里假设你已经通过专家评估得到了这个矩阵
# 矩阵的行和列分别对应文件大小、音质损失、编解码复杂度、适用场景
# 例如，matrix[0][1] 表示文件大小相对于音质损失的重要性
matrix = np.array([
    [1, 3, 5, 2],
    [1/3, 1, 3, 1/2],
    [1/5, 1/3, 1, 1/4],
    [1/2, 2, 4, 1]
])

# 调用AHP函数进行计算
weights, max_eigenvalue, ci, cr = ahp(matrix)

# 输出结果
print("各指标的权重:", weights)
print("最大特征值:", max_eigenvalue)
print("一致性指标 CI:", ci)
print("一致性比率 CR:", cr)
if cr < 0.1:
    print("判断矩阵具有满意的一致性。")
else:
    print("判断矩阵一致性较差，请重新调整判断矩阵。")
#AHP
#层次分析法
    