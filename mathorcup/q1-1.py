import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler

# 假设数据已归一化
X = np.array([[0, 1, 1, 0.47],       # WAV
              [0.9, 0.7, 0.7, 0.65],  # MP3 128kbps
              [0.95, 0.85, 0.6, 0.78]]) # AAC 256kbps
y = np.array([3, 7, 9])  # 专家评分（归一化前的1-10分，需先归一化）

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

model = LinearRegression()
model.fit(X_scaled, y_scaled)

# 输出权重系数（对应 β1-β4，β0为截距）
print("权重：", model.coef_)
print("截距：", model.intercept_)
#回归模型