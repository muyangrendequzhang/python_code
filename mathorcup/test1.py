import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv('音频格式分析结果.csv')

# 检查数据结构
print("数据结构检查:")
print(df.info())
print("\n前几行数据:")
print(df.head())

# 提取真正的格式信息
if '格式' in df.columns:
    # 检查格式列内容
    print("\n格式列内容示例:")
    print(df['格式'].head())
    
    # 从格式列中提取实际音频格式
    print("\n从文件名提取格式...")
    # 先尝试直接根据常见格式名称提取
    df['真实格式'] = df['格式'].str.extract(r'(WAV|MP3|AAC|FLAC|OGG)', flags=re.IGNORECASE)[0]
    
    # 如果某些行没有成功提取到格式，尝试从文件扩展名提取
    mask = df['真实格式'].isna()
    if mask.any():
        df.loc[mask, '真实格式'] = df.loc[mask, '格式'].str.extract(r'\.([^\.]+)$')[0]
    
    # 转换为大写以便统一比较
    df['真实格式'] = df['真实格式'].str.upper()
    
    # 检查提取结果
    print("\n提取的格式分布:")
    print(df['真实格式'].value_counts())
    
    # 使用新提取的格式列
    format_column = '真实格式'
else:
    print("\n警告: 找不到格式列，检查CSV结构")
    format_column = None

# 1. 数据预处理和归一化
# 选择关键特征
storage_features = ['文件大小(KB)', '比特率']
quality_features = ['波形相关度', '梅尔频谱RMSE', 'MFCC差异', '谱质心差异']
scenario_features = ['专业音频制作得分', '移动设备得分', '流媒体服务得分', 
                     '归档存储得分', '实时处理得分', '语音内容得分']

# 确保所有特征列都是数值类型
print("\n检查数据类型并转换...")
for features in [storage_features, quality_features, scenario_features]:
    for feature in features:
        if feature in df.columns:
            if not np.issubdtype(df[feature].dtype, np.number):
                print(f"转换 {feature} 为数值类型")
                df[feature] = pd.to_numeric(df[feature], errors='coerce')

# 创建归一化器
scaler = MinMaxScaler()

# 归一化存储特征 (值越小越好)
storage_normalized = pd.DataFrame(1 - scaler.fit_transform(df[storage_features]), 
                                 columns=storage_features, 
                                 index=df.index)

# 归一化质量特征 (波形相关度越大越好，其他值越小越好)
quality_df = df[quality_features].copy()
quality_df['波形相关度_inv'] = 1 - quality_df['波形相关度']  # 转换为越小越好
# 转换为DataFrame以保留列名
quality_normalized = pd.DataFrame(1 - scaler.fit_transform(quality_df), 
                                 columns=quality_df.columns, 
                                 index=quality_df.index)
quality_normalized['波形相关度'] = 1 - quality_normalized['波形相关度_inv']  # 转换回原始关系
quality_normalized.drop('波形相关度_inv', axis=1, inplace=True)

# 归一化场景适应性评分 (值越大越好)
scenario_normalized = pd.DataFrame(scaler.fit_transform(df[scenario_features]), 
                                  columns=scenario_features, 
                                  index=df.index)

# 2. 创建综合评价指标
# 根据不同场景定义权重
weights = {
    '存储效率权重': 0.3,
    '音质保真权重': 0.4,
    '适用场景权重': 0.3
}

# 计算子指标分数
df['存储效率得分'] = storage_normalized.mean(axis=1)
df['音质保真得分'] = (quality_normalized['波形相关度'] * 0.4 + 
                  (1 - quality_normalized['梅尔频谱RMSE']) * 0.3 + 
                  (1 - quality_normalized['MFCC差异']) * 0.2 + 
                  (1 - quality_normalized['谱质心差异']) * 0.1)

# 3. 使用回归分析建立综合评价模型
# 创建场景适应性得分
for scenario in scenario_features:
    # 为每个场景创建自定义权重的综合得分
    if '专业音频制作' in scenario:
        df[f'{scenario}_综合'] = df['音质保真得分'] * 0.7 + df['存储效率得分'] * 0.1 + df[scenario] * 0.2
    elif '移动设备' in scenario:
        df[f'{scenario}_综合'] = df['音质保真得分'] * 0.2 + df['存储效率得分'] * 0.6 + df[scenario] * 0.2
    elif '流媒体服务' in scenario:
        df[f'{scenario}_综合'] = df['音质保真得分'] * 0.4 + df['存储效率得分'] * 0.4 + df[scenario] * 0.2
    elif '归档存储' in scenario:
        df[f'{scenario}_综合'] = df['音质保真得分'] * 0.6 + df['存储效率得分'] * 0.3 + df[scenario] * 0.1
    elif '实时处理' in scenario:
        df[f'{scenario}_综合'] = df['音质保真得分'] * 0.2 + df['存储效率得分'] * 0.3 + df[scenario] * 0.5
    elif '语音内容' in scenario:
        df[f'{scenario}_综合'] = df['音质保真得分'] * 0.3 + df['存储效率得分'] * 0.4 + df[scenario] * 0.3

# 创建总体综合评价指标
df['音频格式平衡指数'] = (df['存储效率得分'] * weights['存储效率权重'] + 
                    df['音质保真得分'] * weights['音质保真权重'] + 
                    df[scenario_features].mean(axis=1) * weights['适用场景权重'])

# 4. 分析不同音频格式在不同场景的表现
# 按照提取的真实格式分组分析
if format_column:
    # 确保所有需要计算均值的列都是数值类型
    analysis_columns = ['音频格式平衡指数', '存储效率得分', '音质保真得分'] + [f'{s}_综合' for s in scenario_features]
    for col in analysis_columns:
        if col not in df.columns:
            print(f"警告: 列 {col} 不存在")
        elif not np.issubdtype(df[col].dtype, np.number):
            print(f"警告: 列 {col} 不是数值类型，尝试转换")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 使用真实格式列进行分组
    format_analysis = df.groupby(format_column)[analysis_columns].mean().round(3)
    
    # 输出结果
    print("\n音频格式比较分析：")
    print(format_analysis)
    
    # 为每个场景找出最优音频格式
    for scenario in scenario_features:
        scenario_col = f'{scenario}_综合'
        if scenario_col in df.columns:
            best_idx = df[scenario_col].idxmax()
            best_format = df.loc[best_idx, format_column]
            best_score = df.loc[best_idx, scenario_col]
            print(f"\n{scenario}最佳音频格式: {best_format}")
            print(f"平衡指数: {best_score:.3f}")
    
    # 5. 可视化分析
    try:
        plt.figure(figsize=(12, 8))
        sns.heatmap(format_analysis, annot=True, cmap="YlGnBu")
        plt.title("音频格式在不同场景下的表现热力图")
        plt.tight_layout()
        plt.savefig('audio_format_performance.png')
        
        # 检查是否有主要格式可供比较
        main_formats = ['WAV', 'MP3', 'AAC']
        available_formats = [fmt for fmt in main_formats if fmt in format_analysis.index]
        
        if len(available_formats) >= 2:
            # 雷达图比较主要音频格式
            formats_data = format_analysis.loc[available_formats]
            
            # 准备雷达图数据
            categories = analysis_columns
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            
            for fmt in available_formats:
                values = formats_data.loc[fmt].values.tolist()
                values += values[:1]  # 闭合
                ax.plot(angles, values, linewidth=2, label=fmt)
                ax.fill(angles, values, alpha=0.1)
            
            ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            ax.set_ylim(0, 1)
            ax.set_title("音频格式性能雷达图")
            ax.legend(loc='upper right')
            
            plt.savefig('audio_format_radar.png')
            
        plt.show()
        print("\n分析完成，结果已保存为图表。")
    except Exception as e:
        print(f"可视化过程中出错: {e}")
else:
    print("错误: 无法分析音频格式，因为没有有效的格式列")