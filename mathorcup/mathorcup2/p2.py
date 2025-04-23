import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import librosa
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import platform
import io
import sys
from pathlib import Path
import re
import soundfile as sf

# ===== 中文字体强化解决方案 =====
def setup_chinese_font():
    """设置可靠的中文字体支持"""
    # 直接设置字体列表，不尝试访问整个字体目录
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 
                                      'DengXian', 'FangSong', 'Arial Unicode MS', 
                                      'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 尝试手动查找单个字体文件
    chinese_font_found = False
    system = platform.system()
    
    # 检查常见中文字体
    if system == 'Windows':
        potential_fonts = [
            r'C:\Windows\Fonts\simhei.ttf',  # 黑体
            r'C:\Windows\Fonts\msyh.ttc',    # 微软雅黑
            r'C:\Windows\Fonts\simsun.ttc',  # 宋体
            r'C:\Windows\Fonts\simkai.ttf',  # 楷体
        ]
        
        for font_path in potential_fonts:
            try:
                if os.path.exists(font_path):
                    prop = FontProperties(fname=font_path)
                    plt.rcParams['font.family'] = 'sans-serif'
                    print(f"使用字体: {font_path}")
                    chinese_font_found = True
                    break
            except Exception as e:
                print(f"尝试加载字体 {font_path} 时出错: {e}")
                continue
    
    # 打印字体配置结果
    if chinese_font_found:
        print("中文字体配置成功")
    else:
        print("警告: 未能配置中文字体，使用系统默认字体")
        
    return chinese_font_found

# 运行字体配置
setup_chinese_font()

def extract_audio_params(filename):
    """从文件名提取音频参数"""
    # 提取音频类型(语音/音乐)、采样率、格式和比特率
    pattern = r'(语音|音乐)_(\d+)Hz_(WAV|MP3|AAC)_?(\d+)?kbps?'
    match = re.search(pattern, filename)
    if match:
        content_type = match.group(1)
        sample_rate = int(match.group(2))
        format_type = match.group(3)
        bitrate = int(match.group(4)) if match.group(4) else None
        
        return {
            "内容类型": content_type,
            "采样率": sample_rate,
            "格式": format_type,
            "比特率": bitrate
        }
    return None

def analyze_audio_file(audio_path):
    """分析音频文件的特性"""
    file_size = os.path.getsize(audio_path) / 1024  # KB
    
    # 使用librosa加载音频
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # 计算基本特征
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.sqrt(np.mean(y**2))
        
        # 频谱特征
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # 梅尔频谱特征
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCC特征
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        return {
            "文件大小(KB)": file_size,
            "时长(秒)": duration,
            "RMS": rms,
            "频谱质心": spec_centroid,
            "频谱带宽": spec_bandwidth,
            "MFCC均值": mfcc_mean,
            "原始信号": y,
            "采样率": sr
        }
    except Exception as e:
        print(f"分析文件 {audio_path} 时出错: {e}")
        return None

def compare_audio_quality(original, compressed):
    """比较原始音频和压缩后音频的质量"""
    # 确保两个信号长度相同
    min_len = min(len(original["原始信号"]), len(compressed["原始信号"]))
    orig_sig = original["原始信号"][:min_len]
    comp_sig = compressed["原始信号"][:min_len]
    
    # 重采样比较信号
    if original["采样率"] != compressed["采样率"]:
        comp_sig = librosa.resample(comp_sig, 
                                    orig_sr=compressed["采样率"], 
                                    target_sr=original["采样率"])
        min_len = min(len(orig_sig), len(comp_sig))
        orig_sig = orig_sig[:min_len]
        comp_sig = comp_sig[:min_len]
    
    # 1. 波形相关性
    correlation = np.corrcoef(orig_sig, comp_sig)[0, 1]
    
    # 2. 频谱距离
    orig_spec = np.abs(librosa.stft(orig_sig))
    comp_spec = np.abs(librosa.stft(comp_sig))
    min_spec_len = min(orig_spec.shape[1], comp_spec.shape[1])
    spec_dist = np.mean(np.abs(orig_spec[:, :min_spec_len] - comp_spec[:, :min_spec_len]))
    
    # 3. MFCC距离
    orig_mfcc = librosa.feature.mfcc(y=orig_sig, sr=original["采样率"], n_mfcc=13)
    comp_mfcc = librosa.feature.mfcc(y=comp_sig, sr=original["采样率"], n_mfcc=13)
    min_mfcc_len = min(orig_mfcc.shape[1], comp_mfcc.shape[1])
    mfcc_dist = np.mean(np.abs(orig_mfcc[:, :min_mfcc_len] - comp_mfcc[:, :min_mfcc_len]))
    
    # 信噪比计算
    noise = orig_sig - comp_sig
    snr = np.mean(orig_sig**2) / np.mean(noise**2) if np.mean(noise**2) > 0 else float('inf')
    snr_db = 10 * np.log10(snr) if snr < float('inf') else 100
    
    return {
        "波形相关度": correlation,
        "频谱距离": spec_dist,
        "MFCC距离": mfcc_dist,
        "信噪比(dB)": snr_db
    }

def calculate_quality_score(quality_metrics):
    """计算质量得分(0-100)"""
    # 波形相关度(越高越好, 范围0-1)
    corr_score = quality_metrics["波形相关度"] * 40
    
    # 频谱距离(越小越好)
    spec_score = max(0, 30 - quality_metrics["频谱距离"] * 10)
    
    # MFCC距离(越小越好)
    mfcc_score = max(0, 20 - quality_metrics["MFCC距离"] * 5)
    
    # 信噪比(越高越好)
    snr_score = min(10, quality_metrics["信噪比(dB)"] / 5)
    
    total_score = corr_score + spec_score + mfcc_score + snr_score
    return min(100, max(0, total_score))

def calculate_value_index(quality_score, file_size, content_type="音乐"):
    """计算性价比指标"""
    # 根据内容类型调整权重
    if content_type == "语音":
        quality_weight = 0.6  # 语音对质量要求相对较低
        size_weight = 0.4
    else:  # 音乐
        quality_weight = 0.7  # 音乐对质量要求高
        size_weight = 0.3
    
    # 文件大小得分(越小越好): 使用对数来调整大文件的惩罚
    size_score = 100 - 15 * np.log10(1 + file_size/100)
    
    # 计算加权平均
    value_index = (quality_score * quality_weight + size_score * size_weight)
    return value_index

def analyze_audio_folder(folder_path):
    """分析文件夹中的所有音频文件"""
    audio_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.aac')):
                audio_files.append(os.path.join(root, file))
    
    # 找出原始音频文件
    original_music = None
    original_speech = None
    for file in audio_files:
        filename = os.path.basename(file)
        if "原始音乐" in filename:
            original_music = file
        elif "原始语音" in filename:
            original_speech = file
    
    # 分析原始文件
    orig_music_analysis = None
    orig_speech_analysis = None
    if original_music:
        orig_music_analysis = analyze_audio_file(original_music)
    if original_speech:
        orig_speech_analysis = analyze_audio_file(original_speech)
    
    # 分析其他文件
    results = []
    for file in audio_files:
        filename = os.path.basename(file)
        if "原始音乐" in filename or "原始语音" in filename:
            continue  # 跳过原始文件
            
        # 提取参数
        params = extract_audio_params(filename)
        if not params:
            continue
            
        # 分析音频
        analysis = analyze_audio_file(file)
        if not analysis:
            continue
            
        # 根据内容类型选择比较的原始文件
        original = orig_music_analysis if params["内容类型"] == "音乐" else orig_speech_analysis
        if not original:
            continue
            
        # 比较质量
        quality_metrics = compare_audio_quality(original, analysis)
        quality_score = calculate_quality_score(quality_metrics)
        
        # 计算性价比
        value_index = calculate_value_index(quality_score, 
                                           analysis["文件大小(KB)"], 
                                           params["内容类型"])
        
        # 合并结果
        result = {
            "文件名": filename,
            **params,
            "文件大小(KB)": analysis["文件大小(KB)"],
            "时长(秒)": analysis["时长(秒)"],
            "质量得分": quality_score,
            "性价比指标": value_index,
            **quality_metrics
        }
        results.append(result)
    
    return pd.DataFrame(results)

def create_direct_plots(df):
    """使用直接绘图模式创建图表，避免复杂的seaborn"""
    os.makedirs('output_images', exist_ok=True)
    
    # 分离语音和音乐数据
    speech_df = df[df["内容类型"] == "语音"]
    music_df = df[df["内容类型"] == "音乐"]
    
    # 1. 创建柱状图代替箱线图
    plt.figure(figsize=(14, 6))
    
    # 语音格式比较
    speech_grouped = speech_df.groupby(["格式", "采样率"])["性价比指标"].mean().reset_index()
    speech_formats = speech_grouped["格式"].unique()
    speech_sample_rates = speech_grouped["采样率"].unique()
    
    # 绘制语音数据柱状图
    plt.subplot(1, 2, 1)
    bar_width = 0.2
    index = np.arange(len(speech_formats))
    
    for i, sr in enumerate(speech_sample_rates):
        values = [speech_grouped[(speech_grouped["格式"] == fmt) & 
                                (speech_grouped["采样率"] == sr)]["性价比指标"].values[0] 
                 if len(speech_grouped[(speech_grouped["格式"] == fmt) & 
                                     (speech_grouped["采样率"] == sr)]) > 0 else 0 
                 for fmt in speech_formats]
        plt.bar(index + i*bar_width, values, bar_width, label=f'{sr}Hz')
    
    plt.xlabel('音频格式')
    plt.ylabel('性价比指标')
    plt.title('语音文件不同参数的性价比分析')
    plt.xticks(index + bar_width, speech_formats)
    plt.legend()
    
    # 音乐格式比较
    plt.subplot(1, 2, 2)
    music_grouped = music_df.groupby(["格式", "采样率"])["性价比指标"].mean().reset_index()
    music_formats = music_grouped["格式"].unique()
    music_sample_rates = music_grouped["采样率"].unique()
    
    index = np.arange(len(music_formats))
    
    for i, sr in enumerate(music_sample_rates):
        values = [music_grouped[(music_grouped["格式"] == fmt) & 
                               (music_grouped["采样率"] == sr)]["性价比指标"].values[0] 
                 if len(music_grouped[(music_grouped["格式"] == fmt) & 
                                    (music_grouped["采样率"] == sr)]) > 0 else 0 
                 for fmt in music_formats]
        plt.bar(index + i*bar_width, values, bar_width, label=f'{sr}Hz')
    
    plt.xlabel('音频格式')
    plt.ylabel('性价比指标')
    plt.title('音乐文件不同参数的性价比分析')
    plt.xticks(index + bar_width, music_formats)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output_images/音频参数性价比分析.png', dpi=300)
    plt.close()
    
    # 2. 质量vs文件大小散点图
    plt.figure(figsize=(12, 6))
    
    # 绘制语音散点图
    plt.subplot(1, 2, 1)
    formats = speech_df['格式'].unique()
    markers = ['o', 's', '^', 'D', '*']
    
    for i, fmt in enumerate(formats):
        subset = speech_df[speech_df['格式'] == fmt]
        plt.scatter(subset['文件大小(KB)'], subset['质量得分'], 
                   label=fmt, marker=markers[i % len(markers)], s=80)
    
    plt.title('语音文件：质量与大小关系')
    plt.xlabel('文件大小(KB)')
    plt.ylabel('质量得分')
    plt.legend()
    
    # 绘制音乐散点图
    plt.subplot(1, 2, 2)
    formats = music_df['格式'].unique()
    
    for i, fmt in enumerate(formats):
        subset = music_df[music_df['格式'] == fmt]
        plt.scatter(subset['文件大小(KB)'], subset['质量得分'], 
                   label=fmt, marker=markers[i % len(markers)], s=80)
    
    plt.title('音乐文件：质量与大小关系')
    plt.xlabel('文件大小(KB)')
    plt.ylabel('质量得分')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output_images/音频质量大小关系.png', dpi=300)
    plt.close()
    
    # 3. 使用matplotlib表格代替热图
    # 语音最佳参数表
    plt.figure(figsize=(10, 6))
    best_speech = speech_df.sort_values('性价比指标', ascending=False).head(3)
    
    plt.axis('off')  # 关闭坐标轴
    
    col_labels = ['格式', '采样率', '比特率', '质量得分', '文件大小(KB)', '性价比指标']
    table_data = [
        [row['格式'], f"{row['采样率']}", f"{row['比特率']}", 
         f"{row['质量得分']:.1f}", f"{row['文件大小(KB)']:.1f}", 
         f"{row['性价比指标']:.1f}"] 
        for _, row in best_speech.iterrows()
    ]
    
    table = plt.table(
        cellText=table_data, 
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.12, 0.12, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    plt.title('语音内容最佳参数推荐')
    
    plt.savefig('output_images/语音最佳参数.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 音乐最佳参数表
    plt.figure(figsize=(10, 6))
    best_music = music_df.sort_values('性价比指标', ascending=False).head(3)
    
    plt.axis('off')
    
    table_data = [
        [row['格式'], f"{row['采样率']}", f"{row['比特率']}", 
         f"{row['质量得分']:.1f}", f"{row['文件大小(KB)']:.1f}", 
         f"{row['性价比指标']:.1f}"] 
        for _, row in best_music.iterrows()
    ]
    
    table = plt.table(
        cellText=table_data, 
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.12, 0.12, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    plt.title('音乐内容最佳参数推荐')
    
    plt.savefig('output_images/音乐最佳参数.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("所有图表已保存到 'output_images' 文件夹")
    return True

def generate_report(df, best_speech, best_music):
    """生成分析报告，处理NaN值"""
    report = """
# 音频参数分析与最佳参数推荐报告

## 1. 参数影响分析

### 1.1 采样率的影响
{}

### 1.2 格式(压缩算法)的影响
{}

### 1.3 比特率的影响
{}

## 2. 最佳参数推荐

### 2.1 语音内容最佳参数
{}

### 2.2 音乐内容最佳参数
{}

## 3. 结论与建议
{}
"""
    
    # 分析采样率影响（处理NaN）
    sr_analysis = df.groupby(["内容类型", "采样率"]).agg({
        "质量得分": ["mean", "std"],
        "文件大小(KB)": ["mean", "std"],
        "性价比指标": ["mean", "std"]
    }).round(2).fillna("N/A")  # 替换NaN为N/A
    
    # 分析格式影响（处理NaN）
    fmt_analysis = df.groupby(["内容类型", "格式"]).agg({
        "质量得分": ["mean", "std"],
        "文件大小(KB)": ["mean", "std"],
        "性价比指标": ["mean", "std"]
    }).round(2).fillna("N/A")
    
    # 分析比特率影响（处理NaN）
    br_analysis = df.groupby(["内容类型", "比特率"]).agg({
        "质量得分": ["mean", "std"],
        "文件大小(KB)": ["mean", "std"],
        "性价比指标": ["mean", "std"]
    }).round(2).fillna("N/A")
    
    # 语音最佳参数
    speech_params = best_speech[["格式", "采样率", "比特率", "质量得分", "文件大小(KB)", "性价比指标"]]
    
    # 音乐最佳参数
    music_params = best_music[["格式", "采样率", "比特率", "质量得分", "文件大小(KB)", "性价比指标"]]
    
    # 结论
    conclusion = """
根据分析，我们得出以下结论：

1. **语音内容推荐参数**:
   - 采样率: {}Hz
   - 格式: {}
   - 比特率: {}kbps
   
   这种组合在保持较好语音清晰度的同时，大幅降低了文件大小，适合语音通话、播客等应用场景。

2. **音乐内容推荐参数**:
   - 采样率: {}Hz
   - 格式: {}
   - 比特率: {}kbps
   
   这种组合在保留音乐丰富细节的同时，提供了合理的文件大小，适合音乐流媒体、音乐欣赏等应用场景。

3. **通用建议**:
   - 语音内容可以使用较低的采样率和比特率
   - 音乐内容应当使用较高的采样率和比特率
   - AAC格式在大多数情况下提供了较好的性价比
    """.format(
        best_speech.iloc[0]["采样率"], best_speech.iloc[0]["格式"], best_speech.iloc[0]["比特率"],
        best_music.iloc[0]["采样率"], best_music.iloc[0]["格式"], best_music.iloc[0]["比特率"]
    )
    
    # 格式化报告，使用简单文本表示
    filled_report = report.format(
        sr_analysis.to_string(),
        fmt_analysis.to_string(),
        br_analysis.to_string(),
        speech_params.to_string(index=False),
        music_params.to_string(index=False),
        conclusion
    )
    
    # 保存报告
    with open("音频参数分析报告.md", "w", encoding="utf-8") as f:
        f.write(filled_report)
    
    return filled_report

if __name__ == "__main__":
    try:
        # 读取分析结果
        results_df = pd.read_csv("mathorcup2/音频分析结果.csv", encoding="utf-8-sig")
        print(f"已读取 {len(results_df)} 条音频分析记录")
        
        # 替代原始可视化函数，使用更可靠的直接绘图
        create_direct_plots(results_df)
        
        # 获取最佳参数
        best_speech_params = results_df[results_df["内容类型"] == "语音"].sort_values("性价比指标", ascending=False).head(3)
        best_music_params = results_df[results_df["内容类型"] == "音乐"].sort_values("性价比指标", ascending=False).head(3)
        
        # 生成报告
        report = generate_report(results_df, best_speech_params, best_music_params)
        
        print("\n语音内容最佳参数:")
        print(best_speech_params[["格式", "采样率", "比特率", "质量得分", "文件大小(KB)", "性价比指标"]].to_string(index=False))
        
        print("\n音乐内容最佳参数:")
        print(best_music_params[["格式", "采样率", "比特率", "质量得分", "文件大小(KB)", "性价比指标"]].to_string(index=False))
        
        print("\n分析完成！图表已保存到 'output_images' 目录")
        
    except Exception as e:
        import traceback
        print(f"发生错误: {e}")
        traceback.print_exc()