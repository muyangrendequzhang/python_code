import os
import csv
import subprocess
import psutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import librosa
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

def analyze_audio_formats(source_wav, output_folder="output_formats"):
    """完整分析WAV/MP3/AAC三种格式的性能特征"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    base_name = os.path.splitext(os.path.basename(source_wav))[0]
    
    # 生成不同格式和比特率的版本
    formats = [
        {"name": "原始WAV", "ext": "wav", "bitrate": None},
        {"name": "MP3高质量", "ext": "mp3", "bitrate": "320k"},
        {"name": "MP3中质量", "ext": "mp3", "bitrate": "128k"},
        {"name": "AAC高质量", "ext": "aac", "bitrate": "256k"},
        {"name": "AAC中质量", "ext": "aac", "bitrate": "128k"}
    ]
    
    results = []
    
    # 1. 创建各种格式的文件
    output_files = []
    for fmt in formats:
        if fmt["ext"] == "wav" and fmt["bitrate"] is None:
            output_files.append({"format": fmt, "path": source_wav})
            continue
            
        output_path = os.path.join(output_folder, f"{base_name}_{fmt['bitrate'] or 'lossless'}.{fmt['ext']}")
        stats = convert_audio(source_wav, output_path, fmt["ext"], fmt["bitrate"])
        output_files.append({
            "format": fmt, 
            "path": output_path,
            "conversion_stats": stats
        })
    
    # 2. 分析各文件的特性
    for output in output_files:
        # 基本信息
        info = get_audio_info(output["path"], source_wav)
        
        # 计算文件大小 (KB)
        file_size_kb = os.path.getsize(output["path"]) / 1024
        
        # 获取转换时的CPU和内存消耗 (如果有)
        cpu_usage = output.get("conversion_stats", {}).get("avg_cpu", 0)
        mem_usage = output.get("conversion_stats", {}).get("avg_mem", 0)
        
        # 添加到结果
        result = {
            "格式": output["format"]["name"],
            "文件大小(KB)": round(file_size_kb, 2),
            "编码CPU使用率(%)": round(cpu_usage, 2) if cpu_usage else "N/A",
            "编码内存使用率(%)": round(mem_usage, 2) if mem_usage else "N/A",
        }
        
        # 合并音质分析结果
        if "波形相关度" in info:
            result.update({
                "波形相关度": float(info["波形相关度"]),
                "梅尔频谱RMSE": float(info["梅尔频谱RMSE"]),
                "MFCC差异": float(info["MFCC差异"]),
                "质量综合得分": float(info["质量综合得分"]) if "质量综合得分" in info else 100
            })
        
        results.append(result)
    
    # 3. 使用回归模型计算权重
    weights = calculate_format_weights(results)
    
    # 4. 生成最终评估报告
    generate_analysis_report(results, weights, "audio_format_analysis.csv")
    
    return results, weights

def calculate_format_weights(results):
    """使用回归分析计算各因素权重"""
    # 提取特征
    X = []
    formats = []
    
    for result in results:
        if "波形相关度" in result:  # 只考虑有完整数据的结果
            formats.append(result["格式"])
            # 归一化特征
            file_size = result["文件大小(KB)"]
            quality_score = result["质量综合得分"]
            cpu_usage = result["编码CPU使用率(%)"] if result["编码CPU使用率(%)"] != "N/A" else 0
            
            X.append([file_size, quality_score, cpu_usage])
    
    X_array = np.array(X)
    
    # 特征归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_array)
    
    # 专家评分 - 这里使用一个假设的评分，实际应基于真实应用场景评估
    # WAV > AAC高质量 > MP3高质量 > AAC中质量 > MP3中质量
    expert_scores = {
        "原始WAV": 10,
        "AAC高质量": 8.5,
        "MP3高质量": 8.0,
        "AAC中质量": 7.0,
        "MP3中质量": 6.0
    }
    
    y = np.array([expert_scores.get(fmt, 5) for fmt in formats])
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 线性回归模型
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    
    # 系数和截距
    weights = {
        "文件大小权重": round(abs(model.coef_[0]), 4),
        "音质权重": round(model.coef_[1], 4),
        "计算复杂度权重": round(abs(model.coef_[2]), 4),
        "截距": round(model.intercept_, 4)
    }
    
    return weights

def generate_analysis_report(results, weights, output_file):
    """生成分析报告和适用场景建议"""
    df = pd.DataFrame(results)
    
    # 为每个格式添加场景适用性评分
    scenarios = []
    for _, row in df.iterrows():
        # 空间受限/移动场景 (文件大小权重高)
        mobile_score = 10 - (row["文件大小(KB)"] / 1000) * 10
        
        # 高保真场景 (音质权重高)
        hifi_score = row["质量综合得分"] if "质量综合得分" in row else 0
        
        # 实时处理场景 (计算复杂度权重高)
        realtime_score = 10 - (row["编码CPU使用率(%)"] if row["编码CPU使用率(%)"] != "N/A" else 5)
        
        scenarios.append({
            "格式": row["格式"],
            "移动设备适用性": round(min(max(mobile_score, 0), 10), 1),
            "高保真场景适用性": round(min(max(hifi_score, 0), 10), 1),
            "实时处理适用性": round(min(max(realtime_score, 0), 10), 1)
        })
    
    # 添加适用场景信息
    scenario_df = pd.DataFrame(scenarios)
    
    # 合并数据
    results_with_scenarios = pd.merge(df, scenario_df, on="格式")
    
    # 将权重信息添加到报告中
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        results_with_scenarios.to_csv(f, index=False)
        f.write("\n权重分析:\n")
        for key, value in weights.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n格式适用场景建议:\n")
        f.write("1. WAV: 专业录音、音频编辑、归档保存\n")
        f.write("2. 高比特率AAC: 高质量流媒体、音乐服务\n")
        f.write("3. 高比特率MP3: 通用音乐播放、兼容性要求高的场景\n")
        f.write("4. 中比特率AAC: 移动设备、视频配音\n") 
        f.write("5. 中比特率MP3: 语音内容、低带宽场景\n")
    
    print(f"分析报告已保存至 {output_file}")
    
    # 可视化比较
    plot_format_comparison(results)
    
    return results_with_scenarios

def plot_format_comparison(results):
    """生成音频格式比较的可视化图表"""
    df = pd.DataFrame(results)
    
    # 设置样式
    plt.style.use('ggplot')
    
    # 图1: 文件大小比较
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.barplot(x='格式', y='文件大小(KB)', data=df)
    plt.title('不同格式的文件大小比较')
    plt.xticks(rotation=45)
    
    # 图2: 音质评分比较
    plt.subplot(2, 2, 2)
    if '质量综合得分' in df.columns:
        sns.barplot(x='格式', y='质量综合得分', data=df)
        plt.title('不同格式的音质评分')
        plt.xticks(rotation=45)
    
    # 图3: 资源消耗比较
    plt.subplot(2, 2, 3)
    df_cpu = df[df['编码CPU使用率(%)'] != 'N/A'].copy()
    if not df_cpu.empty:
        sns.barplot(x='格式', y='编码CPU使用率(%)', data=df_cpu)
        plt.title('不同格式的CPU使用率')
        plt.xticks(rotation=45)
    
    # 图4: 综合雷达图准备
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, '详细分析请参见报告文件', 
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('audio_format_comparison.png')
    print("可视化图表已保存为 audio_format_comparison.png")

# 继续使用之前已定义的函数
def get_audio_info(file_path, reference_file=None):
    """获取音频文件的详细信息"""
    # [使用之前提供的代码]
    info = {'文件名': os.path.basename(file_path), 
            '类型': os.path.splitext(file_path)[1][1:]}
    
    # 使用ffprobe获取基本音频信息
    try:
        # 获取声道数
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', 
               '-show_entries', 'stream=channels', '-of', 'csv=p=0', file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        info['声道数'] = result.stdout.strip()
        
        # 获取采样率
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', 
               '-show_entries', 'stream=sample_rate', '-of', 'csv=p=0', file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        info['采样率'] = result.stdout.strip()
        
        # 获取比特率
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', 
               '-show_entries', 'stream=bit_rate', '-of', 'csv=p=0', file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        info['比特率'] = result.stdout.strip()
        
        # 获取持续时间
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', 
               '-show_entries', 'format=duration', '-of', 'csv=p=0', file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        info['时长(秒)'] = result.stdout.strip()
        
    except Exception as e:
        print(f"获取'{file_path}'基本信息时出错: {e}")
    
    # 如果提供了参考文件进行质量分析
    if reference_file and os.path.exists(reference_file) and file_path != reference_file:
        try:
            quality_metrics = calculate_audio_difference(reference_file, file_path)
            info.update({
                '波形相关度': f"{quality_metrics['waveform_correlation']:.4f}",
                '梅尔频谱RMSE': f"{quality_metrics['mel_spectrum_rmse']:.2f}",
                'MFCC差异': f"{quality_metrics['mfcc_rmse']:.2f}",
                '谱质心差异': f"{quality_metrics['spectral_centroid_rmse']:.2f}",
                '质量综合得分': f"{quality_metrics['quality_score']:.1f}"
            })
        except Exception as e:
            print(f"计算'{file_path}'质量指标时出错: {e}")
    
    return info

def calculate_audio_difference(original_path, compressed_path):
    """计算两个音频文件之间的质量差异"""
    # [使用之前提供的代码]
    try:
        # 加载音频
        y_original, sr = librosa.load(original_path, sr=None)
        y_compressed, sr_comp = librosa.load(compressed_path, sr=None)
        
        # 采样率调整（确保相同）
        if sr != sr_comp:
            y_compressed = librosa.resample(y_compressed, orig_sr=sr_comp, target_sr=sr)
        
        # 调整长度
        min_length = min(len(y_original), len(y_compressed))
        y_original = y_original[:min_length]
        y_compressed = y_compressed[:min_length]
        
        # 1. 波形相似度 (相关系数)
        waveform_corr, _ = pearsonr(y_original, y_compressed)
        
        # 2. 梅尔频谱差异
        S_original = librosa.feature.melspectrogram(y=y_original, sr=sr)
        S_compressed = librosa.feature.melspectrogram(y=y_compressed, sr=sr)
        S_original_db = librosa.power_to_db(S_original, ref=np.max)
        S_compressed_db = librosa.power_to_db(S_compressed, ref=np.max)
        mel_rmse = np.sqrt(np.mean((S_original_db - S_compressed_db) ** 2))
        
        # 3. MFCC差异 (反映听觉感知)
        mfcc_original = librosa.feature.mfcc(y=y_original, sr=sr, n_mfcc=13)
        mfcc_compressed = librosa.feature.mfcc(y=y_compressed, sr=sr, n_mfcc=13)
        mfcc_rmse = np.sqrt(np.mean((mfcc_original - mfcc_compressed) ** 2))
        
        # 4. 谱质心差异 (音色特征)
        centroid_original = librosa.feature.spectral_centroid(y=y_original, sr=sr)[0]
        centroid_compressed = librosa.feature.spectral_centroid(y=y_compressed, sr=sr)[0]
        min_centroid_len = min(len(centroid_original), len(centroid_compressed))
        centroid_rmse = np.sqrt(np.mean((centroid_original[:min_centroid_len] - 
                                         centroid_compressed[:min_centroid_len]) ** 2))
        
        # 综合得分 (自定义权重)
        score = (waveform_corr * 40 + 
                (100 - mel_rmse) * 0.2 + 
                (100 - mfcc_rmse) * 0.3 +
                (100 - centroid_rmse*0.01) * 0.1)
        quality_score = min(max(score, 0), 100)
        
        return {
            "waveform_correlation": waveform_corr,
            "mel_spectrum_rmse": mel_rmse,
            "mfcc_rmse": mfcc_rmse,
            "spectral_centroid_rmse": centroid_rmse,
            "quality_score": quality_score
        }
    except Exception as e:
        print(f"音频质量分析出错: {e}")
        return {
            "waveform_correlation": 0,
            "mel_spectrum_rmse": 0,
            "mfcc_rmse": 0,
            "spectral_centroid_rmse": 0,
            "quality_score": 0
        }

def convert_audio(input_file, output_file, codec, bitrate='128k', sample_rate=48000):
    """转换音频文件并监控性能"""
    # [使用之前提供的代码]
    if codec == 'mp3':
        command = ['ffmpeg', '-i', input_file, '-b:a', bitrate, '-ar', str(sample_rate), output_file]
    elif codec == 'aac':
        command = ['ffmpeg', '-i', input_file, '-c:a', 'aac', '-b:a', bitrate, '-ar', str(sample_rate), output_file]
    elif codec == 'wav':
        bit_depth = 16  # 默认值
        command = ['ffmpeg', '-i', input_file, '-ar', str(sample_rate), 
                  '-bits_per_raw_sample', str(bit_depth), output_file]
    else:
        print(f"不支持的编码格式: {codec}")
        return None
    
    try:
        print(f"开始转换 {input_file} 为 {codec} 格式...")
        # 启动 ffmpeg 进程
        process = subprocess.Popen(command)
        print(f"进程 ID: {process.pid}")
        # 监控进程
        usage_stats = monitor_process(process.pid)
        # 等待进程结束
        process.wait()
        print(f"{codec} 转换完成。")
        return usage_stats
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        return None

def monitor_process(pid):
    """监控进程的CPU和内存使用情况"""
    # [使用之前提供的代码]
    usage_data = []
    try:
        process = psutil.Process(pid)
        while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
            cpu_percent = process.cpu_percent(interval=0.5)
            memory_percent = process.memory_percent()
            usage_data.append((cpu_percent, memory_percent))
            print(f"CPU 使用率: {cpu_percent}%, 内存使用率: {memory_percent}%")
            time.sleep(0.5)
    except psutil.NoSuchProcess:
        print("进程已结束。")
    
    # 计算平均使用率
    if usage_data:
        avg_cpu = sum(u[0] for u in usage_data) / len(usage_data)
        avg_mem = sum(u[1] for u in usage_data) / len(usage_data)
        return {"avg_cpu": avg_cpu, "avg_mem": avg_mem}
    return {"avg_cpu": 0, "avg_mem": 0}

def calculate_afbi(format_data, scenario_type):
    """计算音频格式平衡指标(AFBI)"""
    # 1. 获取场景权重
    weights = get_scenario_weights(scenario_type)
    
    # 2. 计算存储效率得分 (文件越小分越高，10分满分)
    storage_score = 10 - min(format_data.get("文件大小(KB)", 0) / reference_size, 10)
    
    # 3. 计算音质保真度得分 (0-10分)
    if "质量综合得分" in format_data and format_data["质量综合得分"] not in (None, "N/A"):
        quality_score = float(format_data["质量综合得分"]) / 10
    else:
        quality_score = estimate_quality_by_format(format_data.get("格式", "").lower())
    
    # 4. 计算编解码复杂度得分 (资源消耗越低分越高，10分满分)
    cpu_usage = format_data.get("编码CPU使用率(%)", "N/A")
    if cpu_usage == "N/A" or cpu_usage is None:
        complexity_score = 5.0  # 默认中等复杂度
    else:
        complexity_score = 10 - min(float(cpu_usage) / 10, 10)
    
    # 5. 计算AFBI综合得分
    afbi_score = (
        storage_score * weights["文件大小权重"] +
        quality_score * weights["音质权重"] +
        complexity_score * weights["计算复杂度权重"]
    )
    
    return {
        "AFBI得分": round(afbi_score, 2),
        "存储效率得分": round(storage_score, 2),
        "音质保真度得分": round(quality_score, 2),
        "编解码复杂度得分": round(complexity_score, 2)
    }

if __name__ == "__main__":
    # 指定原始WAV文件路径
    source_wav = "mathorcup/input.wav"  # 替换为实际的WAV文件路径
    
    # 运行分析
    results, weights = analyze_audio_formats(source_wav)
    
    print("\n格式权重分析结果:")
    for key, value in weights.items():
        print(f"{key}: {value}")
    
    print("\n分析结果和适用场景建议已保存到文件中")