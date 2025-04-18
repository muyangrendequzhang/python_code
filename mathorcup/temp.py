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

def monitor_process(pid):
    """监控进程的CPU和内存使用情况"""
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

def get_audio_info(file_path, reference_file=None):
    """获取音频文件的详细信息，包括基本属性和质量指标"""
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
    """计算两个音频文件之间的质量差异 - 整合q1-wav2.py中的参数"""
    try:
        # 加载音频 - 使用q1-wav2.py中的44100采样率
        y_original, sr = librosa.load(original_path, sr=44100)
        y_compressed, sr_comp = librosa.load(compressed_path, sr=44100)
        
        # 采样率调整（确保相同）
        if sr != sr_comp:
            y_compressed = librosa.resample(y_compressed, orig_sr=sr_comp, target_sr=sr)
        
        # 调整长度
        min_length = min(len(y_original), len(y_compressed))
        y_original = y_original[:min_length]
        y_compressed = y_compressed[:min_length]
        
        # 1. 波形相似度 (相关系数)
        waveform_corr, _ = pearsonr(y_original, y_compressed)
        
        # 2. 梅尔频谱差异 - 使用q1-wav2.py中的参数
        S_original = librosa.feature.melspectrogram(y=y_original, sr=sr, n_fft=2048, hop_length=512)
        S_compressed = librosa.feature.melspectrogram(y=y_compressed, sr=sr, n_fft=2048, hop_length=512)
        S_original_db = librosa.power_to_db(S_original, ref=np.max)
        S_compressed_db = librosa.power_to_db(S_compressed, ref=np.max)
        mel_rmse = np.sqrt(np.mean((S_original_db - S_compressed_db) ** 2))
        
        # 3. MFCC差异 - 使用q1-wav2.py中的参数
        mfcc_original = librosa.feature.mfcc(y=y_original, sr=sr, n_mfcc=13)
        mfcc_compressed = librosa.feature.mfcc(y=y_compressed, sr=sr, n_mfcc=13)
        mfcc_rmse = np.sqrt(np.mean((mfcc_original - mfcc_compressed) ** 2))
        
        # 4. 谱质心差异 - 使用q1-wav2.py中的计算方法
        centroid_original = librosa.feature.spectral_centroid(y=y_original, sr=sr)[0]
        centroid_compressed = librosa.feature.spectral_centroid(y=y_compressed, sr=sr)[0]
        min_centroid_len = min(len(centroid_original), len(centroid_compressed))
        centroid_rmse = np.sqrt(np.mean((centroid_original[:min_centroid_len] - 
                                         centroid_compressed[:min_centroid_len]) ** 2))
        
        # 综合得分 - 使用q1-wav2.py中的权重
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
    """转换音频文件并监控性能 - 整合q1-wav1.py中的参数"""
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
    
    # 5. 生成独立分析图表
    plot_audio_analysis(results, output_folder="format_analysis")
    
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
    
    # 确保最小权重不为零
    min_weight = 0.05
    for key in ["文件大小权重", "音质权重", "计算复杂度权重"]:
        if weights[key] < min_weight:
            weights[key] = min_weight
            
    # 重新归一化权重
    total = sum(weights[k] for k in ["文件大小权重", "音质权重", "计算复杂度权重"])
    for key in ["文件大小权重", "音质权重", "计算复杂度权重"]:
        weights[key] = round(weights[key] / total, 4)
    
    return weights

def get_scenario_weights(scenario):
    """根据不同应用场景返回优化的权重配置"""
    scenarios = {
        "专业音频制作": {
            "文件大小权重": 0.15,
            "音质权重": 0.75, 
            "计算复杂度权重": 0.10
        },
        "移动设备": {
            "文件大小权重": 0.50,
            "音质权重": 0.30,
            "计算复杂度权重": 0.20
        },
        "流媒体服务": {
            "文件大小权重": 0.45,
            "音质权重": 0.40,
            "计算复杂度权重": 0.15
        },
        "归档存储": {
            "文件大小权重": 0.35,
            "音质权重": 0.60,
            "计算复杂度权重": 0.05
        },
        "实时处理": {
            "文件大小权重": 0.30,
            "音质权重": 0.20,
            "计算复杂度权重": 0.50
        },
        "语音内容": {
            "文件大小权重": 0.55,
            "音质权重": 0.30,
            "计算复杂度权重": 0.15
        }
    }
    return scenarios.get(scenario, scenarios["流媒体服务"])  # 默认返回流媒体权重

def calculate_format_score(format_data, scenario_weights):
    """根据场景权重计算格式得分"""
    try:
        # 文件大小评分 - 文件越小分越高
        size_score = 10 - min(format_data.get("文件大小(KB)", 0) / 1000, 10)
        
        # 音质评分
        if "质量综合得分" in format_data and format_data["质量综合得分"] not in (None, "N/A"):
            try:
                quality_score = float(format_data["质量综合得分"]) / 10
            except:
                quality_score = estimate_quality_by_format(format_data.get("格式", "").lower())
        else:
            quality_score = estimate_quality_by_format(format_data.get("格式", "").lower())
        
        # CPU使用率评分 - 使用率越低分越高
        cpu_usage = format_data.get("编码CPU使用率(%)", "N/A")
        if cpu_usage == "N/A" or cpu_usage is None:
            complexity_score = 5.0  # 默认中等复杂度
        else:
            try:
                complexity_score = 10 - min(float(cpu_usage) / 10, 10)
            except:
                complexity_score = 5.0
        
        # 加权计算总分
        weighted_score = (
            size_score * scenario_weights["文件大小权重"] +
            quality_score * scenario_weights["音质权重"] +
            complexity_score * scenario_weights["计算复杂度权重"]
        )
        
        return round(weighted_score, 1)
    except Exception as e:
        print(f"计算格式得分时出错: {e}")
        return 5.0  # 默认中等分数

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

    
    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
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

def plot_audio_analysis(results, output_folder="format_analysis"):
    """为每个音频格式生成独立的详细分析图表"""
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 为每个格式生成单独分析图表
    for result in results:
        format_name = result["格式"]
        plt.figure(figsize=(10, 12))
        
        # 1. 文件信息部分
        plt.subplot(4, 1, 1)
        plt.axis('off')
        info_text = (
            f"格式: {format_name}\n"
            f"文件大小: {result['文件大小(KB)']:.2f} KB\n"
        )
        
        # 添加其他可用的信息
        for key in ["采样率", "声道数", "比特率", "时长(秒)"]:
            if key in result:
                info_text += f"{key}: {result[key]}\n"
        
        plt.text(0.1, 0.5, info_text, fontsize=12)
        plt.title(f"{format_name} 基本信息", fontsize=14)
        
        # 2. 音质分析部分 (如果有相关数据)
        quality_metrics = ["波形相关度", "梅尔频谱RMSE", "MFCC差异", "质量综合得分"]
        quality_data_exists = any(metric in result for metric in quality_metrics)

        if quality_data_exists:
            plt.subplot(4, 1, 2)
            
            metrics_names = []
            normalized_values = []
            colors = []
            
            # 逐个处理指标，允许部分缺失
            if "波形相关度" in result:
                try:
                    value = float(result["波形相关度"])
                    metrics_names.append("波形相关度")
                    normalized_values.append(value)
                    colors.append('green')
                except: pass
                
            if "梅尔频谱RMSE" in result:
                try:
                    value = min(float(result["梅尔频谱RMSE"])/50, 1)
                    metrics_names.append("梅尔频谱RMSE")
                    normalized_values.append(value)
                    colors.append('red')
                except: pass
                
            if "MFCC差异" in result:
                try:
                    value = min(float(result["MFCC差异"])/50, 1)
                    metrics_names.append("MFCC差异")
                    normalized_values.append(value)
                    colors.append('red')
                except: pass
                
            if "质量综合得分" in result:
                try:
                    value = float(result["质量综合得分"])/100
                    metrics_names.append("质量综合得分")
                    normalized_values.append(value)
                    colors.append('blue')
                except: pass
            
            if metrics_names:  # 只有当有数据时才绘图
                plt.bar(metrics_names, normalized_values, color=colors)
                plt.ylim(0, 1)
                plt.title(f"{format_name} 音质分析 (归一化显示)", fontsize=14)
                plt.xticks(rotation=30)
            else:
                plt.text(0.5, 0.5, "无音质分析数据", 
                        horizontalalignment='center',
                        verticalalignment='center')
                plt.axis('off')
        
        # 3. 计算复杂度分析 (如果有相关数据)
        resource_metrics = ["编码CPU使用率(%)", "编码内存使用率(%)"]
        valid_resources = []
        resource_values = []
        resource_colors = []

        for metric in resource_metrics:
            if metric in result and result[metric] != "N/A":
                try:
                    value = float(result[metric])
                    valid_resources.append(metric)
                    resource_values.append(value)
                    resource_colors.append('orange' if "CPU" in metric else 'purple')
                except: pass

        if valid_resources:  # 只有有数据时才绘图
            plt.subplot(4, 1, 3)
            plt.bar(valid_resources, resource_values, color=resource_colors)
            plt.title(f"{format_name} 资源消耗分析", fontsize=14)
            plt.ylim(0, max(resource_values) * 1.2 if resource_values else 10)
        
        # 4. 适用场景雷达图
        plt.subplot(4, 1, 4)
        scenario_names = ["专业音频制作", "移动设备", "流媒体服务", "归档存储", "实时处理", "语音内容"]
        
        # 为每个场景计算适用性分数 (示例计算，实际应根据特性确定)
        scenario_scores = []
        for scenario in scenario_names:
            weights = get_scenario_weights(scenario)
            score = calculate_format_score(result, weights)
            scenario_scores.append(score/10)  # 归一化为0-1范围
        
        # 绘制雷达图
        angles = np.linspace(0, 2*np.pi, len(scenario_names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        scenario_scores += scenario_scores[:1]  # 闭合数据
        
        ax = plt.subplot(4, 1, 4, polar=True)
        ax.plot(angles, scenario_scores, 'o-', linewidth=2)
        ax.fill(angles, scenario_scores, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), scenario_names)
        ax.set_ylim(0, 1)
        plt.title(f"{format_name} 场景适用性分析", fontsize=14)
        
        # 保存图表
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(output_folder, f"{format_name}.png"))
        plt.close()
        
    print(f"已为每个音频格式生成独立分析图表，保存在 {output_folder} 文件夹")

def estimate_quality_by_format(format_name):
    """基于格式名称估计音质得分"""
    if "wav" in format_name:
        return 10.0
    elif "aac" in format_name and "高" in format_name:
        return 8.5
    elif "aac" in format_name:
        return 7.5
    elif "mp3" in format_name and "高" in format_name:
        return 8.0
    elif "mp3" in format_name:
        return 6.5
    else:
        return 5.0  # 默认值

def analyze_multiple_audio_files(audio_folder, output_csv="音频格式分析结果.csv"):
    """分析文件夹中的多个音频文件并生成汇总表格"""
    all_results = []
    format_scores = {"wav": [], "aac": [], "mp3": []}
    
    # 获取所有音频文件
    audio_files = []
    for root, _, files in os.walk(audio_folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.wav', '.mp3', '.aac']:
                audio_files.append(os.path.join(root, file))
    
    print(f"找到 {len(audio_files)} 个音频文件进行分析")
    
    # 根据格式分类文件
    wav_files = [f for f in audio_files if f.lower().endswith('.wav')]
    mp3_files = [f for f in audio_files if f.lower().endswith('.mp3')]
    aac_files = [f for f in audio_files if f.lower().endswith('.aac')]
    
    # 找出参考文件(WAV格式，如果有)
    reference_files = {}
    for audio_file in wav_files:
        file_name = os.path.basename(audio_file)
        base_name = os.path.splitext(file_name)[0]
        reference_files[base_name] = audio_file
    
    # 寻找参考WAV文件 (用于质量比较的基准)
    master_reference = None
    for audio_file in audio_files:
        if audio_file.lower().endswith('.wav'):
            master_reference = audio_file
            print(f"选择主参考文件: {master_reference}")
            break
    
    if not master_reference:
        print("警告: 未找到WAV参考文件，质量分析将不可用")
    
    # 分析每个文件
    for i, audio_file in enumerate(audio_files):
        print(f"处理文件 {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
        file_name = os.path.basename(audio_file)
        base_name = os.path.splitext(file_name)[0]
        file_ext = os.path.splitext(file_name)[1][1:].lower()
        
        # 尝试获取与此文件相关的参考文件
        reference_file = reference_files.get(base_name)
        
        # 获取基本信息 - 加入参考文件进行质量比较
        info = get_audio_info(audio_file, reference_file)
        file_size_kb = os.path.getsize(audio_file) / 1024
        
        result = {
            "文件名": file_name,
            "格式": file_ext.upper(),
            "文件大小(KB)": round(file_size_kb, 2)
        }
        
        # 添加音频信息
        for key in ["采样率", "声道数", "比特率", "时长(秒)"]:
            if key in info:
                result[key] = info[key]
        
        # 强制使用主参考文件进行质量分析
        if master_reference and audio_file != master_reference:
            try:
                quality_metrics = calculate_audio_difference(master_reference, audio_file)
                result.update({
                    "波形相关度": round(quality_metrics["waveform_correlation"], 4),
                    "梅尔频谱RMSE": round(quality_metrics["mel_spectrum_rmse"], 2),
                    "MFCC差异": round(quality_metrics["mfcc_rmse"], 2),
                    "谱质心差异": round(quality_metrics["spectral_centroid_rmse"], 2),
                    "质量综合得分": round(quality_metrics["quality_score"], 1)
                })
                print(f"已完成质量分析: {file_name}")
            except Exception as e:
                print(f"质量分析失败: {e}")
        
        # 为不同场景计算得分
        scenario_names = ["专业音频制作", "移动设备", "流媒体服务", "归档存储", "实时处理", "语音内容"]
        for scenario in scenario_names:
            weights = get_scenario_weights(scenario)
            score = calculate_format_score(result, weights)
            result[f"{scenario}得分"] = score
            
            # 记录该格式得分用于计算平均值
            if file_ext in format_scores:
                format_scores[file_ext].append(score)
        
        all_results.append(result)
    
    # 计算每种格式的平均得分
    avg_scores = {}
    for fmt, scores in format_scores.items():
        if scores:
            # 计算总体平均分
            avg_overall = sum(scores) / len(scores)
            avg_scores[f"{fmt.upper()}平均得分"] = round(avg_overall, 2)
            
            # 计算各场景平均分
            for scenario in scenario_names:
                scenario_scores = [
                    result[f"{scenario}得分"] 
                    for result in all_results 
                    if result["格式"].lower() == fmt
                ]
                if scenario_scores:
                    avg_scores[f"{fmt.upper()}-{scenario}平均得分"] = round(sum(scenario_scores) / len(scenario_scores), 2)
    
    # 添加平均得分行
    avg_row = {"文件名": "平均得分", "格式": "汇总"}
    avg_row.update(avg_scores)
    all_results.append(avg_row)
    
    # 保存到CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")  # 使用utf-8-sig支持Excel中文
    print(f"分析结果已保存至 {output_csv}")
    
    # 创建格式比较图表
    plot_format_comparison_chart(avg_scores, scenario_names)
    
    return all_results, avg_scores

def plot_format_comparison_chart(avg_scores, scenario_names):
    """绘制WAV、AAC、MP3三种格式在各场景下的得分比较图表"""
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据
    formats = ["WAV", "MP3", "AAC"]
    format_scenario_scores = {fmt: [] for fmt in formats}
    
    for scenario in scenario_names:
        for fmt in formats:
            key = f"{fmt}-{scenario}平均得分"
            if key in avg_scores:
                format_scenario_scores[fmt].append(avg_scores[key])
            else:
                format_scenario_scores[fmt].append(0)
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 条形图
    x = np.arange(len(scenario_names))
    width = 0.25
    
    plt.bar(x - width, format_scenario_scores["WAV"], width, label='WAV', color='#3498db')
    plt.bar(x, format_scenario_scores["AAC"], width, label='AAC', color='#2ecc71')
    plt.bar(x + width, format_scenario_scores["MP3"], width, label='MP3', color='#e74c3c')
    
    plt.ylabel('得分 (0-10)')
    plt.title('WAV/AAC/MP3在不同场景下的平均得分比较')
    plt.xticks(x, scenario_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('音频格式场景比较.png')
    print("格式比较图表已保存为 音频格式场景比较.png")

if __name__ == "__main__":
    # 音频文件夹路径
    audio_folder = "mathorcup/附件1"  # 修改为您的音频文件夹路径
    if not os.path.exists(audio_folder):
        print(f"错误: 文件夹 {audio_folder} 不存在!")
        exit(1)
    
    # 分析多个音频文件
    results, avg_scores = analyze_multiple_audio_files(audio_folder)
    
    print("\n各格式平均得分:")
    for key, value in avg_scores.items():
        if not "-" in key:  # 只显示总体平均分
            print(f"{key}: {value}")
    
    print("\n分析结果已保存到CSV文件中")