import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy import signal
import scipy.io.wavfile as wav
import noisereduce as nr
from sklearn.cluster import KMeans
from scipy.stats import kurtosis, skew
import pywt
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 创建输出目录
os.makedirs("output_audio", exist_ok=True)
os.makedirs("output_plots", exist_ok=True)
os.makedirs("output_results", exist_ok=True)

class NoiseAnalyzer:
    """噪声分析类"""
    def __init__(self, file_path, frame_length=2048, hop_length=512):
        """初始化噪声分析器"""
        self.file_path = file_path
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.sample_rate = None
        self.audio_data = None
        self.duration = None
        self.file_name = os.path.basename(file_path)
        self.stft = None
        self.mag_spec = None
        self.noise_profile = {}
        self.noise_types = []
        
        # 加载音频
        self._load_audio()
    
    def _load_audio(self):
        """加载音频文件"""
        try:
            self.audio_data, self.sample_rate = librosa.load(self.file_path, sr=None)
            self.duration = librosa.get_duration(y=self.audio_data, sr=self.sample_rate)
            print(f"加载音频: {self.file_name}, 采样率: {self.sample_rate}Hz, 持续时间: {self.duration:.2f}秒")
        except Exception as e:
            print(f"加载音频文件时出错: {e}")
            raise
    
    def perform_stft(self):
        """执行短时傅里叶变换(STFT)"""
        self.stft = librosa.stft(self.audio_data, 
                                n_fft=self.frame_length, 
                                hop_length=self.hop_length)
        self.mag_spec = np.abs(self.stft)
        self.phase = np.angle(self.stft)
        return self.mag_spec
    
    def visualize_spectrogram(self, title_suffix="", save=True):
        """可视化频谱图"""
        plt.figure(figsize=(12, 6))
        
        # 转换为分贝刻度以便更好地可视化
        db_spec = librosa.amplitude_to_db(self.mag_spec, ref=np.max)
        
        librosa.display.specshow(db_spec, 
                               sr=self.sample_rate, 
                               hop_length=self.hop_length, 
                               x_axis='time', 
                               y_axis='hz')
        
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{self.file_name} - 频谱图 {title_suffix}")
        
        if save:
            plt.savefig(f"output_plots/{os.path.splitext(self.file_name)[0]}_spectrogram{title_suffix}.png", 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_waveform(self, audio_data=None, title_suffix="", save=True):
        """可视化波形"""
        if audio_data is None:
            audio_data = self.audio_data
            
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(audio_data, sr=self.sample_rate)
        plt.title(f"{self.file_name} - 波形图 {title_suffix}")
        plt.xlabel("时间 (秒)")
        plt.ylabel("振幅")
        
        if save:
            plt.savefig(f"output_plots/{os.path.splitext(self.file_name)[0]}_waveform{title_suffix}.png", 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_noise_types(self):
        """分析噪声类型"""
        if self.mag_spec is None:
            self.perform_stft()
        
        # 特征提取
        features = self._extract_noise_features()
        
        # 噪声分类
        noise_types = self._classify_noise(features)
        
        return noise_types
    
    def _extract_noise_features(self):
        """提取噪声特征"""
        features = {}
        
        # 1. 频谱统计特征
        # 计算频谱的时间方向均值
        mean_spectrum = np.mean(self.mag_spec, axis=1)
        std_spectrum = np.std(self.mag_spec, axis=1)
        
        # 2. 频带能量分布
        band_energies = self._calculate_band_energies()
        
        # 3. 整体统计特征
        spectral_flatness = librosa.feature.spectral_flatness(S=self.mag_spec)[0]
        spectral_centroid = librosa.feature.spectral_centroid(S=self.mag_spec)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=self.mag_spec)[0]
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(S=self.mag_spec), axis=1)
        
        # 4. 时域统计特征
        # 计算短时帧上的统计量
        frames = librosa.util.frame(self.audio_data, frame_length=self.frame_length, hop_length=self.hop_length)
        frame_energy = np.sum(frames**2, axis=0)
        energy_std = np.std(frame_energy)
        energy_mean = np.mean(frame_energy)
        energy_ratio = energy_std / energy_mean if energy_mean > 0 else 0
        
        # 计算信号的峭度和偏度
        signal_kurtosis = kurtosis(self.audio_data)
        signal_skewness = skew(self.audio_data)
        
        # 5. 检测突发噪声
        energy_threshold = np.mean(frame_energy) + 2 * np.std(frame_energy)
        burst_indices = np.where(frame_energy > energy_threshold)[0]
        has_bursts = len(burst_indices) > 0
        burst_ratio = len(burst_indices) / len(frame_energy) if len(frame_energy) > 0 else 0
        
        # 6. 检测带状噪声
        freq_std = np.std(mean_spectrum)
        freq_peaks = signal.find_peaks(mean_spectrum, height=np.mean(mean_spectrum) + freq_std)[0]
        has_band_noise = len(freq_peaks) > 0 and len(freq_peaks) < len(mean_spectrum) * 0.1
        
        # 7. 检测背景噪声
        # 背景噪声通常在低频段且相对平稳
        low_freq_energy = np.sum(self.mag_spec[:int(self.mag_spec.shape[0] * 0.2), :])
        total_energy = np.sum(self.mag_spec)
        low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
        
        # 存储特征
        features['mean_spectrum'] = mean_spectrum
        features['std_spectrum'] = std_spectrum
        features['band_energies'] = band_energies
        features['spectral_flatness'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)
        features['spectral_centroid'] = np.mean(spectral_centroid)
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        features['spectral_contrast'] = spectral_contrast
        features['energy_ratio'] = energy_ratio
        features['signal_kurtosis'] = signal_kurtosis
        features['signal_skewness'] = signal_skewness
        features['has_bursts'] = has_bursts
        features['burst_ratio'] = burst_ratio
        features['has_band_noise'] = has_band_noise
        features['low_freq_ratio'] = low_freq_ratio
        
        self.noise_features = features
        return features
    
    def _calculate_band_energies(self, n_bands=8):
        """计算不同频带的能量"""
        # 将频谱划分为多个频带
        n_freqs = self.mag_spec.shape[0]
        band_size = n_freqs // n_bands
        band_energies = []
        
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < n_bands - 1 else n_freqs
            band_energy = np.sum(self.mag_spec[start_idx:end_idx, :])
            band_energies.append(band_energy)
        
        # 归一化
        total_energy = sum(band_energies)
        if total_energy > 0:
            band_energies = [e / total_energy for e in band_energies]
        
        return band_energies
    
    def _classify_noise(self, features):
        """基于特征分类噪声类型"""
        noise_types = []
        
        # 背景噪声检测 - 稳定且频谱较为平坦
        if features['spectral_flatness'] > 0.1 and features['energy_ratio'] < 0.5:
            if features['low_freq_ratio'] > 0.3:
                noise_types.append("低频背景噪声")
            else:
                noise_types.append("宽频背景噪声")
        
        # 突发噪声检测 - 能量突变明显
        if features['has_bursts'] and features['burst_ratio'] > 0.05:
            noise_types.append("突发噪声")
            
        # 带状噪声检测 - 某些频带能量集中
        if features['has_band_noise']:
            noise_types.append("带状噪声")
        
        # 高斯白噪声检测 - 频谱平坦且分布均匀
        if features['spectral_flatness'] > 0.5 and abs(features['signal_kurtosis']) < 1:
            noise_types.append("高斯白噪声")
        
        # 谐波噪声检测 - 频谱中有明显的峰值模式
        band_energy_std = np.std(features['band_energies'])
        if band_energy_std > 0.15 and not features['has_band_noise']:
            noise_types.append("谐波噪声")
            
        # 冲击噪声检测 - 高峭度
        if features['signal_kurtosis'] > 5 and features['burst_ratio'] > 0.01:
            noise_types.append("冲击噪声")
        
        # 如果没有检测到明显噪声特征，则认为是混合噪声
        if len(noise_types) == 0:
            noise_types.append("混合噪声")
        
        self.noise_types = noise_types
        self.noise_profile = {
            'types': noise_types,
            'features': features
        }
        
        return noise_types

class AdaptiveNoiseReducer:
    """自适应噪声消除器"""
    def __init__(self, analyzer):
        """初始化噪声消除器"""
        self.analyzer = analyzer
        self.denoised_audio = None
        self.method_used = None
        self.snr_improvement = None
    
    def reduce_noise(self):
        """根据噪声分析结果选择最佳去噪方法"""
        noise_types = self.analyzer.noise_types
        features = self.analyzer.noise_features
        
        # 根据噪声类型判断最佳去噪策略
        if "突发噪声" in noise_types and len(noise_types) == 1:
            # 对于纯突发噪声，使用小波阈值去噪
            self.denoised_audio = self._wavelet_denoising()
            self.method_used = "小波阈值去噪"
            
        elif "带状噪声" in noise_types and features['has_band_noise']:
            # 对于带状噪声，使用自适应陷波滤波
            self.denoised_audio = self._adaptive_notch_filter()
            self.method_used = "自适应陷波滤波"
            
        elif "高斯白噪声" in noise_types or "宽频背景噪声" in noise_types:
            # 对于高斯白噪声或宽频背景噪声，使用谱减法
            self.denoised_audio = self._spectral_subtraction()
            self.method_used = "谱减法"
            
        elif "低频背景噪声" in noise_types:
            # 对于低频背景噪声，使用高通滤波结合谱减法
            self.denoised_audio = self._highpass_filter_with_spectral_subtraction()
            self.method_used = "高通滤波+谱减法"
            
        elif "谐波噪声" in noise_types:
            # 对于谐波噪声，使用梳状滤波器
            self.denoised_audio = self._comb_filter()
            self.method_used = "梳状滤波"
            
        else:
            # 对于混合噪声或其他情况，使用非局部均值去噪
            self.denoised_audio = self._non_local_means_denoising()
            self.method_used = "非局部均值去噪"
        
        # 计算信噪比改善
        self._calculate_snr_improvement()
        
        return self.denoised_audio
    
    def _calculate_snr_improvement(self):
        """计算去噪前后的信噪比改善"""
        original = self.analyzer.audio_data
        denoised = self.denoised_audio
        
        # 估计原始信号的噪声部分
        noise_est = original - denoised
        
        # 计算信号能量
        signal_power = np.sum(denoised**2)
        noise_power = np.sum(noise_est**2)
        
        # 计算SNR (dB)
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
            
        # 计算原始信号的SNR
        # 假设原始信号噪声占总能量的20%（这是一个启发式估计）
        orig_noise_power = np.sum(original**2) * 0.2
        orig_signal_power = np.sum(original**2) * 0.8
        
        if orig_noise_power > 0:
            orig_snr = 10 * np.log10(orig_signal_power / orig_noise_power)
        else:
            orig_snr = float('inf')
        
        self.snr_improvement = {
            'original_snr_est': orig_snr,
            'denoised_snr': snr,
            'improvement': snr - orig_snr
        }
        
        return self.snr_improvement
    
    def _wavelet_denoising(self):
        """小波阈值去噪"""
        # 确定小波类型
        wavelet = 'db4'
        
        # 信号长度应为2的幂次方，否则会进行零填充
        coeffs = pywt.wavedec(self.analyzer.audio_data, wavelet, level=5)
        
        # 对每个分解层次进行阈值处理
        threshold = np.sqrt(2 * np.log(len(self.analyzer.audio_data)))
        for i in range(1, len(coeffs)):  # 跳过近似系数
            coeffs[i] = pywt.threshold(coeffs[i], threshold*np.std(coeffs[i])/0.6745, mode='soft')
        
        # 重构信号
        denoised_data = pywt.waverec(coeffs, wavelet)
        
        # 确保重构的信号长度与原始信号相同
        denoised_data = denoised_data[:len(self.analyzer.audio_data)]
        
        return denoised_data
    
    def _adaptive_notch_filter(self):
        """自适应陷波滤波"""
        # 首先进行STFT
        stft = self.analyzer.stft
        
        # 寻找噪声频带
        mean_spec = np.mean(np.abs(stft), axis=1)
        threshold = np.mean(mean_spec) + 1.5 * np.std(mean_spec)
        
        # 查找超过阈值的频带
        peaks, _ = signal.find_peaks(mean_spec, height=threshold)
        
        # 对每个峰值应用陷波滤波
        filtered_stft = stft.copy()
        
        for peak in peaks:
            # 建立陷波滤波掩码（根据峰值位置和宽度）
            notch_width = max(3, int(peak * 0.1))  # 假设噪声带宽
            mask = np.ones(stft.shape[0])
            start = max(0, peak - notch_width)
            end = min(stft.shape[0], peak + notch_width + 1)
            mask[start:end] = 0.1  # 降低而非完全移除，避免引入伪影
            
            # 应用滤波
            filtered_stft = filtered_stft * mask[:, np.newaxis]
        
        # 反向STFT恢复时域信号
        denoised_data = librosa.istft(filtered_stft, 
                                     hop_length=self.analyzer.hop_length, 
                                     win_length=self.analyzer.frame_length)
        
        # 确保长度匹配 - 修复fix_length调用
        # 使用切片或填充手动调整长度
        if len(denoised_data) > len(self.analyzer.audio_data):
            denoised_data = denoised_data[:len(self.analyzer.audio_data)]
        elif len(denoised_data) < len(self.analyzer.audio_data):
            padding = np.zeros(len(self.analyzer.audio_data) - len(denoised_data))
            denoised_data = np.concatenate([denoised_data, padding])
        
        return denoised_data
    
    def _spectral_subtraction(self):
        """谱减法去噪"""
        # 使用信号的前1%估计噪声特征
        noise_sample = self.analyzer.audio_data[:int(len(self.analyzer.audio_data) * 0.01)]
        
        # 使用库函数进行谱减法
        denoised_data = nr.reduce_noise(
            y=self.analyzer.audio_data,
            sr=self.analyzer.sample_rate,
            y_noise=noise_sample,
            stationary=True,
            prop_decrease=0.75
        )
        
        return denoised_data
    
    def _highpass_filter_with_spectral_subtraction(self):
        """高通滤波加谱减法"""
        # 首先应用高通滤波移除低频噪声
        nyquist = self.analyzer.sample_rate / 2
        cutoff = 120  # Hz，适合去除嗡嗡声等低频噪声
        normal_cutoff = cutoff / nyquist
        
        # 设计滤波器
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        
        # 应用滤波器
        filtered_data = signal.filtfilt(b, a, self.analyzer.audio_data)
        
        # 然后用谱减法处理剩余噪声
        # 使用信号的前1%估计噪声特征
        noise_sample = filtered_data[:int(len(filtered_data) * 0.01)]
        
        denoised_data = nr.reduce_noise(
            y=filtered_data,
            sr=self.analyzer.sample_rate,
            y_noise=noise_sample,
            stationary=True,
            prop_decrease=0.6
        )
        
        return denoised_data
    
    def _comb_filter(self):
        """梳状滤波器用于谐波噪声"""
        # 进行STFT
        stft = self.analyzer.stft
        mag = np.abs(stft)
        phase = np.angle(stft)
        
        # 找出谐波频率
        mean_spec = np.mean(mag, axis=1)
        peaks, _ = signal.find_peaks(mean_spec, prominence=0.1)
        
        # 如果找到明显的峰值
        if len(peaks) > 0:
            # 创建梳状滤波掩码
            mask = np.ones_like(mag)
            for peak in peaks:
                mask[peak-2:peak+3, :] = 0.1  # 抑制峰值周围区域
                
            # 应用掩码
            filtered_mag = mag * mask
            
            # 重构信号
            filtered_stft = filtered_mag * np.exp(1j * phase)
            denoised_data = librosa.istft(filtered_stft, 
                                        hop_length=self.analyzer.hop_length,
                                        win_length=self.analyzer.frame_length)
            
            # 确保长度匹配 - 修复fix_length调用
            if len(denoised_data) > len(self.analyzer.audio_data):
                denoised_data = denoised_data[:len(self.analyzer.audio_data)]
            elif len(denoised_data) < len(self.analyzer.audio_data):
                padding = np.zeros(len(self.analyzer.audio_data) - len(denoised_data))
                denoised_data = np.concatenate([denoised_data, padding])
        else:
            # 如果没有找到明显峰值，使用谱减法
            denoised_data = self._spectral_subtraction()
        
        return denoised_data
    
    def _non_local_means_denoising(self):
        """非局部均值去噪"""
        # 这里使用librosa的noise reduction模块
        # 使用信号的前1%估计噪声特征
        noise_sample = self.analyzer.audio_data[:int(len(self.analyzer.audio_data) * 0.01)]
        
        denoised_data = nr.reduce_noise(
            y=self.analyzer.audio_data,
            sr=self.analyzer.sample_rate,
            y_noise=noise_sample,
            prop_decrease=0.8,
            n_fft=self.analyzer.frame_length,
            win_length=self.analyzer.frame_length,
            hop_length=self.analyzer.hop_length
        )
        
        return denoised_data
    
    def save_denoised_audio(self, output_path=None):
        """保存去噪后的音频文件"""
        if self.denoised_audio is None:
            print("没有可保存的去噪音频。请先运行 reduce_noise() 方法。")
            return
        
        if output_path is None:
            base_name = os.path.splitext(self.analyzer.file_name)[0]
            output_path = f"output_audio/{base_name}_denoised.wav"
        
        try:
            sf.write(output_path, self.denoised_audio, self.analyzer.sample_rate)
            print(f"去噪音频已保存到: {output_path}")
        except Exception as e:
            print(f"保存去噪音频时出错: {e}")
            raise

def process_audio_file(file_path):
    """处理单个音频文件"""
    # 创建噪声分析器
    analyzer = NoiseAnalyzer(file_path)
    
    # 执行时频分析
    analyzer.perform_stft()
    
    # 可视化原始音频
    analyzer.visualize_waveform(title_suffix="原始")
    analyzer.visualize_spectrogram(title_suffix="原始")
    
    # 分析噪声类型
    noise_types = analyzer.analyze_noise_types()
    print(f"检测到的噪声类型: {', '.join(noise_types)}")
    
    # 创建自适应噪声消除器
    noise_reducer = AdaptiveNoiseReducer(analyzer)
    
    # 执行噪声消除
    denoised_audio = noise_reducer.reduce_noise()
    
    # 可视化去噪后的音频
    analyzer.visualize_waveform(denoised_audio, title_suffix="去噪后")
    
    # 执行去噪后的STFT并可视化
    temp_analyzer = NoiseAnalyzer(file_path)  # 创建临时分析器来处理去噪后的音频
    temp_analyzer.audio_data = denoised_audio  # 替换音频数据为去噪后的数据
    temp_analyzer.perform_stft()
    temp_analyzer.visualize_spectrogram(title_suffix="去噪后")
    
    # 保存去噪后的音频
    noise_reducer.save_denoised_audio()
    
    # 返回分析结果
    result = {
        "文件名": analyzer.file_name,
        "噪声类型": ", ".join(noise_types),
        "去噪方法": noise_reducer.method_used,
        "原始信噪比(dB)": noise_reducer.snr_improvement["original_snr_est"],
        "去噪后信噪比(dB)": noise_reducer.snr_improvement["denoised_snr"],
        "信噪比提升(dB)": noise_reducer.snr_improvement["improvement"],
        "噪声特征": analyzer.noise_features
    }
    
    return result

def process_audio_folder(folder_path):
    """处理文件夹中的所有音频文件"""
    # 获取所有支持的音频文件
    audio_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"在文件夹 {folder_path} 中没有找到支持的音频文件")
        return
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 处理每个音频文件
    results = []
    for i, audio_file in enumerate(audio_files):
        print(f"\n处理文件 {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
        try:
            result = process_audio_file(audio_file)
            results.append(result)
        except Exception as e:
            print(f"处理文件时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 将结果保存为CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("output_results/noise_analysis_results.csv", index=False)
    
    # 生成汇总报告
    generate_summary_report(results_df)

def generate_summary_report(results_df):
    """生成汇总报告"""
    # 创建Markdown报告
    report = "# 音频噪声分析与去噪汇总报告\n\n"
    
    # 添加基本统计信息
    report += f"## 基本统计\n\n"
    report += f"- 分析的音频文件数量: {len(results_df)}\n"
    
    # 噪声类型统计
    noise_types = []
    for types_str in results_df["噪声类型"]:
        for noise_type in types_str.split(", "):
            noise_types.append(noise_type)
    
    unique_types = set(noise_types)
    report += f"- 检测到的噪声类型数量: {len(unique_types)}\n"
    report += f"- 噪声类型分布:\n"
    
    for noise_type in unique_types:
        count = noise_types.count(noise_type)
        percentage = (count / len(noise_types)) * 100
        report += f"  - {noise_type}: {count} 次 ({percentage:.1f}%)\n"
    
    # 去噪方法统计
    methods = results_df["去噪方法"].value_counts()
    report += f"\n- 使用的去噪方法:\n"
    for method, count in methods.items():
        percentage = (count / len(results_df)) * 100
        report += f"  - {method}: {count} 次 ({percentage:.1f}%)\n"
    
    # SNR改善统计
    avg_snr_improvement = results_df["信噪比提升(dB)"].mean()
    max_snr_improvement = results_df["信噪比提升(dB)"].max()
    min_snr_improvement = results_df["信噪比提升(dB)"].min()
    
    report += f"\n## 信噪比改善统计\n\n"
    report += f"- 平均信噪比改善: {avg_snr_improvement:.2f} dB\n"
    report += f"- 最大信噪比改善: {max_snr_improvement:.2f} dB\n"
    report += f"- 最小信噪比改善: {min_snr_improvement:.2f} dB\n"
    
    # 详细结果表格
    report += f"\n## 详细结果\n\n"
    report += "| 文件名 | 噪声类型 | 去噪方法 | 原始信噪比(dB) | 去噪后信噪比(dB) | 信噪比提升(dB) |\n"
    report += "| --- | --- | --- | --- | --- | --- |\n"
    
    for _, row in results_df.iterrows():
        report += f"| {row['文件名']} | {row['噪声类型']} | {row['去噪方法']} | {row['原始信噪比(dB)']:.2f} | {row['去噪后信噪比(dB)']:.2f} | {row['信噪比提升(dB)']:.2f} |\n"
    
    # 各种噪声类型的适用范围与局限性分析
    report += f"\n## 噪声类型的适用范围与局限性\n\n"
    
    report += "### 背景噪声\n\n"
    report += "- **适用范围**：背景噪声一般是稳定存在的宽频带噪声，如风噪、环境嗡嗡声等。\n"
    report += "- **处理方法**：谱减法、维纳滤波。\n"
    report += "- **局限性**：难以完全去除而不影响语音质量，尤其是当噪声与语音频谱重叠较多时。\n"
    
    report += "\n### 突发噪声\n\n"
    report += "- **适用范围**：短暂、高强度的噪声，如敲击声、爆破声等。\n"
    report += "- **处理方法**：小波阈值去噪。\n"
    report += "- **局限性**：时间定位准确但可能会引入伪影，难以保持音频的自然过渡。\n"
    
    report += "\n### 带状噪声\n\n"
    report += "- **适用范围**：集中在特定频带的噪声，如电源线嗡嗡声、机械噪声等。\n"
    report += "- **处理方法**：自适应陷波滤波。\n"
    report += "- **局限性**：当噪声频带与语音频带重叠时，会导致语音信息损失。\n"
    
    report += "\n### 高斯白噪声\n\n"
    report += "- **适用范围**：均匀分布在各个频率的随机噪声。\n"
    report += "- **处理方法**：谱减法。\n"
    report += "- **局限性**：难以在保持信号质量的同时完全去除噪声，尤其是在低SNR情况下。\n"
    
    report += "\n### 谐波噪声\n\n"
    report += "- **适用范围**：具有周期性的噪声，如电机噪声、机械设备噪声等。\n"
    report += "- **处理方法**：梳状滤波。\n"
    report += "- **局限性**：可能会过度平滑信号，降低音质；难以处理频率变化的谐波噪声。\n"
    
    report += "\n## 总结\n\n"
    report += "不同类型的噪声需要不同的处理方法。自适应去噪算法能根据噪声特性选择最佳方法，但所有方法都存在信号保真度与噪声去除效果之间的权衡。在实际应用中，应根据具体场景调整算法参数，以达到最佳效果。\n"
    
    # 保存报告
    with open("output_results/noise_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n汇总报告已保存到: output_results/noise_analysis_report.md")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="音频噪声分析与自适应去噪")
    parser.add_argument("path", help="mathorcp4/附件2")
    
    args = parser.parse_args()
    
    path = args.path
    
    if os.path.isdir(path):
        process_audio_folder(path)
    elif os.path.isfile(path):
        result = process_audio_file(path)
        print(f"\n分析结果: {result}")
    else:
        print(f"错误: 路径 {path} 不存在或不是有效的文件/文件夹")