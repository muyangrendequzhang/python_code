import librosa
import numpy as np
from scipy.stats import pearsonr

def calculate_audio_difference(original_path, compressed_path):
    # 加载音频
    y_original, sr = librosa.load(original_path, sr=44100)
    y_compressed, _ = librosa.load(compressed_path, sr=44100)
    
    # 调整长度
    min_length = min(len(y_original), len(y_compressed))
    y_original = y_original[:min_length]
    y_compressed = y_compressed[:min_length]
    
    # 1. 波形相似度 (相关系数)
    waveform_corr, _ = pearsonr(y_original, y_compressed)
    
    # 2. 梅尔频谱差异
    S_original = librosa.feature.melspectrogram(y=y_original, sr=sr, n_fft=2048, hop_length=512)
    S_compressed = librosa.feature.melspectrogram(y=y_compressed, sr=sr, n_fft=2048, hop_length=512)
    S_original_db = librosa.power_to_db(S_original, ref=np.max)
    S_compressed_db = librosa.power_to_db(S_compressed, ref=np.max)
    mel_rmse = np.sqrt(np.mean((S_original_db - S_compressed_db) ** 2))
    
    # 3. MFCC差异 (更反映听觉感知)
    mfcc_original = librosa.feature.mfcc(y=y_original, sr=sr, n_mfcc=13)
    mfcc_compressed = librosa.feature.mfcc(y=y_compressed, sr=sr, n_mfcc=13)
    mfcc_rmse = np.sqrt(np.mean((mfcc_original - mfcc_compressed) ** 2))
    
    # 4. 谱质心差异 (音色特征)
    centroid_original = librosa.feature.spectral_centroid(y=y_original, sr=sr)[0]
    centroid_compressed = librosa.feature.spectral_centroid(y=y_compressed, sr=sr)[0]
    min_centroid_len = min(len(centroid_original), len(centroid_compressed))
    centroid_rmse = np.sqrt(np.mean((centroid_original[:min_centroid_len] - 
                                     centroid_compressed[:min_centroid_len]) ** 2))
    
    return {
        "waveform_correlation": waveform_corr,  # 越接近1越相似
        "mel_spectrum_rmse": mel_rmse,          # 越小越相似
        "mfcc_rmse": mfcc_rmse,                # 越小越相似
        "spectral_centroid_rmse": centroid_rmse  # 越小越相似
    }

# 使用示例
original_file = "mathorcup/input.wav"
formats = ["wav", "mp3", "aac"]
for format in formats:
    compressed_file = f"mathorcup/output.{format}"
    try:
        results = calculate_audio_difference(original_file, compressed_file)
        print(f"\n{format.upper()} 音频质量差异评估:")
        print(f"波形相关度 (1=完全相同): {results['waveform_correlation']:.4f}")
        print(f"梅尔频谱RMSE: {results['mel_spectrum_rmse']:.2f}")
        print(f"MFCC差异: {results['mfcc_rmse']:.2f}")
        print(f"谱质心差异: {results['spectral_centroid_rmse']:.2f}")
        
        # 综合得分 (自定义权重)
        score = (results['waveform_correlation'] * 40 + 
                (100 - results['mel_spectrum_rmse']) * 0.2 + 
                (100 - results['mfcc_rmse']) * 0.3 +
                (100 - results['spectral_centroid_rmse']*0.01) * 0.1)
        print(f"综合质量得分 (0-100): {min(max(score, 0), 100):.1f}")
    except FileNotFoundError:
        print(f"未找到 {compressed_file} 文件，请检查文件路径和文件名。")
    except Exception as e:
        print(f"处理 {compressed_file} 时出现错误: {e}")