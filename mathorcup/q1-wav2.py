import librosa
import numpy as np

def calculate_spectral_difference(original_path, compressed_path):
    # 加载音频（统一采样率为44.1kHz）
    y_original, sr = librosa.load(original_path, sr=44100)
    y_compressed, _ = librosa.load(compressed_path, sr=44100)
    
    # 关键修改：显式使用 y= 和 sr= 关键字（librosa 0.10.0+ 要求）
    S_original = librosa.feature.melspectrogram(y=y_original, sr=sr, n_fft=2048, hop_length=512)
    S_compressed = librosa.feature.melspectrogram(y=y_compressed, sr=sr, n_fft=2048, hop_length=512)
    
    # 转换为dB尺度
    S_original_db = librosa.power_to_db(S_original, ref=np.max)
    S_compressed_db = librosa.power_to_db(S_compressed, ref=np.max)
    
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(np.mean((S_original_db - S_compressed_db)**2))
    return rmse

# 使用示例
original_file = "mathorcup/input.wav"
compressed_file = "mathorcup/output.mp3"
spectral_rmse = calculate_spectral_difference(original_file, compressed_file)
print(f"频谱RMSE（值越小损失越小）: {spectral_rmse:.2f}")