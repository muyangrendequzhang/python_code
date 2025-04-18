import subprocess
import psutil
import time
import GPUtil  # 用于获取 GPU 占用率，需要额外安装：pip install GPUtil


def monitor_process(pid):
    try:
        process = psutil.Process(pid)
        total_time = 0
        total_gpu_usage = 0
        total_memory_usage = 0
        count = 0

        while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
            cpu_percent = process.cpu_percent(interval=1)
            memory_percent = process.memory_percent()
            # 获取 GPU 占用率
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
                else:
                    gpu_usage = 0
            except Exception as e:
                print(f"获取 GPU 信息时出现错误: {e}")
                gpu_usage = 0

            print(f"CPU 使用率: {cpu_percent}%, 内存使用率: {memory_percent}%, GPU 使用率: {gpu_usage}%")
            total_time += 1
            total_gpu_usage += gpu_usage
            total_memory_usage += memory_percent
            count += 1
            time.sleep(1)

        if count > 0:
            average_time = total_time
            average_gpu_usage = total_gpu_usage / count
            average_memory_usage = total_memory_usage / count
            # 权重系数
            w1 = 0.5
            w2 = 0.25
            w3 = 0.25
            # 计算编码复杂度
            complexity = w1 * average_time + w2 * average_gpu_usage + w3 * average_memory_usage
            print(f"编码复杂度: {complexity}")
        else:
            print("未获取到有效的监控数据。")
    except psutil.NoSuchProcess:
        print("进程已结束。")


def convert_audio(input_file, output_file, codec):
    if codec == 'mp3':
        command = ['ffmpeg', '-y', '-i', input_file, '-b:a', '128k', output_file]
    elif codec == 'aac':
        command = ['ffmpeg', '-y', '-i', input_file, '-c:a', 'aac', '-b:a', '128k', output_file]
    else:
        print("不支持的编码格式。")
        return

    try:
        # 启动 ffmpeg 进程
        process = subprocess.Popen(command)
        print(f"开始转换，进程 ID: {process.pid}")
        # 监控进程
        monitor_process(process.pid)
        # 等待进程结束
        process.wait()
        print("转换完成。")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")


if __name__ == "__main__":
    input_file = "mathorcup/input.wav"
    # 转换为 MP3
    output_mp3 = "output.mp3"
    convert_audio(input_file, output_mp3, 'mp3')
    # 转换为 AAC
    output_aac = "output.aac"
    convert_audio(input_file, output_aac, 'aac')
    