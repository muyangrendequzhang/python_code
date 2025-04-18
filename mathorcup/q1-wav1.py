import subprocess
import psutil
import time


def monitor_process(pid):
    try:
        process = psutil.Process(pid)
        while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
            cpu_percent = process.cpu_percent(interval=1)
            memory_percent = process.memory_percent()
            print(f"CPU 使用率: {cpu_percent}%, 内存使用率: {memory_percent}%")
            time.sleep(1)
    except psutil.NoSuchProcess:
        print("进程已结束。")


def convert_audio(input_file, output_file, codec):
    if codec == 'mp3':
        command = ['ffmpeg', '-i', input_file, '-b:a', '128k', output_file]
    elif codec == 'aac':
        command = ['ffmpeg', '-i', input_file, '-c:a', 'aac', '-b:a', '128k', output_file]
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