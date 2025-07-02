# 分配GPU
# 输出：结果音频（人声+混声）。output/
# 1_vocal.wav
# 1_mixed.mp3
# 一个json，拆分（只要音频）

import os
import json
import argparse
import subprocess
import tempfile
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, \
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount, nvmlShutdown

def get_available_gpus(min_free_memory_gb=8):
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    available = []

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        free_gb = mem_info.free / 1024 / 1024 / 1024
        if free_gb >= min_free_memory_gb:
            available.append(i)

    nvmlShutdown()
    return available

def split_json(json_path, num_splits):
    with open(json_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    per_gpu = (total + num_splits - 1) // num_splits  # 向上取整

    split_paths = []
    for i in range(num_splits):
        part = lines[i * per_gpu: (i + 1) * per_gpu]
        if not part:
            break
        tmp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json", encoding="utf-8")
        tmp_file.writelines(part)
        tmp_file.close()
        split_paths.append(tmp_file.name)

    return split_paths

def launch_processes(split_paths, gpu_indices, output_dir):
    processes = []

    for i, (json_file, gpu_id) in enumerate(zip(split_paths, gpu_indices)):
        cmd = [
            "python", "infer_json.py",
            "--json_path", json_file,
            "--output_dir", output_dir,
            "--cuda_idx", str(gpu_id)
        ]
        print(f"🚀 启动 GPU{gpu_id} 任务，输入：{json_file}")
        p = subprocess.Popen(cmd)
        processes.append(p)

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("⚠️ 检测到中断，终止所有子进程...")
        for p in processes:
            p.terminate()
    finally:
        # 清理临时文件
        for path in split_paths:
            os.remove(path)
        print("🧹 所有临时 JSON 子文件已清除")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="总 JSON 输入文件")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出路径")
    parser.add_argument("--min_free_mem", type=int, default=8, help="可用GPU所需最小空闲显存(GB)")
    args = parser.parse_args()

    # 检测可用 GPU
    gpus = get_available_gpus(min_free_memory_gb=args.min_free_mem)
    if not gpus:
        print("❌ 没有满足条件的GPU（空闲显存 ≥ {}GB）".format(args.min_free_mem))
        exit(1)

    print(f"✅ 检测到可用 GPU: {gpus}")

    # 拆分 JSON
    split_paths = split_json(args.json_path, len(gpus))

    # 启动并行任务
    launch_processes(split_paths, gpus, args.output_dir)
