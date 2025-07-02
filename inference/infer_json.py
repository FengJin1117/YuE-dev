import os
import json
import shutil
import glob
import time
from pathlib import Path
import subprocess
import argparse


# 检查是否生成了音乐
def song_exists(song_dir):
    mixed_path = os.path.join(song_dir, "vocoder") 
    if os.path.exists(mixed_path): 
        return True 
    else: 
        return False

def yue_infer(song_dir, cuda_idx, silent=False): 
    # song_dir = shlex.quote(song_dir)
    genre_prompt_path = os.path.join(song_dir, "genre_prompt.txt") 
    lyrics_path = os.path.join(song_dir, "lyrics.txt") 
    output_dir = song_dir
    # print("genre prompt:", genre_prompt_path)
    # print("lyrics:", lyrics_path)
    # print("output dir:", output_dir)

    command = [
        "python", "infer.py",
        "--cuda_idx", str(cuda_idx),
        "--stage1_model", "m-a-p/YuE-s1-7B-anneal-zh-cot",
        "--stage2_model", "m-a-p/YuE-s2-1B-general",
        "--genre_txt", genre_prompt_path,
        "--lyrics_txt", lyrics_path,
        "--run_n_segments", "2",
        "--stage2_batch_size", "64",
        "--output_dir", output_dir,
        "--max_new_tokens", "3000",
        "--repetition_penalty", "1.1"
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL if silent else None,
            stderr=subprocess.DEVNULL if silent else None,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Execution failed: {e}")
        return False

    end_time = time.time()
    print(f"Execution Time: {(end_time - start_time) / 60:.2f} minutes")

    
    if song_exists(song_dir): 
        print(f"Succeed") 
        return True 
    else: 
        print(f"Failed") 
        return False
    
def process_json(json_path, output_dir, cuda_idx):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        data = json.loads(line.strip())
        song_id = data["ID"]
        genre_prompt = data["genre prompt"]
        lyrics = data["lyrics"]

        tmp_dir = "tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        # 写入文本文件
        with open(os.path.join(tmp_dir, "genre_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(genre_prompt)
        with open(os.path.join(tmp_dir, "lyrics.txt"), "w", encoding="utf-8") as f:
            f.write(lyrics)

        print(f"\n🔄 开始生成歌曲：{song_id}")
        success = yue_infer(song_dir=tmp_dir, cuda_idx=cuda_idx)

        if not success:
            print(f"❌ 歌曲 {song_id} 生成失败，跳过\n")
            continue

        # 搬运 .wav 文件
        mixed_wav = glob.glob(os.path.join(tmp_dir, "*.wav"))
        if mixed_wav:
            target_wav = os.path.join(output_dir, f"{song_id}_mixed.wav")
            shutil.copy(mixed_wav[0], target_wav)
            print(f"✅ 保存混音文件：{target_wav}")
        else:
            print(f"⚠️ 未找到混音 WAV 文件：{song_id}")

        # 搬运 .mp3 文件
        vocal_mp3 = os.path.join(tmp_dir, "vocoder", "stems", "vtrack.mp3")
        if os.path.exists(vocal_mp3):
            target_mp3 = os.path.join(output_dir, f"{song_id}_vocal.mp3")
            shutil.copy(vocal_mp3, target_mp3)
            print(f"✅ 保存人声文件：{target_mp3}")
        else:
            print(f"⚠️ 未找到人声 MP3 文件：{song_id}")

        # 清理 tmp
        shutil.rmtree(tmp_dir)
        print(f"🧹 清理临时文件夹 tmp 完成\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="输入JSON路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出路径")
    parser.add_argument("--cuda_idx", type=int, default=0, help="使用的GPU编号")
    args = parser.parse_args()

    process_json(args.json_path, args.output_dir, args.cuda_idx)