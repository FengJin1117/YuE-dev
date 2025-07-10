<p align="center">
    <picture>
        <source srcset="./assets/logo/黑底.svg" media="(prefers-color-scheme: dark)">
        <img src="./assets/logo/白底.svg" width="40%">
    </picture>
</p>


<br>


This is the batch music generation code for **YuE**.

## 🔧 QuickStart Changes

- Steps 1–2 (environment setup) are the same as in the [YuE repo](https://github.com/multimodal-art-projection/YuE).
- Step 3 has been replaced with a Chinese example.
- Step 4 supports batch generation by specifying a JSON file and target GPU.
  
Here are the detailed steps——

## 🐧 Linux Quickstart
For a **quick start**, watch this **video tutorial** by Fahd: [Watch here](https://www.youtube.com/watch?v=RSMNH9GitbA).  
If you're new to **machine learning** or the **command line**, we highly recommend watching this video first.  

### 1. Install environment and dependencies
Make sure properly install flash attention 2 to reduce VRAM usage. 
```bash
# We recommend using conda to create a new environment.
conda create -n yue python=3.8 # Python >=3.8 is recommended.
conda activate yue
# install cuda >= 11.8
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
pip install -r <(curl -sSL https://raw.githubusercontent.com/multimodal-art-projection/YuE/main/requirements.txt)

# For saving GPU memory, FlashAttention 2 is mandatory. 
# Without it, long audio may lead to out-of-memory (OOM) errors.
# Be careful about matching the cuda version and flash-attn version
pip install flash-attn --no-build-isolation
```

### 2. Download the infer code and tokenizer
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
# if you don't have root, see https://github.com/git-lfs/git-lfs/issues/4134#issuecomment-1635204943
sudo apt update
sudo apt install git-lfs
git lfs install
git clone https://github.com/multimodal-art-projection/YuE.git

cd YuE-dev/inference/
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
```

### 3. Run the inference
Now generate music with **YuE** using 🤗 Transformers. Make sure your step [1](#1-install-environment-and-dependencies) and [2](#2-download-the-infer-code-and-tokenizer) are properly set up. 

Note:
- Set `--run_n_segments` to the number of lyric sections if you want to generate a full song. 
- You may customize the prompt in `genre.txt` and `lyrics.txt`. 
- LM ckpts will be automatically downloaded from huggingface. 

```bash
# This is the CoT mode.
cd YuE-dev/inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 64 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1
```


### 4. Batch inference

Note:
- `--json_path`: Path to the input JSON file containing lyrics and genre prompt
- The result are saved to `YuE-dev/output_batch` by default. You can customize the location using `--output_dir`.
  
The test command is as follows:
```
# Run inference on a single song.
cd YuE-dev/inference/
nohup python -u infer_json.py \
        --json_path ../lyrics/lyrics_test.jsonl \
        --cuda_idx 0 \
        > ../logs/lyrics_test.log 2>&1 &
```
`10000_lyrics.jsonl` contains a total of 10,000 lyrics.
It is recommended to split `10000_lyrics.jsonl` into smaller chunks and distribute the workload across multiple GPUs for parallel processing.
```
cd YuE-dev/inference/
nohup python -u infer_json.py \
        --json_path ../lyrics/10000_lyrics.jsonl \
        --cuda_idx 0 \
        > ../logs/10000_lyrics.log 2>&1 &
```

---


<br>
