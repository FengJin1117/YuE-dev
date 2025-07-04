<p align="center">
    <picture>
        <source srcset="./assets/logo/黑底.svg" media="(prefers-color-scheme: dark)">
        <img src="./assets/logo/白底.svg" width="40%">
    </picture>
</p>

<p align="center">
    <a href="https://map-yue.github.io/">Demo 🎶</a> &nbsp;|&nbsp; 📑 <a href="https://arxiv.org/abs/2503.08638">Paper</a>
    <br>
    <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot">YuE-s1-7B-anneal-en-cot 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl">YuE-s1-7B-anneal-en-icl 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-jp-kr-cot">YuE-s1-7B-anneal-jp-kr-cot 🤗</a>
    <br>
    <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-jp-kr-icl">YuE-s1-7B-anneal-jp-kr-icl 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-cot">YuE-s1-7B-anneal-zh-cot 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-icl">YuE-s1-7B-anneal-zh-icl 🤗</a>
    <br>
    <a href="https://huggingface.co/m-a-p/YuE-s2-1B-general">YuE-s2-1B-general 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-upsampler">YuE-upsampler 🤗</a>
</p>

<br>

## 🐧 Linux/WSL Users Quickstart
For a **quick start**, watch this **video tutorial** by Fahd: [Watch here](https://www.youtube.com/watch?v=RSMNH9GitbA).  
If you're new to **machine learning** or the **command line**, we highly recommend watching this video first.  

To use a **GUI/Gradio** interface, check out:  
- [YuE-exllamav2-UI](https://github.com/WrongProtocol/YuE-exllamav2-UI)
- [YuEGP](https://github.com/deepbeepmeep/YuEGP)
- [YuE-Interface](https://github.com/alisson-anjos/YuE-Interface)  

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

cd YuE/inference/
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
```

### 3. Run the inference
Now generate music with **YuE** using 🤗 Transformers. Make sure your step [1](#1-install-environment-and-dependencies) and [2](#2-download-the-infer-code-and-tokenizer) are properly set up. 

Note:
- Set `--run_n_segments` to the number of lyric sections if you want to generate a full song. Additionally, you can increase `--stage2_batch_size` based on your available GPU memory.

- You may customize the prompt in `genre.txt` and `lyrics.txt`. See prompt engineering guide [here](#prompt-engineering-guide).

- You can increase `--stage2_batch_size` to speed up the inference, but be careful for OOM.

- LM ckpts will be automatically downloaded from huggingface. 


```bash
# This is the CoT mode.
cd YuE/inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1
```

We also support music in-context-learning (provide a reference song), there are 2 types: single-track (mix/vocal/instrumental) and dual-track. 

Note: 
- ICL requires a different ckpt, e.g. `m-a-p/YuE-s1-7B-anneal-en-icl`.

- Music ICL generally requires a 30s audio segment. The model will write new songs with similar style of the provided audio, and may improve musicality.

- Dual-track ICL works better in general, requiring both vocal and instrumental tracks.

- For single-track ICL, you can provide a mix, vocal, or instrumental track.

- You can separate the vocal and instrumental tracks using [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) or [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui).

```bash
# This is the dual-track ICL mode.
# To turn on dual-track mode, enable `--use_dual_tracks_prompt`
# and provide `--vocal_track_prompt_path`, `--instrumental_track_prompt_path`, 
# `--prompt_start_time`, and `--prompt_end_time`
# The ref audio is taken from GTZAN test set.
cd YuE/inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1 \
    --use_dual_tracks_prompt \
    --vocal_track_prompt_path ../prompt_egs/pop.00001.Vocals.mp3 \
    --instrumental_track_prompt_path ../prompt_egs/pop.00001.Instrumental.mp3 \
    --prompt_start_time 0 \
    --prompt_end_time 30 
```

```bash
# This is the single-track (mix/vocal/instrumental) ICL mode.
# To turn on single-track ICL, enable `--use_audio_prompt`, 
# and provide `--audio_prompt_path` , `--prompt_start_time`, and `--prompt_end_time`. 
# The ref audio is taken from GTZAN test set.
cd YuE/inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1 \
    --use_audio_prompt \
    --audio_prompt_path ../prompt_egs/pop.00001.mp3 \
    --prompt_start_time 0 \
    --prompt_end_time 30 
```
---
 
## Prompt Engineering Guide
The prompt consists of three parts: genre tags, lyrics, and ref audio.

### Genre Tagging Prompt
1. An example genre tagging prompt can be found [here](prompt_egs/genre.txt).

2. A stable tagging prompt usually consists of five components: genre, instrument, mood, gender, and timbre. All five should be included if possible, separated by space (space delimiter).

3. Although our tags have an open vocabulary, we have provided the top 200 most commonly used [tags](./top_200_tags.json). It is recommended to select tags from this list for more stable results.

3. The order of the tags is flexible. For example, a stable genre tagging prompt might look like: "inspiring female uplifting pop airy vocal electronic bright vocal vocal."

4. Additionally, we have introduced the "Mandarin" and "Cantonese" tags to distinguish between Mandarin and Cantonese, as their lyrics often share similarities.

### Lyrics Prompt
1. An example lyric prompt can be found [here](prompt_egs/lyrics.txt).

2. We support multiple languages, including but not limited to English, Mandarin Chinese, Cantonese, Japanese, and Korean. The default top language distribution during the annealing phase is revealed in [issue 12](https://github.com/multimodal-art-projection/YuE/issues/12#issuecomment-2620845772). A language ID on a specific annealing checkpoint indicates that we have adjusted the mixing ratio to enhance support for that language.

3. The lyrics prompt should be divided into sessions, with structure labels (e.g., [verse], [chorus], [bridge], [outro]) prepended. Each session should be separated by 2 newline character "\n\n".

4. **DONOT** put too many words in a single segment, since each session is around 30s (`--max_new_tokens 3000` by default).

5. We find that [intro] label is less stable, so we recommend starting with [verse] or [chorus].

6. For generating music with no vocal (instrumental only), see [issue 18](https://github.com/multimodal-art-projection/YuE/issues/18).


### Audio Prompt

1. Audio prompt is optional. Providing ref audio for ICL usually increase the good case rate, and result in less diversity since the generated token space is bounded by the ref audio. CoT only (no ref) will result in a more diverse output.

2. We find that dual-track ICL mode gives the best musicality and prompt following. 

3. Use the chorus part of the music as prompt will result in better musicality.

4. Around 30s audio is recommended for ICL.

5. For music continuation, see [YuE-extend by Mozer](https://github.com/Mozer/YuE-extend). Also supports Colab.

---


<br>
