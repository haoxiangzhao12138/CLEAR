# COOPER

<p align="center">
  <a href="https://arxiv.org/pdf/2512.04563">Paper</a> |
  <a href="https://huggingface.co/Starrrrrry/COOPER">Model</a> |
  <a href="https://huggingface.co/Starrrrrry/COOPER-AMG">COOPER-AMG Model</a> |
  <a href="https://huggingface.co/datasets/Starrrrrry/COOPER_Train_Set">Training Data</a>
</p>

Official implementation of **COOPER**, a unified multimodal large language model for visual spatial intelligence that cooperatively couples perception and reasoning. Built on the [BAGEL](https://github.com/ByteDance-Seed/BAGEL) framework, COOPER endows a single model with intrinsic perception enhancement (e.g., depth estimation and semantic segmentation) and reasoning enhancement via multimodal chain-of-thought. We further extend COOPER with reinforcement learning and a cooperative perception-reasoning reward, enabling the model to adaptively decide when to "perceive" and when to "reason" during inference.

<p align="center">
  <img src="./assets/motivation.png" width="50%" />
</p>

![model](./assets/model.png)

## Key Features

- **SFT + GRPO Training Pipeline**: Full training pipeline including supervised fine-tuning with corruption-aware interleaved reasoning and GRPO reinforcement learning with cooperative perception-reasoning rewards.
- **VLMEvalKit Integration**: One-line evaluation on multimodal benchmarks (MMBench, MMVet, MMStar, MMVP, CV-Bench, RealWorldQA, R-Bench, etc.) with corruption-level variants.
- **Corruption-Aware Reasoning**: The model learns to detect image degradation and decide when to invoke `<image_restore>` for perception enhancement before answering.

---

## Project Structure

```
COOPER/
├── modeling/                # Model architecture (BAGEL + Qwen2 + SigLIP + VAE)
├── data/                    # Dataset classes and configs
│   ├── configs/             # Training dataset YAML configs
│   ├── corruption_datasets_create/  # Data generation scripts
│   └── interleave_datasets/         # Interleaved reasoning dataset classes
├── train/                   # Training code
│   ├── pretrain_unified_corruption.py  # SFT entry point
│   └── grpo/                # GRPO RL training
├── scripts/                 # Launch scripts
├── VLMEvalKit/              # Evaluation framework
├── transformers-4.54.0/     # Vendored HuggingFace Transformers (with custom modifications)
├── trl/                     # Vendored TRL (with custom modifications)
├── inferencer.py            # Inference engine (shared by training and evaluation)
├── prompts.py               # System prompt templates
└── requirements.txt
```

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/zhangzef/COOPER.git
cd COOPER

conda create -n cooper python=3.10 -y
conda activate cooper

pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
pip install -e ./transformers-4.54.0
pip install -e ./trl
```

### 2. Download Checkpoints and Datasets

```bash
# Download pretrained BAGEL-7B-MoT (required)
cd models
huggingface-cli download --resume-download --local-dir-use-symlinks False \
    ByteDance-Seed/BAGEL-7B-MoT --local-dir BAGEL-7B-MoT
cd ..

# (Optional) Download COOPER checkpoint for direct inference
cd models
huggingface-cli download --resume-download --local-dir-use-symlinks False \
    Starrrrrry/COOPER --local-dir COOPER
cd ..

# (Optional) Download COOPER-AMG checkpoint
cd models
huggingface-cli download --resume-download --local-dir-use-symlinks False \
    Starrrrrry/COOPER-AMG --local-dir COOPER-AMG
cd ..

# Download training data
huggingface-cli download --resume-download --repo-type dataset \
    Starrrrrry/COOPER_Train_Set --local-dir datasets

cd datasets
# Merge and extract (use pigz for faster decompression if available)
cat COOPER_Train_Set.tar.gz.part.* | pigz -d | tar xf -
# OR without pigz:
# cat COOPER_Train_Set.tar.gz.part.* | gzip -dc | tar xf -
cd ..
```

---

## Training

### SFT (Supervised Fine-Tuning)

Train the corruption-aware interleaved reasoning model from BAGEL:

```bash
bash scripts/train_mix.sh
```

This runs `train/pretrain_unified_corruption.py` with FSDP on 8 GPUs, using the `corruption_mix.yaml` dataset config which combines interleave-reason and text-reason datasets.

Key parameters (editable in `scripts/train_mix.sh`):
- `--dataset_config_file`: Dataset YAML config (see `data/configs/`)
- `--model_path`: Pretrained BAGEL-7B-MoT path
- `--lr`: Learning rate (default: 1e-5)
- `--total_steps`: Total training steps (default: 600)

### GRPO (Reinforcement Learning)

Train with cooperative perception-reasoning rewards via GRPO:

```bash
bash scripts/train_reason_interleave_grpo.sh
```

This runs `train/grpo/interleave_grpo.py` with DeepSpeed ZeRO-3 on 8 GPUs. The model learns to optimize:
- **Accuracy reward**: LLM-judged answer correctness
- **Format reward**: Adherence to `<think>/<answer>/<image_restore>` format
- **Decision reward**: Whether to invoke image restoration appropriately
- **Latent quality reward**: VAE-based image restoration quality

Key parameters (editable in `scripts/train_reason_interleave_grpo.sh`):
- `--model_param_path`: SFT checkpoint to initialize from
- `--reward_funcs`: Reward functions to enable
- `--learning_rate`: Learning rate (default: 5e-6)
- `--max_steps`: Total training steps (default: 200)
- `--num_generations`: Number of rollout generations per sample (default: 8)

---

## Evaluation

Evaluation uses a customized [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) with COOPER/BAGEL model support and corruption-level benchmark variants.

### Run Evaluation

```bash
cd VLMEvalKit

# Edit the config to set your model path and benchmark
# Available configs: config/bagel_with_judge.json, config/all_benchmark_corruption.json, etc.
bash eval.sh
```

### Config Format

Evaluation configs are JSON files in `VLMEvalKit/config/`. Example:

```json
{
    "model": {
        "COOPER": {
            "class": "BagelInference",
            "model_config_path": "../models/BAGEL-7B-MoT",
            "model_param_path": "../results/<your_checkpoint>",
            "reasoning_mode": "interleave",
            "max_think_token_n": 4096,
            "is_ema": false
        }
    },
    "data": {
        "MMBench_DEV_EN_V11": {
            "class": "ImageMCQDataset",
            "dataset": "MMBench_DEV_EN_V11"
        }
    }
}
```

Key model parameters:
- `reasoning_mode`: `"interleave"` (COOPER full), `"text"` (text-only reasoning), or `"image"` (perception enhancement only)
- `max_think_token_n`: Max thinking tokens (4096 for reasoning, 256 for perception)
- `is_ema`: Whether to load EMA weights

If using LLM-as-judge (e.g., for MMVet), set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

---

## Results

![main_result](./assets/main_result.png)

### Cases

![cases](./assets/cases.png)

![generation_cases](./assets/generation_cases.png)

---

## Citation

```bibtex
@article{zhang2025cooper,
  title={COOPER: A Unified Model for Cooperative Perception and Reasoning in Spatial Intelligence},
  author={Zhang, Zefeng and Hao, Xiangzhao and Tang, Hengzhu and Zhang, Zhenyu and Sheng, Jiawei and Li, Xiaodong and Li, Zhenyang and Gao, Li and Shi, Daiting and Yin, Dawei and others},
  journal={arXiv preprint arXiv:2512.04563},
  year={2025}
}
```
