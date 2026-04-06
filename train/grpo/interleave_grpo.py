import os
import re
import torch
from datetime import datetime
from dataclasses import dataclass, field
from train.grpo.bagel_interleave_grpo_trainer import (
    BagelInterleaveGRPOTrainer,
)
from grpo_data_module import GRPODataset
from trl.trl import (
    GRPOConfig,
    ScriptArguments,
    TrlParser,
)
from transformers import set_seed
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from data.data_utils import add_special_tokens
from modeling.qwen2 import Qwen2Tokenizer
from data.transforms import ImageTransform
from data.output_transfer import OutputTransfer, DataConfig
from safetensors.torch import load_file
from accelerate import init_empty_weights
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from copy import deepcopy
from collections import defaultdict
import math
import torch
import torch.nn.functional as F
from PIL import Image
from data.data_utils import patchify

# Global variable for decision reward smooth setting
_decision_reward_smooth = True

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# import torch
# def custom_repr(self):
#     return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

# original_repr = torch.Tensor.__repr__
# torch.Tensor.__repr__ = custom_repr



@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "decision"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'decision', 'latent_quality'"
        },
    )
    enable_reward_accuracy: bool = field(
        default=True,
        metadata={"help": "Enable the accuracy reward function."},
    )
    enable_reward_format: bool = field(
        default=True,
        metadata={"help": "Enable the format reward function."},
    )
    enable_reward_decision: bool = field(
        default=True,
        metadata={"help": "Enable the decision reward function."},
    )
    enable_reward_latent_quality: bool = field(
        default=True,
        metadata={"help": "Enable the latent_quality reward function."},
    )
    jsonl_path: str = field(
        default="",
        metadata={"help": "Path to the JSONL file containing the dataset."},
    )
    image_root: str = field(
        default="",
        metadata={"help": "Root directory containing the images."},
    )


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."},
    )
    model_param_path: str = field(
        default="",
        metadata={"help": "Path of the pretrained BAGEL model."},
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."},
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."},
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."},
    )
    max_latent_size: int = field(
        default=32,
        metadata={
            "help": "Maximum latent grid size (patches per side) for the VAE latent tensor."
        },
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."},
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."},
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={
            "help": "Maximum number of ViT patches along one image side after cropping / resize."
        },
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={
            "help": "Activation function used in the latent-to-text connector MLP."
        },
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={
            "help": "Interpolate positional embeddings when image resolution differs from pre-training."
        },
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={
            "help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."
        },
    )
    vit_rope: bool = field(
        default=False, metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."},
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."},
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={
            "help": "Probability of dropping ViT visual features during training."
        },
    )
    clean_image_root: str = field(
        default="",
        metadata={"help": "Root directory containing the clean/reference images."},
    )
    mse_scale: float = field(
        default=0.5,
        metadata={"help": "Scale factor for MSE-to-reward exponential mapping in latent_quality_reward."},
    )
    latent_reward_mode: str = field(
        default="both",
        metadata={"help": "Feature mode for latent_quality_reward: 'vae', 'vit', or 'both'."},
    )


@dataclass
class GRPOTrainingArguments(GRPOConfig):
    # --- optimization & scheduler ---
    timestep_shift: float = field(
        default=3.0,
        metadata={
            "help": "Shift applied to diffusion timestep indices (for latent prediction)."
        },
    )
    num_timesteps: int = field(
        default=30,
        metadata={
            "help": "Number of timesteps for image generation during inference."
        },
    )
    save_dir: str = field(
        default="",
        metadata={"help": "Output directory where the trained models will be saved."},
    )
    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."},
    )
    freeze_vit: bool = field(
        default=True, metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={
            "help": "Keep VAE weights fixed; only predict latents, don't fine-tune encoder/decoder."
        },
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."},
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={
            "help": "Duplicate initial MoE experts so each has identical initialisation."
        },
    )
    use_flex: bool = field(
        default=False,
        metadata={
            "help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."
        },
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={
            "help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."
        },
    )
    max_think_token_n: int = field(
        default=4096,
        metadata={
            "help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."
        },
    )
    output_need_vae: bool = field(
        default=False,
        metadata={
            "help": "Whether to insert VAE tokens into context after image generation."
        },
    )
    output_need_vit: bool = field(
        default=True,
        metadata={
            "help": "Whether to insert ViT tokens into context after image generation."
        },
    )
    # --- Flow-GRPO parameters ---
    use_text_grpo: bool = field(
        default=True,
        metadata={"help": "Enable text GRPO (PPO-clip) loss. Set False to zero out text loss."},
    )
    use_flow_grpo: bool = field(
        default=True,
        metadata={"help": "Enable Flow-GRPO for image generation optimization."},
    )
    sde_sigma: float = field(
        default=1.0,
        metadata={"help": "Sigma for SDE-based sampling during training."},
    )
    num_timesteps_train: int = field(
        default=30,  # 改为 30，与 num_timesteps 接近
        metadata={
            "help": "Number of timesteps for image generation during training (Denoising Reduction)."
        },
    )
    image_loss_weight: float = field(
        default=0.3,  # 改为 0.3，增加图像权重
        metadata={
            "help": "Weight for image GRPO loss in total loss computation."
        },
    )
    trajectory_selection_strategy: str = field(
        default="round_robin",
        metadata={
            "help": "Strategy for selecting trajectory step in Flow-GRPO: 'random', 'round_robin', or 'weighted'. "
                  "round_robin ensures each step is trained equally; weighted favors middle steps."
        },
    )
    separate_image_rewards: bool = field(
        default=False,
        metadata={
            "help": "If True, compute separate advantages for image generation using only latent_quality reward. "
                  "This can provide more precise gradient signals for image optimization."
        },
    )
    decision_reward_smooth: bool = field(
        default=True,
        metadata={
            "help": "If True, use smoother decision reward values (1.0, 0.9, 0.5, 0.0) instead of "
                  "original extreme values (1.0, 0.8, 0.1, 0.0)."
        },
    )


# ============ Reward 组件 ============
_image_reward_components = {}
# accuracy 结果缓存，供 decision_reward_auto 复用（避免重复调 LLM API）
_accuracy_cache = {"results": None}


def format_reward_v2(completions, **kwargs):
    """
    分级格式奖励（纯格式结构，不评判是否调用工具）：
    - 0.35: 有完整的 <think>...</think>
    - 0.35: 有完整的 <answer>...</answer>
    - 0.3:  think 在 answer 之前（结构正确）
    总分 [0.0, 1.0]
    """
    rewards = []
    for completion in completions:
        parts = completion[3:]  # 跳过 prompt 部分（前3个元素是 system_prompt, image, question）
        # 把所有文本段拼接
        text = " ".join(x for x in parts if isinstance(x, str))
        score = 0.0

        # (1) 有完整的 think 标签
        if re.search(r'<think>.+?</think>', text, re.DOTALL):
            score += 0.35

        # (2) 有完整的 answer 标签
        if re.search(r'<answer>.+?</answer>', text, re.DOTALL):
            score += 0.35

        # (3) think 在 answer 之前（结构完整性）
        think_end = text.find('</think>')
        answer_start = text.find('<answer>')
        if think_end > 0 and answer_start > think_end:
            score += 0.3

        rewards.append(score)
    return rewards



def _extract_answer_text(completion) -> str:
    """从 completion 列表中提取 <answer>...</answer> 中的文本"""
    if not completion:
        return ""
    last_text = completion[-1] if isinstance(completion[-1], str) else ""
    match = re.search(r'<answer>([\s\S]*?)</answer>', last_text)
    if match:
        return match.group(1).strip()
    return last_text.strip()


def decision_reward_auto(completions, solution, question, **kwargs):
    """
    基于结果回溯的策略决策奖励（无需难度标签）。
    复用 accuracy_reward_v2 缓存的 LLM judge 结果来判断是否答对。

    核心设计原则：
    - 训练数据全部是退化图像，模型应该更积极地尝试生成
    - 不生成+答错 给负惩罚，迫使模型在不确定时倾向生成
    - 生成+答错 仍给正奖励，鼓励探索

    逻辑矩阵（原始值，_decision_reward_smooth=False）：
    | 是否恢复 | 是否答对 | 奖励 | 原因 |
    |----------|----------|------|------|
    | 恢复了   | 答对了   | 1.0  | 最优：正确使用了工具 |
    | 没恢复   | 答对了   | 0.6  | 不错，但未必是最优策略 |
    | 恢复了   | 没答对   | 0.4  | 鼓励探索，尝试了就值得 |
    | 没恢复   | 没答对   | -0.2 | 应该生成但没生成，需要惩罚 |

    逻辑矩阵（平滑值，_decision_reward_smooth=True）：
    | 是否恢复 | 是否答对 | 奖励 | 原因 |
    |----------|----------|------|------|
    | 恢复了   | 答对了   | 1.0  | 最优：正确使用了工具 |
    | 没恢复   | 答对了   | 0.5  | 可以，但鼓励更积极生成 |
    | 恢复了   | 没答对   | 0.5  | 中等：尝试了就值得鼓励 |
    | 没恢复   | 没答对   | -0.2 | 最差：应该生成但没生成 |

    注意：reward_funcs 列表中 accuracy 必须排在 decision 前面，
    这样 _accuracy_cache 在本函数执行时已有数据。
    """
    # 使用全局变量 decision_reward_smooth
    use_smooth = _decision_reward_smooth

    # 读取 accuracy_reward_v2 缓存的结果
    acc_results = _accuracy_cache.get("results")
    if acc_results is None:
        # 如果 accuracy 没在前面跑过（不应该发生），fallback 调 LLM
        acc_results = _call_llm_judge(completions, solution, question)

    rewards = []
    for idx, completion in enumerate(completions):
        parts = completion[3:]  # 跳过 prompt
        text_content = " ".join(x for x in parts if isinstance(x, str))
        did_restore = "<image_restore>" in text_content
        correct = acc_results[idx] > 0.5

        if use_smooth:
            # 平滑奖励曲线
            if did_restore and correct:
                rewards.append(1.0)       # 最优
            elif not did_restore and correct:
                rewards.append(0.5)       # 可以，但鼓励更积极
            elif did_restore and not correct:
                rewards.append(0.5)       # 尝试了就值得鼓励
            else:
                rewards.append(-0.2)      # 应该生成但没生成
        else:
            # 原始奖励曲线
            if did_restore and correct:
                rewards.append(1.0)
            elif not did_restore and correct:
                rewards.append(0.6)
            elif did_restore and not correct:
                rewards.append(0.4)
            else:
                rewards.append(-0.2)

    return rewards


def _extract_vit_features(image, vit_model, vit_transform, patch_size,
                           get_position_ids_fn, max_patches, device=None):
    """从单张 PIL Image 提取 ViT pooled feature。"""
    # 如果没有传入 device，自动使用当前 GPU 设备
    if device is None:
        device = torch.cuda.current_device()
    image_tensor = vit_transform(image).to(device)
    position_ids = get_position_ids_fn(
        image_tensor.size(1), image_tensor.size(2),
        patch_size, max_num_patches_per_side=max_patches,
    ).to(device)
    patches = patchify(image_tensor, patch_size).to(device)
    cu_seqlens = torch.tensor([0, patches.shape[0]], dtype=torch.int32, device=device)
    features = vit_model(
        packed_pixel_values=patches,
        packed_flattened_position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        max_seqlen=patches.shape[0],
    )
    return features.mean(dim=0, keepdim=True)  # (1, hidden)


def latent_quality_reward(completions, image_name, **kwargs):
    """
    生成图像质量 reward，支持三种模式（由 latent_reward_mode 控制）：
    - "vae":  VAE latent 空间三指标（r_mse + r_cos + r_local）
    - "vit":  ViT 语义特征 cosine 相似度（r_vit）
    - "both": 四指标综合（r_mse + r_cos + r_local + r_vit）

    没有生成图像的样本返回 NaN（不参与该 reward 聚合）。
    """

    comp = _image_reward_components
    if not comp:
        return [float('nan')] * len(completions)

    mode = comp.get("mode", "vae")
    clean_root = comp["clean_image_root"]

    # VAE 相关组件
    need_vae = mode in ("vae", "both")
    vae_model = None
    if need_vae:
        vae_model = comp["vae_model"]
        vae_transform = comp["vae_transform"]
        latent_patch_size = comp["latent_patch_size"]
        latent_channel = comp["latent_channel"]
        mse_scale = comp.get("mse_scale", 0.5)

    # ViT 相关组件
    need_vit = mode in ("vit", "both")
    vit_model = None
    if need_vit:
        vit_model = comp["vit_model"]
        vit_transform = comp["vit_transform"]
        vit_patch_size = comp["vit_patch_size"]
        vit_max_patches = comp["vit_max_patches"]
        get_position_ids_fn = comp["get_position_ids_fn"]

    # 动态获取正确的设备，避免硬编码 "cuda" 导致多 GPU 死锁
    # 在 DeepSpeed ZeRO-3 下，使用 torch.cuda.current_device() 获取当前 GPU
    device = torch.cuda.current_device()

    rewards = []
    for idx, completion in enumerate(completions):
        # ---- 1. 从 completion 中提取生成的 latent 和 Image ----
        gen_latent = None
        gen_image = None
        for item in completion:
            if isinstance(item, dict) and item.get("type") == "generated_latent":
                gen_latent = item["latent"]
            elif isinstance(item, Image.Image) and gen_latent is not None:
                # 取紧跟 latent dict 之后的第一张 Image（即生成图）
                gen_image = item
                break

        # vit 模式只需要 Image；vae/both 模式需要 latent
        if need_vae and gen_latent is None:
            rewards.append(float('nan'))
            continue
        if need_vit and gen_image is None:
            # 如果 vit 模式没有 Image，但 vae 模式有 latent，也尝试继续
            if not need_vae:
                rewards.append(float('nan'))
                continue

        # ---- 2. 加载清晰参考图像 ----
        fname = image_name[idx]
        clean_path = os.path.join(clean_root, fname)
        if not os.path.exists(clean_path):
            base = os.path.splitext(fname)[0]
            clean_path = None
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                candidate = os.path.join(clean_root, base + ext)
                if os.path.exists(candidate):
                    clean_path = candidate
                    break
        if clean_path is None or not os.path.exists(clean_path):
            print(f"[latent_quality_reward] clean image not found: {fname}")
            rewards.append(float('nan'))
            continue

        try:
            clean_image = Image.open(clean_path).convert("RGB")

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # ---- 3a. VAE latent 子指标 ----
                r_mse = r_cos = r_local = 0.0
                if need_vae and gen_latent is not None:
                    clean_tensor = vae_transform(clean_image).unsqueeze(0).to(device)
                    clean_latent_raw = vae_model.encode(clean_tensor)  # (1, C, H, W)

                    p = latent_patch_size
                    c = latent_channel
                    _, _, h_lat, w_lat = clean_latent_raw.shape
                    h_patch = h_lat // p
                    w_patch = w_lat // p
                    clean_lat = clean_latent_raw[0, :, :h_patch * p, :w_patch * p]
                    clean_lat = clean_lat.reshape(c, h_patch, p, w_patch, p)
                    clean_lat = torch.einsum("chpwq->hwpqc", clean_lat)
                    clean_latent_flat = clean_lat.reshape(-1, p * p * c)

                    gen_latent_flat = gen_latent.to(device)
                    n_tokens = min(gen_latent_flat.shape[0], clean_latent_flat.shape[0])
                    gen_latent_flat = gen_latent_flat[:n_tokens]
                    clean_latent_flat = clean_latent_flat[:n_tokens]

                    mse_val = F.mse_loss(gen_latent_flat, clean_latent_flat).item()
                    r_mse = math.exp(-mse_val * mse_scale)

                    cos_sim = F.cosine_similarity(
                        gen_latent_flat.reshape(1, -1),
                        clean_latent_flat.reshape(1, -1),
                        dim=-1
                    ).item()
                    r_cos = max(cos_sim, 0.0)

                    per_token_cos = F.cosine_similarity(
                        gen_latent_flat, clean_latent_flat, dim=-1
                    )
                    r_local = per_token_cos.clamp(min=0).mean().item()

                # ---- 3b. ViT 语义子指标 ----
                r_vit = 0.0
                if need_vit and gen_image is not None:
                    try:
                        # 确保 vit_model 在 eval 模式，避免 ZeRO-3 同步问题
                        vit_model.eval()
                        gen_feat = _extract_vit_features(
                            gen_image, vit_model, vit_transform,
                            vit_patch_size, get_position_ids_fn,
                            vit_max_patches,
                        )
                        clean_feat = _extract_vit_features(
                            clean_image, vit_model, vit_transform,
                            vit_patch_size, get_position_ids_fn,
                            vit_max_patches,
                        )
                        sim = F.cosine_similarity(gen_feat, clean_feat, dim=-1).item()
                        r_vit = max(sim, 0.0)
                    except Exception as e:
                        print(f"[latent_quality_reward] ViT feature extraction error for {fname}: {e}")
                        import traceback
                        traceback.print_exc()
                        r_vit = 0.0

            # ---- 4. 按模式综合得分 ----
            if mode == "vae":
                score = 0.3 * r_mse + 0.3 * r_cos + 0.4 * r_local
            elif mode == "vit":
                score = r_vit
            else:  # both
                score = 0.2 * r_mse + 0.2 * r_cos + 0.3 * r_local + 0.3 * r_vit

        except Exception as e:
            print(f"[latent_quality_reward] error for {fname}: {e}")
            score = 0.0

        rewards.append(float(score))

    return rewards


def _call_llm_judge(completions, solutions, questions):
    """
    调用 LLM 判断答案正确性。
    从原 accuracy_reward_with_llm 中提取的核心逻辑。
    """
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
    api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    system_prompt = """
    You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.
    Your task is to compare the predicted answer with the correct answer and rate the correctness on a continuous scale. Here's how you can accomplish the task:
    INSTRUCTIONS:
    - Focus on the meaningful match between the predicted answer and the correct answer.
    - Consider synonyms or paraphrases as valid matches.
    - Evaluate the correctness of the prediction compared to the answer.
    - Rate the correctness from 0.0 (completely wrong) to 1.0 (perfectly correct).
    """
    user_prompt_template = """
    I will give you a question related to an image and the following text as inputs:
    1. **Question Related to the Image**: {Question}
    2. **Ground Truth Answer**: {Ground_Truth}
    3. **Model Predicted Answer**: {Prediction}
    Your task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:
    - **Relevance**: Does the predicted answer directly address the question posed, considering the information provided by the given question?
    - **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:
    (1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should receive a high score.
    (2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's prediction should receive a high score.

    **Output Format**:
    Rate the correctness of the prediction on a scale from 0.0 to 1.0, where 0.0 means completely incorrect and 1.0 means perfectly correct. Consider partial correctness for answers that are close but not exact.
    Your response should ONLY contain a single line in the following format:
    Score: <float between 0.0 and 1.0>
    Example: Score: 0.85
    """
    max_retries = 3
    timeout = 30
    max_workers = 64
    model = "gpt-4.1"

    def process_single(prompt: str) -> float:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, max_tokens=512, temperature=0.0, n=1,
                )
                content = response.choices[0].message.content.strip().lower()
                # Parse "Score: 0.85" format
                match = re.search(r'score:\s*([\d.]+)', content)
                if match:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))  # clamp to [0, 1]
                # Fallback: try to parse any float
                match = re.search(r'([\d.]+)', content)
                if match:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
                # Fallback: handle legacy True/False responses
                if "true" in content:
                    return 1.0
                elif "false" in content:
                    return 0.0
            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                print(f"Error: {e}")
                if attempt == max_retries - 1:
                    return 0.0
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"Error: {e}")
                if attempt == max_retries - 1:
                    return 0.0
        return 0.0

    prompts = []
    skip_indices = set()
    answer_list = []
    for idx, comp in enumerate(completions):
        pred = _extract_answer_text(comp)
        if not pred:
            answer_list.append("-")
            skip_indices.add(idx)
        else:
            answer_list.append(pred)

    for i in range(len(answer_list)):
        prompts.append(user_prompt_template.format(
            Question=questions[i],
            Ground_Truth=solutions[i],
            Prediction=answer_list[i],
        ))

    results = [0.0] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single, prompt): idx
            for idx, prompt in enumerate(prompts)
            if idx not in skip_indices
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = float(future.result())
            except Exception:
                results[idx] = 0.0

    return results


def accuracy_reward_v2(completions, solution, question, **kwargs):
    """
    答案正确性奖励。
    全量调用 LLM judge 判断，结果缓存到 _accuracy_cache 供 decision_reward_auto 复用。
    """
    rewards = _call_llm_judge(completions, solution, question)
    # 缓存结果，decision_reward_auto 按顺序在后面执行时直接读取
    _accuracy_cache["results"] = rewards
    # Log API success rate for debugging
    nonzero_count = sum(1 for r in rewards if r > 0)
    print(f"[accuracy_reward] scores: mean={sum(rewards)/len(rewards):.3f}, "
          f"nonzero={nonzero_count}/{len(rewards)}, "
          f"values={[round(r, 2) for r in rewards]}")
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward_v2,
    "format": format_reward_v2,
    "decision": decision_reward_auto,
    "latent_quality": latent_quality_reward,
}


def main(grpo_args, training_args, model_args):
    set_seed(training_args.seed)
    if model_args.model_param_path == "":
        model_args.model_param_path = model_args.model_path

    finetune_from_ema = True
    llm_config = Qwen2Config.from_json_file(
        os.path.join(model_args.model_path, "llm_config.json")
    )
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    language_model = Qwen2ForCausalLM(llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    vit_config = SiglipVisionConfig.from_json_file(
        os.path.join(model_args.model_path, "vit_config.json")
    )
    vit_config.num_hidden_layers = (
        vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
    )
    vit_config.rope = model_args.vit_rope
    vit_model = SiglipVisionModel(vit_config)

    vae_model, vae_config = load_ae(
        local_path=(os.path.join(model_args.model_path, "ae.safetensors"))
    )

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    if training_args.freeze_vae:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # Setup FSDP and load pretrained model:
    # ema_model = deepcopy(model)
    if finetune_from_ema:
        model_state_dict_path = os.path.join(
            model_args.model_param_path, f"ema.safetensors"
        )
    else:
        model_state_dict_path = os.path.join(
            model_args.model_param_path, f"model.safetensors"
        )
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    model_state_dict.pop("latent_pos_embed.pos_embed")
    model_state_dict.pop("vit_pos_embed.pos_embed")
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(f"model load msg: {msg}")
    del model_state_dict

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(518, 224, 14)
    vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
    data_config = DataConfig(
        vae_image_downsample=vae_image_downsample,
        max_latent_size=model_args.max_latent_size,
        vit_patch_size=model_args.vit_patch_size,
        max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
    )
    output_transfer = OutputTransfer(
        tokenizer,
        vae_transform,
        vit_transform,
        data_config,
        training_args.max_num_tokens,
        new_token_ids,
        use_flex=training_args.use_flex,
    )

    # ---- 初始化 reward 组件 ----
    # Build active reward list from switches
    reward_switch_map = {
        "accuracy": grpo_args.enable_reward_accuracy,
        "format": grpo_args.enable_reward_format,
        "decision": grpo_args.enable_reward_decision,
        "latent_quality": grpo_args.enable_reward_latent_quality,
    }
    active_reward_names = [name for name in grpo_args.reward_funcs if reward_switch_map.get(name, True)]

    if "latent_quality" in active_reward_names:
        # 在 ZeRO-3 环境下，ViT 模型参数被分片，调用时会触发 all-gather 导致死锁
        # 解决方案：如果使用 ViT reward，创建一个独立的 ViT 副本
        if model_args.latent_reward_mode in ("vit", "both"):
            # 创建 ViT 的深拷贝，避免 ZeRO-3 分片问题
            vit_model_for_reward = deepcopy(model.vit_model)
            vit_model_for_reward.eval()
            for param in vit_model_for_reward.parameters():
                param.requires_grad = False
            # 移动到当前 GPU
            vit_model_for_reward = vit_model_for_reward.to(torch.cuda.current_device())
            print("[GRPO Config] Created independent ViT copy for reward calculation")
        else:
            vit_model_for_reward = None

        _image_reward_components.update({
            "clean_image_root": model_args.clean_image_root,
            "mode": model_args.latent_reward_mode,
            # VAE 组件
            "vae_model": vae_model,
            "vae_transform": vae_transform,
            "latent_patch_size": model_args.latent_patch_size,
            "latent_channel": vae_config.z_channels,
            "mse_scale": model_args.mse_scale,
            # ViT 组件 - 使用独立副本避免 ZeRO-3 死锁
            "vit_model": vit_model_for_reward,
            "vit_transform": vit_transform,
            "vit_patch_size": model_args.vit_patch_size,
            "vit_max_patches": model_args.vit_max_num_patch_per_side,
            "get_position_ids_fn": model.get_flattened_position_ids,
        })

    # Get reward functions
    reward_funcs = []
    for func in active_reward_names:
        reward_funcs.append(reward_funcs_registry[func])
    # Load the dataset
    train_set = GRPODataset(
        jsonl_path=grpo_args.jsonl_path, image_root=grpo_args.image_root
    )

    # Store decision_reward_smooth in a global variable for decision_reward_auto
    global _decision_reward_smooth
    _decision_reward_smooth = training_args.decision_reward_smooth

    # Log active components
    print(f"[GRPO Config] Active rewards: {active_reward_names}")
    print(f"[GRPO Config] use_text_grpo: {training_args.use_text_grpo}")
    print(f"[GRPO Config] use_flow_grpo: {training_args.use_flow_grpo}")
    print(f"[GRPO Config] trajectory_selection_strategy: {training_args.trajectory_selection_strategy}")
    print(f"[GRPO Config] separate_image_rewards: {training_args.separate_image_rewards}")
    print(f"[GRPO Config] decision_reward_smooth: {training_args.decision_reward_smooth}")
    print(f"[GRPO Config] num_timesteps: {training_args.num_timesteps}")
    print(f"[GRPO Config] num_timesteps_train: {training_args.num_timesteps_train}")
    print(f"[GRPO Config] image_loss_weight: {training_args.image_loss_weight}")

    # Initialize the GRPO trainer
    trainer = BagelInterleaveGRPOTrainer(
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
        vae_model=vae_model,
        output_transfer=output_transfer,
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=None,
        # output_record_file=f"./sample_output/{training_args.run_name}.txt",
    )

    # 设置 reward 权重：以答案正确性为核心，decision 鼓励自适应生成
    weight_map = {
        "accuracy": 0.75,
        "format": 0.1,
        "decision": 0.15,
        "latent_quality": 0.0,  # 保留 key 兼容，但权重为 0（不鼓励照片级重建）
    }
    reward_weights = [weight_map[name] for name in active_reward_names]
    trainer.reward_weights = torch.tensor(reward_weights, dtype=torch.float32)

    # Train and push the model to the Hub
    trainer.train()
    # Save and push to hub
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOTrainingArguments, ModelArguments))
    grpo_args, training_args, model_args = parser.parse_args_and_config()
    main(grpo_args, training_args, model_args)
