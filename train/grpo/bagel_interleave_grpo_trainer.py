from typing import Any, Callable, Optional, Union
from peft import PeftConfig
import torch.nn.functional as F
import torch
import torch.utils.data
from datasets import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
import time

from trl.trl.models import unwrap_model_for_generation
from trl.trl.trainer import GRPOTrainer
from trl.trl.trainer.grpo_config import GRPOConfig
from trl.trl.trainer.grpo_trainer import RewardFunc
from accelerate.utils import gather
from trl.trl.extras.profiling import profiling_context, profiling_decorator
import numpy as np
import warnings
from inferencer import InterleaveInferencer
import os
import uuid
import json
from PIL import Image
from typing import List, Union
import re
import safetensors.torch as st
from typing import Optional, Union, List, Mapping, Dict


def save_list_with_images(
    input_list: List[Union[str, Image.Image]],
    target_dir: str,
    raw_image: Optional[Image.Image] = None,
    step: Optional[int] = None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    rewards: Optional[List[float]] = None,
    reward_func_names: Optional[List[str]] = None,
    data_idx: Optional[str] = None,
) -> str:
    """
    Process a list containing strings and PIL images, save images to a unique folder, and generate a JSON record.
    Additional features:
      - JSON includes `answer` and `rewards`
      - If step is provided, the saved image filenames include the step number (e.g., step000123_1.png)

    Args:
        input_list: Mixed list containing strings and PIL.Image objects
        target_dir: Target root directory path
        step: Current training step (optional; if provided, will be written as image filename prefix)
        answer: Answer string (optional; if not provided, will attempt to extract <answer>...</answer> from input_list)
        rewards: Reward list (optional; will be directly written to JSON)

    Returns:
        Path to the created folder
    """
    # 1) Create unique folder
    folder_name = f"step{step}_{data_idx}_session_{uuid.uuid4().hex}"
    folder_path = os.path.join(target_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # 2) Process items: save images and replace with relative paths; also try to extract <answer>...</answer>
    processed_list: List[Union[str, str]] = []
    image_counter = 1

    # If answer was not explicitly provided, try to extract from text
    answer_pred = ""
    ans_pat = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
    last_text = input_list[-1] if isinstance(input_list[-1], str) else ""
    if isinstance(last_text, str) and ans_pat.search(last_text):
        answer_pred = ans_pat.search(last_text).group(1)

    raw_image.save(os.path.join(folder_path, "raw_image.png"), "PNG")
    for item in input_list:
        if isinstance(item, dict):
            continue  # Skip latent dict, it is only used for reward computation
        elif isinstance(item, Image.Image):
            # Generate filename (with step prefix)
            if step is not None:
                filename = f"{image_counter}.png"
            else:
                filename = f"{image_counter}.png"
            filepath = os.path.join(folder_path, filename)
            item.save(filepath, "PNG")
            processed_list.append(filename)  # Save relative name only
            image_counter += 1
        else:
            processed_list.append(item)

    # 3) Write out JSON
    json_obj = {
        "question": question,
        "items": processed_list,  # Backward compatible: original list
        "answer": answer,
        "answer_pred": answer_pred,
        "reward_func_names": reward_func_names,
        "rewards": rewards if rewards is not None else [],  # Default empty list
    }
    json_path = os.path.join(folder_path, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=4)

    return folder_path


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean(
        (tensor - torch.nanmean(tensor, keepdim=True)) ** 2
    )  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def _to_bf16_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert all floating-point tensors in state_dict to CPU bfloat16;
    other dtypes (int/bool/uint/long etc.) are only moved to CPU without changing dtype.
    """
    out = {}
    for name, tensor in state_dict.items():
        t = tensor.detach()
        if torch.is_floating_point(t):
            # Move to CPU and convert to bfloat16 in one step; copy=True avoids modifying original tensor
            t = t.to(dtype=torch.bfloat16, device="cpu", copy=True)
        else:
            # Non-floating-point: keep dtype, only move to CPU
            t = t.to(device="cpu", copy=True)
        out[name] = t.contiguous()
    return out


def shuffle_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]],
) -> dict[str, Optional[torch.Tensor]]:
    """
    Shuffles a dictionary of tensors along the first dimension in unison.

    Example:
    ```python
    >>> x = torch.arange(6).reshape(3, 2)
    >>> y = torch.arange(3).reshape(3, 1)
    >>> tensor_dict = {"x": x, "y": y}
    >>> shuffle_tensor_dict(tensor_dict)
    {'x': tensor([[2, 3],
                    [0, 1],
                    [4, 5]]),
        'y': tensor([[1],
                    [0],
                    [2]])}
    ```
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    batch_size = first_tensor.shape[0]
    permutation = torch.randperm(batch_size)
    return {
        key: tensor[permutation] if tensor is not None else None
        for key, tensor in tensor_dict.items()
    }


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
    ```python
    >>> x = torch.arange(12).reshape(6, 2)
    >>> y = torch.arange(6).reshape(6, 1)
    >>> tensor_dict = {"x": x, "y": y}
    >>> split_tensor_dict(tensor_dict, 3)
    [
        {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
        {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
        {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
    ]
    ```
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: (
                tensor[i * chunk_size : (i + 1) * chunk_size]
                if tensor is not None
                else None
            )
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def shuffle_and_split_tensor_dict(
    tensor_dict: Mapping[str, Optional[Union[torch.Tensor, List]]],
):
    # Get the first non-None value to determine batch size and device
    first_val = tensor_dict["advantages"]
    if isinstance(first_val, torch.Tensor):
        batch = first_val.shape[0]
        device = first_val.device
        perm = torch.randperm(batch, device=device)
        perm_list = perm.tolist()
    else:  # list
        batch = len(first_val)
        perm = torch.randperm(batch)  # CPU is fine
        perm_list = perm.tolist()

    # Synchronized shuffle: Tensor uses tensor indexing; List uses reordered list
    shuffled: Dict[str, Optional[Union[torch.Tensor, List]]] = {}
    for k, v in tensor_dict.items():
        if v is None:
            shuffled[k] = None
        elif isinstance(v, torch.Tensor):
            shuffled[k] = v[perm]
        elif isinstance(v, list):
            if len(v) != batch:
                raise ValueError(f"Key {k} has length {len(v)} != {batch}")
            shuffled[k] = [v[j] for j in perm_list]
        else:
            raise TypeError(f"Unsupported value type for key {k}: {type(v)}")

    # Split into single-sample chunks (typically per_device_batch=1)
    out = []
    for i in range(batch):
        out.append(
            {
                k: (
                    v[i : i + 1]
                    if isinstance(v, torch.Tensor)
                    else [v[i]] if isinstance(v, list) else None
                )
                for k, v in shuffled.items()
            }
        )
    return out  # Default chunk_size is 1, since per_device_batch is typically 1


def selective_log_softmax(logits_list, index_list):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    per_token_logps_list = []
    for logits, index in zip(logits_list, index_list):
        assert (
            logits.shape[0] == index.shape[0]
        ), f"logits shape[0]: {logits.shape[0]} mismatch index shape[0]: {index.shape[0]}"
        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(
                logits, dim=-1, index=index.unsqueeze(-1)
            ).squeeze(-1)
            # loop to reduce peak mem consumption
            logsumexp_values = torch.stack(
                [torch.logsumexp(lg, dim=-1) for lg in logits]
            )
            per_token_logps = (
                selected_logits - logsumexp_values
            )  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
            per_token_logps = []
            for row_logits, row_labels in zip(
                logits, index
            ):  # loop to reduce peak mem consumption
                row_logps = F.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(
                    dim=-1, index=row_labels.unsqueeze(-1)
                ).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        per_token_logps_list.append(per_token_logps.unsqueeze(0))

    return per_token_logps_list


@torch.no_grad()
def average_entropy_from_logits_list(logits_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute the average token-level entropy of logits_list (averaged over all samples and timesteps), in nats.
    Each logits has shape [T, V] (T: sequence length, V: vocabulary size).
    Returns a scalar tensor.
    """
    if len(logits_list) == 0:
        return torch.tensor(0.0)

    total_H = torch.tensor(0.0, device=logits_list[0].device)
    total_T = 0

    for logits in logits_list:  # logits: [T, V]
        if logits.dtype in (torch.float32, torch.float64):
            # Numerically stable: H_t = logsumexp(z_t) - sum_i softmax(z_t)_i * z_{t,i}
            lse = torch.logsumexp(logits, dim=-1)  # [T]
            logp = logits - lse.unsqueeze(-1)  # [T, V]
            H = -(logp.exp() * logp).sum(dim=-1)  # [T]
        else:
            # bf16/fp16: convert row-by-row to fp32 for better numerical stability
            per_row = []
            for row in logits:  # [V]
                row_logp = F.log_softmax(row.float(), dim=-1)  # [V]
                per_row.append(-(row_logp.exp() * row_logp).sum())  # scalar
            H = torch.stack(per_row).to(logits.device)  # [T]

        total_H += H.sum()
        total_T += H.numel()

    return total_H / max(total_T, 1)


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class BagelInterleaveGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        vae_transform,
        vit_transform,
        new_token_ids,
        vae_model,
        output_transfer,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        super().__init__(
            model,
            reward_funcs,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            reward_processing_classes,
            callbacks,
            optimizers,
            peft_config,
        )
        self.output_transfer = output_transfer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.vae_model = vae_model.to(self.accelerator.device)
        self.vae_model.eval()
        self.vae_model.requires_grad_(False)
        # print(f"max_grad_norm: {args.max_grad_norm}")

        # Flow-GRPO trajectory selection counter for round_robin strategy
        self._trajectory_step_counter = 0
        # Reward function names for indexing
        # Handle various types of callable: function, partial, lambda, etc.
        def get_func_name(f):
            if hasattr(f, '__name__'):
                return f.__name__
            elif hasattr(f, 'func'):
                return f.func.__name__
            elif hasattr(f, '__class__') and hasattr(f, '__call__'):
                return f.__class__.__name__
            else:
                return str(f)

        self.reward_func_names = [get_func_name(f) for f in reward_funcs]

    def _generate_and_score_completions(
            self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
        ) -> dict[str, Union[torch.Tensor, Any]]:
        # Get current device (CPU/GPU)
        device = self.accelerator.device
        # Determine current mode: training or evaluation
        mode = "eval" if self.control.should_evaluate else "train"
        # Initialize a list for storing processed prompt token IDs
        input_list = []
        id_list = []

        for example in inputs:
            input_list.append(
                [example["image"], example["question"], example["data_id"]]
            )
            id_list.append(example["data_id"])

        start_time = time.time()
        # --- Completion Generation ---
        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as bare_model:
            bare_model.eval()

            # ---------- 2. Build InterleaveInferencer -----------
            inferencer = InterleaveInferencer(
                model=bare_model,
                vae_model=self.vae_model,
                tokenizer=self.processing_class,
                vae_transform=self.vae_transform,
                vit_transform=self.vit_transform,
                new_token_ids=self.new_token_ids,
                device=device,
            )

            # ---------- 4. Inference (batch or per-sample) ------------
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                input_dict_list = []
                output_list = []
                is_eos = []
                completions_tokens = []
                sequence_length = []
                
                # [Modification 2] Initialize image_restore count list
                image_restore_num_list = []
                all_trajectories = []  # [New] Store image generation trajectories

                # [Modification 3] Define target token and answer regex
                restore_token = "<image_restore>"
                result_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"

                for example in input_list:
                    output, trajectories = inferencer.interleave_reason_tool_condition(
                        example[:2],
                        do_sample=True,
                        text_temperature=self.args.temperature,
                        timestep_shift=self.args.timestep_shift,
                        num_timesteps=self.args.num_timesteps,  # ODE steps (50) for quality
                        max_think_token_n=self.args.max_think_token_n,
                        top_p=self.args.top_p,
                        image_shapes=example[0].size[::-1],
                        output_need_vae=self.args.output_need_vae,
                        output_need_vit=self.args.output_need_vit,
                        sde_sigma=self.args.sde_sigma if self.args.use_flow_grpo else 0.0,
                        record_trajectory=self.args.use_flow_grpo,
                        num_timesteps_sde=self.args.num_timesteps_train,  # SDE steps (10) for trajectory
                    )  # Returns Tuple[List[Union[str, Image]], List[Dict]]
                    output_list.append(output)
                    if self.args.use_flow_grpo:
                        all_trajectories.append(trajectories)
                    # Use search instead of fullmatch: when the model takes the image_restore path,
                    # output[-1] is the second gen_text segment (does not start with <think>),
                    # fullmatch requires the entire string to match, which would cause all image_restore samples
                    # to be incorrectly marked as is_eos=False. When mask_truncated_completions=True,
                    # all loss for these samples would be zeroed out, completely preventing them from participating in training.
                    # Using search: as long as the string contains a complete <answer>...</answer>, it is considered finished.
                    match = re.search(r"<answer>.*?</answer>", output[-1], re.DOTALL)
                    if match:
                        is_eos.append(True)
                    else:
                        is_eos.append(False)

                    # [Modification 4] Count occurrences of <image_restore>
                    restore_num = 0
                    # Start from the 3rd element (skip the prompt part)
                    for i in range(3, len(output)):
                        item = output[i]
                        if isinstance(item, str):
                            # Count occurrences of the token in the string directly, no regex needed
                            restore_num += item.count(restore_token)
                    
                    image_restore_num_list.append(restore_num)

                input_dict_list = self.output_transfer(
                    output_list, device, id_list
                )  # Convert output to dict{str: tensor}

                for i in range(len(input_dict_list)):
                    completions_tokens.append(input_dict_list[i]["completions_tokens"])
                    sequence_length.append(input_dict_list[i]["sequence_length"])
                completions_tokens = torch.tensor(completions_tokens).to(device)
                sequence_length = torch.tensor(sequence_length).to(device)
                is_eos = torch.tensor(is_eos).to(device)
                
                # [Modification 5] Convert statistics to Tensor
                image_restore_nums = torch.tensor(image_restore_num_list).to(device)

        # print(f"interleave_reason_tool_condition time: {time.time() - start_time}")

        # --- Compute Log Probabilities (for KL divergence or ratio) ---
        # Disable gradient computation, since we only care about log probabilities
        #
        # When num_iterations > 1, old_per_token_logps is needed for importance sampling ratio (PPO clip);
        # When beta > 0, ref_per_token_logps is needed for KL penalty.
        # Both can use the current model's (pre-update) logps, requiring only one computation.
        with torch.no_grad():
            need_logps = self.num_iterations > 1 or self.beta != 0.0
            if need_logps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, input_dict_list
                )
            else:
                old_per_token_logps = None
            # KL constraint: use generation-time model logps as reference (no extra ref_model, zero memory overhead)
            ref_per_token_logps = old_per_token_logps if self.beta != 0.0 else None

        start_time = time.time()
        # --- Compute Rewards ---
        # Initialize a tensor to store scores for each prompt-completion pair under each reward function
        rewards_per_func = torch.zeros(
            len(output_list), len(self.reward_funcs), device=device
        )
        # Iterate over all reward functions
        for i, (reward_func, reward_func_name) in enumerate(
            zip(
                self.reward_funcs,  # Reward function list
                self.reward_func_names,  # Reward function name list
            )
        ):
            # Use profiling_context to record the time spent in this reward function
            with profiling_context(self, reward_func_name):
                # If the reward function is a callable Python function
                # Extract input fields other than prompt and completion
                keys = [key for key in inputs[0] if key not in ["image"]]
                reward_kwargs = {
                    key: [example[key] for example in inputs] for key in keys
                }
                # Call the reward function
                output_reward_func = reward_func(
                    completions=output_list, **reward_kwargs
                )
                # Replace None values with NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan
                    for reward in output_reward_func
                ]
                # Store results in rewards_per_func
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )
        # print(f"reward_funcs time: {time.time() - start_time}")

        # --- Check and Warn ---
        # Check if any row has all reward functions returning NaN (i.e., None)
        if torch.isnan(rewards_per_func).all(dim=1).any():
            # Get the index of the first all-NaN row
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            # Build detailed info for the row for warning
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["question"] = inputs[nan_row_idx]["question"]
            row_reward_kwargs["completion"] = output[nan_row_idx]
            # Issue warning
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # --- Aggregate and Normalize Rewards ---
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        # In multi-GPU/multi-process setup, gather reward scores from all processes
        rewards_per_func = gather(rewards_per_func)
        # Apply weights to each reward function's output and sum
        # Answer-accuracy-centric reward aggregation:
        # - No longer uses no_image_penalty (decision_reward independently handles generation decisions)
        # - NaN values (e.g., latent_quality when no image was generated) are replaced with 0, no extra penalty
        weights = self.reward_weights.to(device).unsqueeze(0)           # (1, num_funcs)
        weighted = rewards_per_func * weights                           # (N, num_funcs)
        weighted = weighted.nan_to_num(0.0)

        # Compute total rewards
        rewards = weighted.sum(dim=1)  # (N,)

        # Compute grouped-wise rewards (compute per-group mean and std)
        # Reshape rewards to (num_unique_prompts, num_generations_per_prompt)
        # Then compute per-group mean and std
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        # Expand per-group mean and std back to the original batch size
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        # Compute advantages: current reward - group mean
        advantages = rewards - mean_grouped_rewards
        # Optional: scale advantages (divide by per-group std)
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)  # Add small value to prevent division by zero

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )
        advantages = advantages[process_slice]

        # === Compute separate image advantages if enabled ===
        image_advantages = None
        if self.args.separate_image_rewards:
            # Use only latent_quality reward for image advantage calculation
            # Find the index of latent_quality in reward_funcs
            latent_quality_idx = -1
            for i, func_name in enumerate(self.reward_func_names):
                if "latent_quality" in func_name.lower():
                    latent_quality_idx = i
                    break

            if latent_quality_idx >= 0:
                # Get latent_quality rewards (rewards_per_func is already gathered at line 619)
                latent_quality_rewards = rewards_per_func[:, latent_quality_idx]
                # Replace NaN with 0.0 before computing advantages (for samples without generated images)
                # This prevents NaN from propagating to image_advantages and image_loss
                latent_quality_rewards = latent_quality_rewards.nan_to_num(0.0)
                # Compute advantages based on latent_quality only
                # For image advantages, use group-based normalization too
                # No need to gather again — rewards_per_func was already gathered above
                # Reshape to (num_unique_prompts, num_generations)
                latent_quality_rewards_reshaped = latent_quality_rewards.view(-1, self.num_generations)
                mean_latent_quality = latent_quality_rewards_reshaped.mean(dim=1)
                std_latent_quality = latent_quality_rewards_reshaped.std(dim=1)
                mean_latent_quality = mean_latent_quality.repeat_interleave(
                    self.num_generations, dim=0
                )
                std_latent_quality = std_latent_quality.repeat_interleave(
                    self.num_generations, dim=0
                )
                image_advantages = latent_quality_rewards - mean_latent_quality
                if self.scale_rewards:
                    image_advantages = image_advantages / (std_latent_quality + 1e-4)
                # Slice to keep only the local part of the data
                process_slice_advantages = slice(
                    self.accelerator.process_index * len(inputs),
                    (self.accelerator.process_index + 1) * len(inputs),
                )
                image_advantages = image_advantages[process_slice_advantages]

        # --- Log Metrics ---
        # Record token count
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather_for_metrics(sequence_length.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        # Record completion length statistics
        completions_tokens = self.accelerator.gather_for_metrics(completions_tokens)
        self._metrics[mode]["completions/mean_length"].append(
            completions_tokens.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            completions_tokens.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            completions_tokens.float().max().item()
        )

        # identify sequences that terminated with EOS and log their lengths
        # Record EOS-terminated sequence length statistics
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos)
        self._metrics[mode]["completions/clipped_ratio"].append(
            1 - agg_terminated_with_eos.float().mean().item()
        )

        # [Modification 6] Record image_restore statistics
        image_restore_nums = self.accelerator.gather_for_metrics(image_restore_nums)
        self._metrics[mode]["image_restore_nums"].append(
            image_restore_nums.float().mean().item()
        )
        
        # Update total_gen_nums to include only restore count
        self._metrics[mode]["total_gen_nums"].append(
            image_restore_nums.float().mean().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        # Record mean and std of each reward function
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)

        # Record overall reward mean and std
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts (for debugging or visualization)
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        # --- Return Results ---
        # Return a dict containing key tensors from the generation and scoring process
        return {
            "input_dict_list": input_dict_list,  # Input data
            "advantages": advantages,  # Computed advantage values
            "image_advantages": image_advantages,  # Separate image advantage values (if enabled)
            "is_eos": is_eos,  # Whether terminated with EOS
            "old_per_token_logps": old_per_token_logps,  # Old log probabilities
            "ref_per_token_logps": ref_per_token_logps,  # Reference model log probabilities
            # Flow-GRPO: trajectories for per-step backward in _compute_loss
            "image_trajectories": all_trajectories if self.args.use_flow_grpo else None,
        }

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_dict_list, get_entropy=False):
        logits_list = []
        ce_loss_text_ids_list = []

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.train()
            for i, input_dict in enumerate(input_dict_list):
                if input_dict.get("padded_latent") is None:
                    input_dict["padded_latent"] = self.vae_model.encode(
                        input_dict["padded_images"]
                    )
                logits = model.forward_logits(**input_dict)
                # Divide logits by sampling temperature.
                # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
                logits = logits / self.temperature
                logits_list.append(logits)
                ce_loss_text_ids_list.append(input_dict["ce_loss_text_ids"])
        if get_entropy:
            return (
                selective_log_softmax(logits_list, ce_loss_text_ids_list),
                average_entropy_from_logits_list(logits_list),
            )
        return selective_log_softmax(logits_list, ce_loss_text_ids_list)

    def _build_override_for_step(self, model, input_dict, num_gen_images, img_idx, step, device):
        """Build override_packed_latent and override_packed_timesteps for one denoising step."""
        p = model.latent_patch_size
        latent_channel = model.latent_channel
        vae_shapes = input_dict["patchified_vae_latent_shapes"]
        padded_latent = input_dict["padded_latent"]

        num_input_images = len(vae_shapes) - num_gen_images

        override_parts = []
        override_t_parts = []

        for vae_idx, (h, w) in enumerate(vae_shapes):
            n_tokens = h * w

            if vae_idx < num_input_images:
                # Input image: patchify clean latent, timestep=0
                latent = padded_latent[vae_idx]
                latent = latent[:, :h*p, :w*p].reshape(latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p*p*latent_channel)
                override_parts.append(latent.to(device))
                override_t_parts.append(torch.zeros(n_tokens, device=device))
            elif vae_idx - num_input_images == img_idx:
                # Current training target: use x_t from trajectory
                x_t = step['x_t'].to(device)
                override_parts.append(x_t)
                override_t_parts.append(torch.full((n_tokens,), step['timestep'], device=device))
            else:
                # Other generated images: zero-fill, timestep=0
                override_parts.append(torch.zeros(n_tokens, p*p*latent_channel, device=device))
                override_t_parts.append(torch.zeros(n_tokens, device=device))

        return torch.cat(override_parts, dim=0), torch.cat(override_t_parts, dim=0)

    def _build_clean_override(self, model, input_dict, device):
        """Build override that replicates normal t=0 forward for all images.

        Used as a dummy override for samples without trajectory, ensuring
        ZeRO-3 code path symmetry while producing identical hidden states
        to the normal (non-override) forward path.

        Normal forward with packed_timesteps=-inf:
            sigmoid(-inf) = 0  →  shifted t = 0  →  x_t = clean latent
        This override explicitly provides the same patchified clean latent
        with timestep=0, yielding the same vae2llm + time_embedder output.
        """
        p = model.latent_patch_size
        latent_channel = model.latent_channel
        vae_shapes = input_dict["patchified_vae_latent_shapes"]
        padded_latent = input_dict["padded_latent"]

        override_parts = []
        override_t_parts = []

        for vae_idx, (h, w) in enumerate(vae_shapes):
            n_tokens = h * w
            latent = padded_latent[vae_idx]
            latent = latent[:, :h * p, :w * p].reshape(latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * latent_channel)
            override_parts.append(latent.to(device))
            override_t_parts.append(torch.zeros(n_tokens, device=device))

        return torch.cat(override_parts, dim=0), torch.cat(override_t_parts, dim=0)

    def _extract_gen_vpred(self, v_pred_all, input_dict, num_gen_images, img_idx):
        """Extract v_pred slice for the img_idx-th generated image from v_pred_all."""
        vae_shapes = input_dict["patchified_vae_latent_shapes"]
        num_input_images = len(vae_shapes) - num_gen_images

        offset = 0
        for vae_idx, (h, w) in enumerate(vae_shapes):
            n_tokens = h * w
            if vae_idx == num_input_images + img_idx:
                return v_pred_all[offset:offset + n_tokens]
            offset += n_tokens
        raise ValueError(f"Image index {img_idx} not found in vae_shapes")

    def _compute_loss(self, model, inputs):
        # ================================================================
        # Single-step Unbiased Interleaved Flow-GRPO
        #
        # Core idea: merge the text-logits forward and the image-vpred
        # forward into ONE forward_logits(return_vpred=True) call so that
        # text and image gradients share the same computation graph.
        #
        # OLD: 2 forwards per sample
        #   1) _get_per_token_logps → forward_logits() → text logits
        #   2) forward_logits(return_vpred_only=True) → image v_pred
        #   → text/image gradients disconnected
        #
        # NEW: 1 forward per sample
        #   forward_logits(return_vpred=True, override_packed_latent=x_t*)
        #   → (text logits, image v_pred) from the SAME hidden states
        #   → gradients flow: img_loss → v_pred → attention → text repr
        #                      text_loss → logits → attention → image repr
        #
        # Modification notes:
        # 1. Flow GRPO uses image_advantages (advantage computed solely from latent_quality)
        # 2. Step selection strategy: round_robin/weighted/random
        # ================================================================
        device = self.accelerator.device

        input_dict_list = inputs["input_dict_list"]
        advantages = inputs["advantages"]
        # Separate advantages for image generation if enabled
        image_advantages = inputs.get("image_advantages")

        completion_tokens_text = []
        for input_dict in input_dict_list:
            completion_tokens_text.append(input_dict["completions_tokens_text"])
        completion_tokens_text = torch.tensor(completion_tokens_text).to(device)

        # === Phase 1: Determine trajectory step for Flow-GRPO ===
        has_step = False
        chosen_sample_idx = 0
        chosen_img_idx = 0
        chosen_step = None
        sde_sigma = self.args.sde_sigma if self.args.use_flow_grpo else 0.0
        _img_delta_logps = []
        _img_ratios = []
        _img_v_norms = []
        image_loss = torch.tensor(0.0, device=device, requires_grad=False)
        image_loss_value = 0.0

        if self.args.use_flow_grpo:
            trajectories_list = inputs.get("image_trajectories")

            if trajectories_list is not None:
                candidates = []
                for sample_idx, sample_trajs in enumerate(trajectories_list):
                    if not sample_trajs:
                        continue
                    for img_idx, traj in enumerate(sample_trajs):
                        if not traj:
                            continue
                        steps = [s for s in traj if '_context' not in s]
                        for s in steps:
                            candidates.append((sample_idx, img_idx, s))
                if candidates:
                    # Use the specified trajectory selection strategy
                    strategy = self.args.trajectory_selection_strategy
                    if strategy == "round_robin":
                        # Round-robin: select each step in order
                        chosen_idx = self._trajectory_step_counter % len(candidates)
                        chosen_sample_idx, chosen_img_idx, chosen_step = candidates[chosen_idx]
                        self._trajectory_step_counter += 1
                    elif strategy == "weighted":
                        # Weighted sampling: favor middle steps (t ≈ 0.5)
                        # Middle steps have more signal for learning
                        import random
                        weights = [1.0 / (abs(0.5 - s['timestep']) + 0.1) for _, _, s in candidates]
                        total_weight = sum(weights)
                        weights = [w / total_weight for w in weights]
                        chosen_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
                        chosen_sample_idx, chosen_img_idx, chosen_step = candidates[chosen_idx]
                    else:  # "random" (default)
                        # Pure random sampling
                        import random
                        chosen_sample_idx, chosen_img_idx, chosen_step = random.choice(candidates)
                    has_step = True

        # === Phase 2: Combined forward — text logits + image v_pred ===
        # When use_flow_grpo=True, every rank does exactly ONE forward_logits
        # call with override + return_vpred=True, ensuring:
        #   - ZeRO-3 symmetry (same params touched on all ranks)
        #   - Gradient connectivity (text↔image through shared hidden states)
        #   - One backward for both text_loss and image_loss
        logits_list = []
        ce_loss_text_ids_list = []

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.train()
            for i, input_dict in enumerate(input_dict_list):
                # VAE encode if not already done
                if input_dict.get("padded_latent") is None:
                    input_dict["padded_latent"] = self.vae_model.encode(
                        input_dict["padded_images"]
                    )

                has_vae = input_dict.get("packed_vae_token_indexes") is not None

                if self.args.use_flow_grpo and has_vae:
                    # --- Build override latent ---
                    # Real override for chosen sample: x_{t*} at generated image position
                    # Clean override for others: replicates normal t=0 forward exactly
                    if has_step and i == chosen_sample_idx:
                        num_gen_images = len(trajectories_list[chosen_sample_idx])
                        override_latent, override_timesteps = self._build_override_for_step(
                            model, input_dict, num_gen_images,
                            chosen_img_idx, chosen_step, device
                        )
                    else:
                        override_latent, override_timesteps = self._build_clean_override(
                            model, input_dict, device
                        )

                    # --- Single combined forward: both logits AND v_pred ---
                    logits, v_pred_all = model.forward_logits(
                        **input_dict,
                        override_packed_latent=override_latent,
                        override_packed_timesteps=override_timesteps,
                        return_vpred=True,
                    )

                    # Anchor: value=0 but keeps vae2llm/llm2vae/time_embedder in the
                    # backward graph on ALL ranks, preventing ZeRO-3 reduce-scatter deadlock.
                    anchor = (v_pred_all * 0).sum()

                    # --- Compute image Flow-GRPO loss from the SAME forward ---
                    if has_step and i == chosen_sample_idx:
                        v_pred_img = self._extract_gen_vpred(
                            v_pred_all, input_dict,
                            len(trajectories_list[chosen_sample_idx]), chosen_img_idx
                        )

                        x_t = chosen_step['x_t'].to(device)
                        x_next = chosen_step['x_next'].to(device)
                        t_val = chosen_step['timestep']
                        dt_val = chosen_step['dt']

                        # Numerical stability: prevent division by zero and overflow for small t
                        # For very small t, the SDE becomes unstable, so we use a safe threshold
                        t_safe = max(t_val, 1e-4)  # Increased from 1e-6 to 1e-4 for better stability
                        # For very small t, the score can blow up, so we use gradient clipping
                        # Additionally, we use a weighted blend when t is very small
                        t_small_threshold = 0.1
                        if t_val < t_small_threshold:
                            # For small t, use a smoothed formula that reduces to 1/t behavior asymptotically
                            # This avoids division by near-zero values while preserving the gradient signal
                            weight = t_val / t_small_threshold
                            score_neg_standard = (x_t + (1 - t_safe) * v_pred_img) / t_safe
                            score_neg_smoothed = (1 - t_val) * v_pred_img  # No division
                            score_neg = weight * score_neg_smoothed + (1 - weight) * score_neg_standard
                        else:
                            score_neg = (x_t + (1 - t_safe) * v_pred_img) / t_safe

                        # Clip score to prevent numerical instability when t is very small
                        # This prevents the score from exploding when t → 0
                        score_neg = torch.clamp(score_neg, min=-50.0, max=50.0)  # Reduced from 100 to 50

                        # Additional safety: check for NaN or Inf and replace with safe values
                        if torch.isnan(score_neg).any() or torch.isinf(score_neg).any():
                            print(f"[Flow-GRPO] Warning: score_neg contains NaN/Inf, replacing with safe values")
                            score_neg = torch.zeros_like(score_neg)

                        drift = v_pred_img + (sde_sigma ** 2 / 2) * score_neg
                        mu_new = x_t - drift * dt_val
                        variance = sde_sigma ** 2 * dt_val

                        # Compute log probability with per-element mean normalization
                        # Using .mean() instead of .sum():
                        #   Mathematically ratio = exp(log_p_new - log_p_old) is correct as long as both sides are consistent
                        #   But .sum() makes gradient magnitude proportional to D (latent dimension ~65536)
                        #   While text loss is per-token mean, with gradient magnitude proportional to 1
                        #   Using .mean() aligns image gradient magnitude with text, making training more stable
                        sq_error = ((x_next - mu_new) ** 2 / variance)
                        # Clip squared error to prevent overflow/explosion
                        sq_error = torch.clamp(sq_error, min=-1e8, max=1e8)
                        log_p_new = -0.5 * sq_error.mean()

                        if self.num_iterations > 1:
                            log_p_old = torch.tensor(chosen_step['log_prob_old'], device=device)
                        else:
                            log_p_old = log_p_new.detach()

                        delta_logp = log_p_new - log_p_old
                        ratio = torch.exp(delta_logp)

                        # Use separate image advantage if enabled, otherwise use combined advantage
                        if self.args.separate_image_rewards and image_advantages is not None:
                            adv = image_advantages[chosen_sample_idx]
                        else:
                            adv = advantages[chosen_sample_idx]

                        clipped_ratio = torch.clamp(
                            ratio, 1 - self.epsilon_low, 1 + self.epsilon_high
                        )
                        image_loss = -torch.min(ratio * adv, clipped_ratio * adv) + anchor
                        image_loss_value = image_loss.item()

                        _img_delta_logps.append(delta_logp.item())
                        _img_ratios.append(ratio.item())
                        _img_v_norms.append(v_pred_img.detach().norm().item())
                    else:
                        image_loss = image_loss + anchor  # value≈0, but grad-connected

                else:
                    # No Flow-GRPO or no VAE tokens: standard text-only forward
                    logits = model.forward_logits(**input_dict)

                logits = logits / self.temperature
                logits_list.append(logits)
                ce_loss_text_ids_list.append(input_dict["ce_loss_text_ids"])

        # === Phase 3: Text GRPO loss ===
        per_token_logps = selective_log_softmax(logits_list, ce_loss_text_ids_list)
        entropy = average_entropy_from_logits_list(logits_list)
        # per_device_batch_size = 1, so the list has 1 element
        per_token_logps = per_token_logps[0]

        # Mask: only real completion tokens (exclude right-padding)
        max_len = per_token_logps.size(1)
        token_indices = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        mask = (token_indices < completion_tokens_text.unsqueeze(1)).to(
            per_token_logps.dtype
        )  # (B, max_len)

        # KL divergence
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][0]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            per_token_kl = per_token_kl * mask

        # PPO-clip loss
        # num_iterations > 1: use cached logps from generation time (model has been updated, ratio != 1)
        # num_iterations == 1: use detach of current forward (ratio = 1, gradient still flows through per_token_logps)
        old_per_token_logps = (
            inputs["old_per_token_logps"][0]
            if inputs["old_per_token_logps"] is not None
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        if self.mask_truncated_completions:
            mask = mask * inputs["is_eos"].to(per_token_loss.dtype).unsqueeze(1)

        valid_counts = mask.sum(-1).clamp(min=1.0)  # (B,)

        if self.loss_type == "grpo":
            text_loss = ((per_token_loss * mask).sum(-1) / valid_counts).mean()
        elif self.loss_type == "bnpo":
            text_loss = (per_token_loss * mask).sum() / valid_counts.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            text_loss = (per_token_loss * mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Gate text loss with use_text_grpo switch
        if not self.args.use_text_grpo:
            text_loss = torch.tensor(0.0, device=per_token_loss.device)

        # === Phase 4: Combine — ONE backward for both ===
        total_loss = text_loss + self.args.image_loss_weight * image_loss

        # Check for NaN/Inf in losses and log warning if found
        if torch.isnan(text_loss) or torch.isinf(text_loss):
            print(f"[WARNING] text_loss is NaN or Inf: {text_loss.item()}, replacing with 0.0")
            text_loss = torch.tensor(0.0, device=per_token_loss.device, requires_grad=True)

        if torch.isnan(image_loss) or torch.isinf(image_loss):
            print(f"[WARNING] image_loss is NaN or Inf: {image_loss.item()}, replacing with 0.0")
            image_loss = torch.tensor(0.0, device=image_loss.device, requires_grad=True)

        # Recompute total loss if either was replaced
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[CRITICAL] total_loss is NaN or Inf: text_loss={text_loss}, image_loss={image_loss}")
            print(f"Advantages - mean: {advantages.mean()}, std: {advantages.std()}, min: {advantages.min()}, max: {advantages.max()}")
            if image_advantages is not None:
                print(f"Image Advantages - mean: {image_advantages.mean()}, std: {image_advantages.std()}, min: {image_advantages.min()}, max: {image_advantages.max()}")
            # Replace with safe value to prevent training crash - but maintain gradient connection
            total_loss = text_loss + self.args.image_loss_weight * image_loss

        # === Phase 5: Metrics logging (unchanged) ===
        mode = "eval" if self.control.should_evaluate else "train"

        gathered_entropy = self.accelerator.gather_for_metrics(entropy)
        self._metrics[mode]["entropy"].append(gathered_entropy.nanmean().item())

        if self.beta != 0.0:
            mean_kl = (per_token_kl).sum() / valid_counts.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).nanmean().item()
            )

        # Clipped probability ratio stats
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        denom = valid_counts.sum().clamp(min=1.0)
        low_clip = (is_low_clipped.to(mask.dtype) * mask).sum() / denom
        high_clip = (is_high_clipped.to(mask.dtype) * mask).sum() / denom
        clip_ratio = (is_region_clipped.to(mask.dtype) * mask).sum() / denom

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            nanmin(gathered_low_clip).item()
        )
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            nanmax(gathered_high_clip).item()
        )
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item()
        )

        # Image GRPO diagnostics
        if self.args.use_flow_grpo:
            self._metrics[mode].setdefault("image_grpo_loss", []).append(image_loss_value)
            self._metrics[mode].setdefault("image_loss_weight", []).append(
                self.args.image_loss_weight
            )
            if self.args.separate_image_rewards and image_advantages is not None:
                self._metrics[mode].setdefault("image/advantages_mean", []).append(
                    image_advantages.mean().item()
                )
                # Only compute std if there are at least 2 elements to avoid warning
                if image_advantages.numel() > 1:
                    self._metrics[mode].setdefault("image/advantages_std", []).append(
                        image_advantages.std().item()
                    )
                else:
                    self._metrics[mode].setdefault("image/advantages_std", []).append(0.0)
            if _img_delta_logps:
                n = len(_img_delta_logps)
                self._metrics[mode].setdefault("image/delta_logp_mean", []).append(
                    sum(_img_delta_logps) / n
                )
                self._metrics[mode].setdefault("image/ratio_mean", []).append(
                    sum(_img_ratios) / n
                )
                self._metrics[mode].setdefault("image/ratio_max", []).append(
                    max(_img_ratios)
                )
                self._metrics[mode].setdefault("image/anchor_loss", []).append(
                    image_loss_value
                )
                self._metrics[mode].setdefault("image/v_norm", []).append(
                    sum(_img_v_norms) / n
                )

        # Return combined loss — framework does ONE backward call
        return total_loss

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"

        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            # Synchronize step counter across all GPUs to ensure consistent regeneration
            # Gather all step values and use the maximum to determine if regeneration is needed
            # This prevents desynchronization issues in multi-GPU training
            current_step = self._step
            if self.accelerator.num_processes > 1:
                # Gather step from all processes
                step_tensor = torch.tensor([current_step], dtype=torch.long, device=self.accelerator.device)
                gathered_steps = self.accelerator.gather(step_tensor)
                # All processes use the maximum step (some processes may be ahead)
                max_step = gathered_steps.max().item()
                # If any process needs regeneration, all processes regenerate together
                should_regenerate = (max_step % generate_every == 0) or self._buffered_inputs is None
            else:
                should_regenerate = (current_step % generate_every == 0) or self._buffered_inputs is None

            if should_regenerate:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                start_time = time.time()
                generation_batch = self._generate_and_score_completions(
                    generation_batch
                )
                self._buffered_inputs = shuffle_and_split_tensor_dict(generation_batch)
                # Synchronize all processes after generation to ensure buffers are ready
                if self.accelerator.num_processes > 1:
                    self.accelerator.wait_for_everyone()
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Override default _save:
        - Regardless of model type, uniformly save only state_dict (safetensors)
        - Convert all floating-point parameters to bfloat16 before saving (saved on CPU)
        - Preserve training_args for reproducibility (can be removed as needed)
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # If state_dict was not passed from above (e.g., normal backend), get one here
        if state_dict is None:
            target = self.deepspeed if self.is_deepspeed_enabled else self.model
            # Deepspeed ZeRO-3 requires stage3_gather_16bit_weights_on_model_save=True to gather weights to a single node
            state_dict = self.accelerator.get_state_dict(target)

        # Uniformly convert to CPU bfloat16 (floating-point only)
        state_dict = _to_bf16_cpu(state_dict)

        # Only rank 0 saves: under DeepSpeed ZeRO-3, only rank 0 has the complete weights;
        # other ranks have empty state_dict, writing would overwrite the correct file and corrupt the checkpoint
        if self.is_world_process_zero():
            # Save only one safetensors weight file
            st.save_file(
                state_dict,
                os.path.join(output_dir, "model.safetensors"),
                metadata={"format": "pt"},  # Optional metadata
            )
            print(f"Saved model checkpoint to {output_dir}.")
        del state_dict
        torch.cuda.empty_cache()
