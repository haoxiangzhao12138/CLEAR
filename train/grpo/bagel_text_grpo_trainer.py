# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# from trl.models import (
#     unwrap_model_for_generation,
# )
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
    # ans_pat = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
    # if ans_pat.search(input_list[-1]):
    #     answer_pred = ans_pat.search(input_list[-1]).group(1)
    ans_pat = re.compile(r"^<think>.*?</think>(.*)$", flags=re.DOTALL)
    m = ans_pat.search(input_list[-1])
    answer_pred = m.group(1) if m else ""

    raw_image.save(os.path.join(folder_path, "raw_image.png"), "PNG")
    for item in input_list:
        if isinstance(item, Image.Image):
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

    # # Find the maximum sequence length across all samples
    # max_len = max(t.shape[0] for t in per_token_logps_list)
    # padded_tensors = []

    # # Pad each tensor to max_len and stack
    # for t in per_token_logps_list:
    #     if t.shape[0] < max_len:
    #         # Pad with zeros on the right (dim=0)
    #         padding = torch.zeros(max_len - t.shape[0], device=t.device, dtype=t.dtype)
    #         padded_t = torch.cat([t, padding], dim=0)
    #     else:
    #         padded_t = t
    #     padded_tensors.append(padded_t)

    # return torch.stack(
    #     padded_tensors, dim=0
    # )  # Shape: (batch_size, max_completion_length)
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


class BagelTextGRPOTrainer(GRPOTrainer):
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
        print(f"max_grad_norm: {args.max_grad_norm}")

    # This is a core method overridden in the BagelGRPOTrainer class
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
            input_list.append([example["image"], example["question"]])
            id_list.append(example["data_id"])

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
                result_pattern = r"<think>.*?</think>.*"
                for example in input_list:
                    output = inferencer.text_reason(
                        example,
                        do_sample=True,
                        text_temperature=self.args.temperature,
                        max_think_token_n=self.args.max_think_token_n,
                        top_p=self.args.top_p,
                    )  # Returns List[Union[str, Image]]
                    output_list.append(output)
                    match = re.fullmatch(result_pattern, output[-1], re.DOTALL)
                    if match:
                        is_eos.append(True)
                    else:
                        is_eos.append(False)

                input_dict_list = self.output_transfer(
                    output_list, device, id_list
                )  # Convert output to dict{str: tensor}
                for i in range(len(input_dict_list)):
                    completions_tokens.append(input_dict_list[i]["completions_tokens"])
                    sequence_length.append(input_dict_list[i]["sequence_length"])
                completions_tokens = torch.tensor(completions_tokens).to(device)
                sequence_length = torch.tensor(sequence_length).to(device)
                is_eos = torch.tensor(is_eos).to(device)

        # --- Compute Log Probabilities (for KL divergence or ratio) ---
        # Disable gradient computation, since we only care about log probabilities
        with torch.no_grad():
            # When num_iterations == 1, old_per_token_logps equals current per_token_logps
            if self.num_iterations > 1:
                # Compute old (previous iteration) per-token log probabilities
                old_per_token_logps = self._get_per_token_logps(
                    self.model, input_dict_list
                )
            else:
                # If only iterating once, old logps are not needed
                old_per_token_logps = None
            # If beta is 0, reference model logps are not needed
            if self.beta == 0.0:
                ref_per_token_logps = None
            # If there is a separate reference model
            elif self.ref_model is not None:
                # Compute reference model per-token log probabilities
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, input_dict_list
                )
            else:
                # Otherwise, disable adapter (e.g., LoRA) on the current model to simulate the reference model
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_dict_list
                    )

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

        # # ========= Insert save logic before distributed gather =========
        # os.makedirs(os.path.join(self.args.save_dir, "samples", mode), exist_ok=True)
        # step_val = getattr(self.state, "global_step", None)
        # if step_val is None:
        #     step_val = getattr(self, "_step", 0)
        # base_dir = os.path.join(self.args.save_dir, "samples", mode)
        # for i in range(len(output_list)):
        #     save_list_with_images(
        #         input_list=output_list[i],
        #         target_dir=base_dir,
        #         raw_image=inputs[i]["image"],
        #         step=step_val,
        #         question=inputs[i]["question"],
        #         answer=inputs[i]["solution"],
        #         rewards=rewards_per_func[i].detach().float().tolist(),
        #         reward_func_names=self.reward_func_names,
        #         data_idx=id_list[i],
        #     )

        # --- Check and Warn ---
        # If all reward functions return None for a given row, issue a detailed warning
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
        # Apply weights to each reward function and sum to get final rewards
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(
            dim=1
        )  # nansum ignores NaN values when summing

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
            "is_eos": is_eos,  # Whether terminated with EOS
            "old_per_token_logps": old_per_token_logps,  # Old log probabilities
            "ref_per_token_logps": ref_per_token_logps,  # Reference model log probabilities
        }

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_dict_list, get_entropy=False):
        logits_list = []
        ce_loss_text_ids_list = []

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.train()
            torch.cuda.empty_cache()
            for input_dict in input_dict_list:
                # with torch.no_grad():
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

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        device = self.accelerator.device
        input_dict_list = inputs["input_dict_list"]
        completion_tokens_text = []
        for input_dict in input_dict_list:
            completion_tokens_text.append(input_dict["completions_tokens_text"])
        completion_tokens_text = torch.tensor(completion_tokens_text).to(device)

        # (per_token_logps, entropy) = self._get_per_token_logps(
        #     model, input_dict_list, get_entropy=True
        # )
        # if self.args.ce_weight != 0.0:
        (per_token_logps, entropy) = self._get_per_token_logps(
            model, input_dict_list, get_entropy=True
        )
        # per_device_batch_size = 1, so the length of list returned from selective_log_softmax is 1
        per_token_logps = per_token_logps[0]

        # [MOD] Build per-token mask based on each sample's actual text completion length
        # Reason: selective_log_softmax right-pads with 0 when aligning across samples; without a mask, these padding positions would participate in GRPO loss and metrics, causing systematic bias.
        max_len = per_token_logps.size(1)
        token_indices = torch.arange(max_len, device=device).unsqueeze(
            0
        )  # (1, max_len)
        mask = (token_indices < completion_tokens_text.unsqueeze(1)).to(
            per_token_logps.dtype
        )  # (B, max_len)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][0]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            # [MOD] KL is also computed only on valid tokens
            # Reason: consistent with loss, masking padding to prevent KL from being diluted or amplified by invalid positions.
            per_token_kl = per_token_kl * mask

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            inputs["old_per_token_logps"][0]
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)  # 1
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        # per_token_loss = torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        if self.mask_truncated_completions:
            mask = mask * inputs["is_eos"].to(per_token_loss.dtype).unsqueeze(1)

        # [MOD] Change all per-token sum/mean computations to be mask-based
        # Reason: aggregate only over real completion text tokens, avoiding treating padding as valid tokens.
        valid_counts = mask.sum(-1).clamp(min=1.0)  # (B,)

        if self.loss_type == "grpo":
            loss = ((per_token_loss * mask).sum(-1) / valid_counts).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / valid_counts.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            # Note: keep denom consistent with TRL (batch_size * max_completion_length), but values come only from valid tokens
            loss = (per_token_loss * mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        gathered_entropy = self.accelerator.gather_for_metrics(entropy)
        self._metrics[mode]["entropy"].append(gathered_entropy.nanmean().item())

        if self.beta != 0.0:
            mean_kl = (per_token_kl).sum() / valid_counts.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).nanmean().item()
            )

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        # [MOD] Clip statistics also use mask weights and normalize by valid tokens
        # Reason: otherwise padding (right-side) would inflate/deflate clip ratios, making metrics unreliable.
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
        return loss

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
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                start_time = time.time()
                generation_batch = self._generate_and_score_completions(
                    generation_batch
                )
                self._buffered_inputs = shuffle_and_split_tensor_dict(generation_batch)
                print(
                    f"Generated new completions in {time.time() - start_time:.2f} seconds."
                )
                # generation_batch = shuffle_tensor_dict(generation_batch)
                # self._buffered_inputs = split_tensor_dict(
                #     generation_batch, self.args.steps_per_generation
                # )
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

        # Save only one safetensors weight file
        st.save_file(
            state_dict,
            os.path.join(output_dir, "model.safetensors"),
            metadata={"format": "pt"},  # Optional metadata
        )
        print(f"Saved model checkpoint to {output_dir}.")
        del state_dict
        torch.cuda.empty_cache()
