# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io, base64
import random
from PIL import Image, ImageFile, PngImagePlugin
import json
from .interleave_t2i_dataset import (
    InterleavedBaseIterableDataset,
    ParquetStandardIterableDataset,
    JSONLStandardIterableDataset,
)
from ..data_utils import pil_img2rgb
import os
from io import BytesIO
from pathlib import Path
import sys
# Add parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from prompts import VLM_THINK_SYSTEM_PROMPT, GEN_THINK_SYSTEM_PROMPT, INTERLEAVE_REASON_SYSTEM_PROMPT, TEXT_REASON_SYSTEM_PROMPT, RESTORE_TOKEN

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte



def base64_to_image(base64_str):
    """
    Convert a base64 string to a PIL Image.

    Args:
        base64_str (str): Base64-encoded string of the image.
    """
    # Remove possible prefix from the base64 string (e.g., 'data:image/jpeg;base64,')
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]

    # Decode the base64 string to bytes
    image_bytes = base64.b64decode(base64_str)

    # Convert bytes to a PIL Image object
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def load_image_from_path(directory, filename):
    """
    Helper function: load an image from the specified directory.
    """
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        print(f"[Warning] Image not found: {path}")
        # Return an all-black image to prevent training interruption
        return Image.new('RGB', (224, 224), (0, 0, 0))
    return Image.open(path).convert("RGB")


class InterleaveReasonIterableDataset(
    InterleavedBaseIterableDataset, JSONLStandardIterableDataset
):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        transform,          # VAE transform (for the clean image used as generation target)
        vit_transform,      # ViT transform (for the corrupted image used as input)
        jsonl_path_list,    # From PackedDataset (list)
        data_dir_list,      # From PackedDataset
        num_used_data,      # From YAML
        clean_image_dir,    # [Custom parameter] Clean image directory
        corrupted_image_dir,# [Custom parameter] Corrupted image directory
        output_need_vit=True,  # [New] Controls whether output images need ViT
        enable_distill=False,  # When True, align ViT resize to match VAE token grid
        local_rank=0,
        world_size=1,
        num_workers=1,
        data_status=None,
        **kwargs
    ):
        # 1. Save custom paths
        self.clean_image_dir = clean_image_dir
        self.corrupted_image_dir = corrupted_image_dir
        self.output_need_vit = output_need_vit
        self.enable_distill = enable_distill

        # 2. Initialize parent class JSONLStandardIterableDataset
        # This parent class handles: distributed sharding, checkpoint resumption (data_status), multi-worker reading
        JSONLStandardIterableDataset.__init__(
            self,
            dataset_name=dataset_name,
            transform=transform,
            tokenizer=tokenizer,
            vit_transform=vit_transform,
            data_dir_list=data_dir_list,
            jsonl_path_list=jsonl_path_list,
            num_used_data=num_used_data,
            local_rank=local_rank,
            world_size=world_size,
            num_workers=num_workers,
            data_status=data_status,
            shuffle_lines=True,  # Shuffle data by default
            shuffle_seed=42
        )

    def parse_row(self, image_dir, row):
        """
        Compatibility logic:
        1. Load the corrupted image as Input (ViT).
        2. When encountering a GPT reply, check if it contains RESTORE_TOKEN.
        3. If it does -> split the text, concatenate tokens, insert MSE image generation task.
        4. If not -> pure text CE Loss, skip image generation.
        """
        image_filename = row.get("image")
        conversations = row.get("conversations", [])

        # 1. Initialize data structure
        data = self._init_data()

        # 2. Load images
        # Corrupted image: must be loaded, used as ViT input (Condition)
        corrupted_img = load_image_from_path(self.corrupted_image_dir, image_filename)

        # Clean image: only used when computing generation Loss, but preloading is fine
        clean_img = load_image_from_path(self.clean_image_dir, image_filename)

        # 3. Add System Prompt (no loss computation)
        data = self._add_text(
            data,
            INTERLEAVE_REASON_SYSTEM_PROMPT,
            need_loss=False,
        )

        # 4. [Input] Add the corrupted image
        # Regardless of whether there is a generation task, the model needs to see this corrupted image to answer questions
        data = self._add_image(
            data,
            corrupted_img,
            need_loss=False,
            need_vae=True,
            need_vit=True, # Used as Encoder input
        )

        # 5. Process multi-turn conversation
        for turn in conversations:
            role = turn["from"]
            content = turn["value"]

            if role == "human":
                
                # User question: no loss computation
                content = content.replace("<image>", "").strip()
                data = self._add_text(data, content, need_loss=False)
            
            elif role == "gpt":
                # === Branch decision ===
                if RESTORE_TOKEN in content:
                    # --- Case A: Contains generation task ---
                    parts = content.split(RESTORE_TOKEN)
                    
                    # 1. Thinking process + Token
                    # Append the token after it, so the model learns to “press the button” after thinking
                    thought_process = parts[0] + RESTORE_TOKEN
                    data = self._add_text(data, thought_process, need_loss=True)
                    
                    # 2. [Output] Insert clean image (generation action)
                    # Only add MSE Loss task here
                    data = self._add_image(
                        data,
                        clean_img,
                        need_loss=True,  # Compute MSE
                        need_vae=True,   # Use VAE
                        need_vit=self.output_need_vit,  # Controlled by parameter whether to use as ViT input
                    )
                    
                    # 3. Subsequent text (if any)
                    if len(parts) > 1 and parts[1].strip():
                        data = self._add_text(data, parts[1], need_loss=True)
                
                else:
                    # --- Case B: Pure text answer (no generation) ---
                    # Directly use the entire content as a single text block
                    # No add_image operation here, so this step only has CE Loss
                    data = self._add_text(data, content, need_loss=True)

        return data

class TextReasonIterableDataset(
    InterleavedBaseIterableDataset, JSONLStandardIterableDataset
):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        transform,          # VAE transform (for the clean image used as generation target)
        vit_transform,      # ViT transform (for the corrupted image used as input)
        jsonl_path_list,    # From PackedDataset (list)
        data_dir_list,      # From PackedDataset
        num_used_data,      # From YAML
        corrupted_image_dir,# [Custom parameter] Corrupted image directory
        enable_distill=False,
        local_rank=0,
        world_size=1,
        num_workers=1,
        data_status=None,
        **kwargs
    ):
        # 1. Save custom paths
        self.corrupted_image_dir = corrupted_image_dir
        self.enable_distill = enable_distill

        # 2. Initialize parent class JSONLStandardIterableDataset
        # This parent class handles: distributed sharding, checkpoint resumption (data_status), multi-worker reading
        JSONLStandardIterableDataset.__init__(
            self,
            dataset_name=dataset_name,
            transform=transform,
            tokenizer=tokenizer,
            vit_transform=vit_transform,
            data_dir_list=data_dir_list,
            jsonl_path_list=jsonl_path_list,
            num_used_data=num_used_data,
            local_rank=local_rank,
            world_size=world_size,
            num_workers=num_workers,
            data_status=data_status,
            shuffle_lines=True,  # Shuffle data by default
            shuffle_seed=42
        )



    def parse_row(self, image_dir, row):
        # 1. Get image path and load
        # image_dir is specified in data_info; row["image"] is typically the filename
        image_dir = self.corrupted_image_dir
        image_path = os.path.join(image_dir, row["image"])
        image = Image.open(image_path).convert("RGB")

        data = self._init_data()

        # 2. Add System Prompt
        # Even for pure text reasoning, a System Prompt is usually needed to define agent behavior
        data = self._add_text(
            data,
            TEXT_REASON_SYSTEM_PROMPT,
            need_loss=False,
        )

        # 3. Add input image
        # Key modification: need_vae=False (no image generation), need_vit=True (used as input encoding)
        data = self._add_image(
            data,
            image,
            need_loss=False, # Input image does not need loss computation
            need_vae=False,  # No VAE tokenization needed (no image generation)
            need_vit=True,   # Need ViT to extract features
        )

        # 4. Parse conversations (Standard LLaVA/SFT format)
        conversations = row["conversations"]
        
        for turn in conversations:
            role = turn["from"]
            value = turn["value"]

            if role == "human":
                # Remove <image> tag since the image has already been added via _add_image
                value = value.replace("<image>", "").strip()
                
                # User instruction -> no loss needed
                data = self._add_text(
                    data, 
                    value, 
                    need_loss=False
                )
                
            elif role == "gpt":
                # Model answer (contains <think> and <answer>) -> needs loss
                data = self._add_text(
                    data, 
                    value, 
                    need_loss=True
                )

        return data