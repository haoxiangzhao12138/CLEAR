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
# 将上级目录添加到 sys.path
sys.path.append(str(Path(__file__).parent.parent))
from prompts import VLM_THINK_SYSTEM_PROMPT, GEN_THINK_SYSTEM_PROMPT, INTERLEAVE_REASON_SYSTEM_PROMPT, TEXT_REASON_SYSTEM_PROMPT, RESTORE_TOKEN

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte



def base64_to_image(base64_str):
    """
    将base64字符串转换为PIL Image并保存到本地

    参数:
        base64_str (str): 图像的base64编码字符串
    """
    # 移除base64字符串可能包含的前缀（如'data:image/jpeg;base64,'）
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]

    # 解码base64字符串为字节数据
    image_bytes = base64.b64decode(base64_str)

    # 将字节数据转换为PIL Image对象
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def load_image_from_path(directory, filename):
    """
    辅助函数：从指定目录加载图片
    """
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        print(f"[Warning] Image not found: {path}")
        # 返回全黑图防止训练中断
        return Image.new('RGB', (224, 224), (0, 0, 0))
    return Image.open(path).convert("RGB")


class InterleaveReasonIterableDataset(
    InterleavedBaseIterableDataset, JSONLStandardIterableDataset
):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        transform,          # VAE transform (用于生成目标的清晰图)
        vit_transform,      # ViT transform (用于输入的损毁图)
        jsonl_path_list,    # 来自 PackedDataset (列表)
        data_dir_list,      # 来自 PackedDataset
        num_used_data,      # 来自 YAML
        clean_image_dir,    # 【自定义参数】清晰图片目录
        corrupted_image_dir,# 【自定义参数】损毁图片目录
        local_rank=0,
        world_size=1,
        num_workers=1,
        data_status=None,
        **kwargs
    ):
        # 1. 保存自定义路径
        self.clean_image_dir = clean_image_dir
        self.corrupted_image_dir = corrupted_image_dir

        # 2. 初始化父类 JSONLStandardIterableDataset
        # 这个父类负责：分布式切分(Sharding)、断点续训(data_status)、多线程读取(Worker)
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
            shuffle_lines=True,  # 默认打乱数据
            shuffle_seed=42
        )

    def parse_row(self, image_dir, row):
        """
        兼容逻辑：
        1. 加载损毁图作为 Input (ViT)。
        2. 遇到 GPT 回复时检测是否包含 RESTORE_TOKEN。
        3. 包含 -> 拆分文本，拼接 Token，插入 MSE 图片生成任务。
        4. 不包含 -> 纯文本 CE Loss，跳过图片生成。
        """
        image_filename = row.get("image")
        conversations = row.get("conversations", [])

        # 1. 初始化数据结构
        data = self._init_data()

        # 2. 加载图片
        # 损毁图：必须加载，作为 ViT 输入 (Condition)
        corrupted_img = load_image_from_path(self.corrupted_image_dir, image_filename)
        
        # 清晰图：仅在需要计算生成 Loss 时使用，但预加载也没问题
        clean_img = load_image_from_path(self.clean_image_dir, image_filename)

        # 3. 添加 System Prompt (不计算 Loss)
        data = self._add_text(
            data,
            INTERLEAVE_REASON_SYSTEM_PROMPT,
            need_loss=False,
        )

        # 4. 【Input】添加损毁图片
        # 无论是否有生成任务，模型都需要看到这张损毁的图来回答问题
        data = self._add_image(
            data,
            corrupted_img,
            need_loss=False,
            need_vae=True,
            need_vit=True, # 作为 Encoder 输入
        )

        # 5. 处理多轮对话
        for turn in conversations:
            role = turn["from"]
            content = turn["value"]

            if role == "human":
                
                # 用户问题：不计算 Loss
                content = content.replace("<image>", "").strip()
                data = self._add_text(data, content, need_loss=False)
            
            elif role == "gpt":
                # === 分支判断 ===
                if RESTORE_TOKEN in content:
                    # --- Case A: 包含生成任务 ---
                    parts = content.split(RESTORE_TOKEN)
                    
                    # 1. 思考过程 + Token
                    # 将 Token 拼在后面，让模型学习在思考结束后“按下按钮”
                    thought_process = parts[0] + RESTORE_TOKEN
                    data = self._add_text(data, thought_process, need_loss=True)
                    
                    # 2. 【Output】插入清晰图片 (生成动作)
                    # 只有在这里才添加 MSE Loss 任务
                    data = self._add_image(
                        data,
                        clean_img,
                        need_loss=True,  # 计算 MSE
                        need_vae=True,   # 走 VAE
                        need_vit=True,  # 不作为 ViT 输入 (防止 Leak)
                    )
                    
                    # 3. 后续文本 (如果有)
                    if len(parts) > 1 and parts[1].strip():
                        data = self._add_text(data, parts[1], need_loss=True)
                
                else:
                    # --- Case B: 纯文本回答 (无生成) ---
                    # 直接将整个 content 作为一个文本块
                    # 没有任何 add_image 操作，所以这步只有 CE Loss
                    data = self._add_text(data, content, need_loss=True)

        return data

class TextReasonIterableDataset(
    InterleavedBaseIterableDataset, JSONLStandardIterableDataset
):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        transform,          # VAE transform (用于生成目标的清晰图)
        vit_transform,      # ViT transform (用于输入的损毁图)
        jsonl_path_list,    # 来自 PackedDataset (列表)
        data_dir_list,      # 来自 PackedDataset
        num_used_data,      # 来自 YAML
        corrupted_image_dir,# 【自定义参数】损毁图片目录
        local_rank=0,
        world_size=1,
        num_workers=1,
        data_status=None,
        **kwargs
    ):
        # 1. 保存自定义路径
        self.corrupted_image_dir = corrupted_image_dir

        # 2. 初始化父类 JSONLStandardIterableDataset
        # 这个父类负责：分布式切分(Sharding)、断点续训(data_status)、多线程读取(Worker)
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
            shuffle_lines=True,  # 默认打乱数据
            shuffle_seed=42
        )



    def parse_row(self, image_dir, row):
        # 1. 获取图片路径并加载
        # data_info 中指定了 image_dir，row["image"] 通常是文件名
        image_dir = self.corrupted_image_dir
        image_path = os.path.join(image_dir, row["image"])
        image = Image.open(image_path).convert("RGB")

        data = self._init_data()

        # 2. 添加 System Prompt
        # 即使是纯文本推理，通常也需要 System Prompt 来设定 Agent 行为
        data = self._add_text(
            data,
            TEXT_REASON_SYSTEM_PROMPT,
            need_loss=False,
        )

        # 3. 添加输入图片
        # 关键修改：need_vae=False (不生成图片), need_vit=True (作为输入编码)
        data = self._add_image(
            data,
            image,
            need_loss=False, # 输入图片不需要计算 Loss
            need_vae=False,  # 不需要 VAE 进行 Tokenize (因为不进行图像生成)
            need_vit=True,   # 需要 ViT 提取特征
        )

        # 4. 解析对话 (Standard LLaVA/SFT format)
        conversations = row["conversations"]
        
        for turn in conversations:
            role = turn["from"]
            value = turn["value"]

            if role == "human":
                # 去除 <image> 标签，因为图片已经通过 _add_image 添加了
                value = value.replace("<image>", "").strip()
                
                # 用户指令 -> 不需要 Loss
                data = self._add_text(
                    data, 
                    value, 
                    need_loss=False
                )
                
            elif role == "gpt":
                # 模型回答 (包含 <think> 和 <answer>) -> 需要 Loss
                data = self._add_text(
                    data, 
                    value, 
                    need_loss=True
                )

        return data