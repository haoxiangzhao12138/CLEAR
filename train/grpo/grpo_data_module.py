# -*- coding: utf-8 -*-

import json
import os
import random
from typing import Dict, Any, Optional

from torch.utils.data import Dataset
from PIL import Image


class GRPODataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str, shuffle: bool = True, seed: int = 42):
        super().__init__()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data_list = [line for line in f if line.strip()]

        if shuffle:
            random.seed(seed)
            random.shuffle(self.data_list)

        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.data_list)

    @staticmethod
    def _extract_conv(conversations, who: str) -> str:
        """Extract the first message value from conversations with from==who."""
        if not isinstance(conversations, list):
            return ""
        for msg in conversations:
            if isinstance(msg, dict) and msg.get("from") == who:
                return msg.get("value", "") or ""
        return ""

    @staticmethod
    def _clean_human_text(text: str) -> str:
        """Remove leading <image> tag if present."""
        if not isinstance(text, str):
            return ""
        t = text.strip()
        if t.startswith("<image>"):
            # remove only the first line "<image>" and possible following newline
            t = t[len("<image>"):].lstrip("\n").lstrip()
        return t

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = json.loads(self.data_list[index])

        # --------- id / flag ----------
        data_id = data.get("id", data.get("data_id", str(index)))
        # flag = data.get("split_type", data.get("flag", "boundary"))

        # --------- conversations -> question / solution ----------
        convs = data.get("conversations", [])
        question = self._clean_human_text(self._extract_conv(convs, "human"))
        answer = self._extract_conv(convs, "gpt")

        # --------- image ----------
        image_name = data.get("image")
        if not image_name:
            raise KeyError(f"Missing 'image' field for sample id={data_id}")

        img_path = os.path.join(self.image_root, image_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path} (sample id={data_id})")

        img = Image.open(img_path).convert("RGB")

        return {
            "question": question,
            "solution": answer,
            "image": img,
            "data_id": data_id,
            "image_name": image_name,
        }