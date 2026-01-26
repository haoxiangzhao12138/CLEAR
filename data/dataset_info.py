# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import InterleaveReasonIterableDataset, TextReasonIterableDataset


DATASET_REGISTRY = {
    "reason_interleave_dataset": InterleaveReasonIterableDataset,
    "reason_text_dataset": InterleaveReasonIterableDataset,
    "pure_text_dataset": TextReasonIterableDataset,
}


DATASET_INFO = {
    "reason_interleave_dataset": {
        "reason_interleave_dataset": {
            "clean_image_dir": "/root/CLEAR/datasets/processed_dataset/sft/images",
            "corrupted_image_dir": "/root/CLEAR/datasets/processed_dataset/sft/corruption_images",
            "jsonl_path": "/root/CLEAR/datasets/processed_dataset/sft/agent_interleave_data_filtered_tool.jsonl",
            "num_total_samples": 3628,
        }
    },
    "reason_text_dataset":{
        "reason_text_dataset": {
            "clean_image_dir": "/root/CLEAR/datasets/processed_dataset/sft/images",
            "corrupted_image_dir": "/root/CLEAR/datasets/processed_dataset/sft/corruption_images",
            "jsonl_path": "/root/CLEAR/datasets/processed_dataset/sft/agent_interleave_data_filtered_no_tool.jsonl",
            "num_total_samples": 23075,
        }
    },
    "pure_text_dataset": {
        "pure_text_dataset": {
            "corrupted_image_dir": "/root/CLEAR/datasets/processed_dataset/sft/corruption_images",
            "jsonl_path": "/root/CLEAR/datasets/processed_dataset/sft/pure_text_data.jsonl",
            "num_total_samples": 9746,
        }
    },

}
