# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
CLEAR: Standalone inference demo.

Usage:
    python demo.py \
        --model_config_path ./models/BAGEL-7B-MoT \
        --model_param_path ./results/<YOUR_CHECKPOINT> \
        --image_path /path/to/degraded_image.jpg \
        --question "What is shown in this image?"
"""

import argparse
import torch
from PIL import Image
from accelerate import load_checkpoint_and_dispatch, init_empty_weights

from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer


def load_model(model_config_path, model_param_path, is_ema=False, device="cuda:0"):
    """Load CLEAR model from checkpoint."""
    llm_config = Qwen2Config.from_json_file(
        f"{model_config_path}/llm_config.json"
    )
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(
        f"{model_config_path}/vit_config.json"
    )
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(
        local_path=f"{model_config_path}/ae.safetensors"
    )
    vae_model = vae_model.to(device=device, dtype=torch.bfloat16).eval()

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )

    tokenizer = Qwen2Tokenizer.from_pretrained(model_config_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    checkpoint_file = f"{model_param_path}/{'ema' if is_ema else 'model'}.safetensors"
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_file,
        device_map={"": device},
        offload_buffers=False,
        dtype=torch.bfloat16,
    )
    model = model.eval()

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(518, 224, 14)

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
        device=device,
    )

    return inferencer


def main():
    parser = argparse.ArgumentParser(description="CLEAR inference demo")
    parser.add_argument("--model_config_path", type=str, required=True,
                        help="Path to BAGEL-7B-MoT config directory")
    parser.add_argument("--model_param_path", type=str, required=True,
                        help="Path to CLEAR checkpoint directory")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--question", type=str, required=True,
                        help="Question about the image")
    parser.add_argument("--is_ema", action="store_true",
                        help="Load EMA weights instead of raw model weights")
    parser.add_argument("--reasoning_mode", type=str, default="interleave",
                        choices=["interleave", "text", "image"],
                        help="Reasoning mode: interleave (default), text, or image")
    parser.add_argument("--max_think_token_n", type=int, default=4096,
                        help="Maximum number of thinking tokens")
    args = parser.parse_args()

    print(f"Loading model from {args.model_param_path}...")
    inferencer = load_model(
        args.model_config_path,
        args.model_param_path,
        is_ema=args.is_ema,
    )

    image = Image.open(args.image_path).convert("RGB")
    input_lists = [image, args.question]

    print(f"Running {args.reasoning_mode} reasoning...")
    if args.reasoning_mode == "interleave":
        output_list, _ = inferencer.interleave_reason_tool_condition(
            input_lists=input_lists,
            max_think_token_n=args.max_think_token_n,
            do_sample=False,
            text_temperature=0.3,
        )
        # Print all text outputs
        for item in output_list:
            if isinstance(item, str):
                print(item)
    elif args.reasoning_mode == "text":
        output_list = inferencer.text_reason(
            input_lists=input_lists,
            max_think_token_n=args.max_think_token_n,
            do_sample=False,
            text_temperature=0.3,
        )
        print(output_list[-1])
    else:
        raise ValueError(f"Unsupported reasoning mode for demo: {args.reasoning_mode}")


if __name__ == "__main__":
    main()
