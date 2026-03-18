# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any
import io, base64, sys, os
from PIL import Image
import torch
from .data.data_utils import pil_img2rgb
from .modeling.bagel.qwen2_navit import NaiveCache
import re
import numpy as np
import matplotlib

# Add CLEAR directory to path to import prompts
CLEAR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, CLEAR_ROOT)
from prompts import VLM_THINK_SYSTEM_PROMPT, GEN_THINK_SYSTEM_PROMPT, INTERLEAVE_REASON_SYSTEM_PROMPT, RESTORE_TOKEN


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)  # 把图像写入内存缓冲区
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64


def dict_to_device(dict: Dict, device):
    for key in dict:
        if isinstance(dict[key], torch.Tensor):
            dict[key] = dict[key].to(device)
    return dict


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    else:
        raise TypeError("img should be np.ndarray or torch.Tensor")
    return hwc


class InterleaveInferencer:
    def __init__(
        self,
        model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids,
        device,
        verbose_print: bool = False,
        enable_image_gen_stats: bool = False,
    ):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.device = device
        self.verbose_print = verbose_print
        self.enable_image_gen_stats = enable_image_gen_stats

        # Image generation statistics
        self.image_gen_count = 0
        self.data_count = 0

    def increment_data_count(self):
        """Increment the data sample count."""
        self.data_count += 1

    def increment_image_gen_count(self):
        """Increment the image generation count."""
        self.image_gen_count += 1
        if self.enable_image_gen_stats:
            print(f"\033[33m[Image Gen Stats] Data: {self.data_count}, Image Generations: {self.image_gen_count}\033[0m")

    def get_image_gen_stats(self):
        """Get current image generation statistics."""
        return {
            "data_count": self.data_count,
            "image_gen_count": self.image_gen_count,
        }

    def print_image_gen_stats(self):
        """Print current image generation statistics."""
        stats = self.get_image_gen_stats()
        print(f"\033[36m[Final Image Gen Stats] Total Data: {stats['data_count']}, Total Image Generations: {stats['image_gen_count']}\033[0m")

    def init_gen_context(self):
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(
                self.model.config.llm_config.num_hidden_layers
            ),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference,

        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = dict_to_device(generation_input, self.device)

        past_key_values = self.model.forward_cache_update_text(
            past_key_values, **generation_input
        )
        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values

        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference,

        assert vae or vit
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = dict_to_device(generation_input, self.device)
            past_key_values = self.model.forward_cache_update_vae(
                self.vae_model, past_key_values, **generation_input
            )

        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = dict_to_device(generation_input, self.device)
            past_key_values = self.model.forward_cache_update_vit(
                past_key_values, **generation_input
            )

        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values

        return gen_context

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=3.0,
    ):
        # Increment image generation count
        self.increment_image_gen_count()

        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )
        generation_input = dict_to_device(generation_input, self.device)

        # text cfg
        cfg_text_past_key_values = cfg_text_precontext["past_key_values"]
        kv_lens_cfg = cfg_text_precontext["kv_lens"]
        ropes_cfg = cfg_text_precontext["ropes"]
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )
        generation_input_cfg_text = dict_to_device(
            generation_input_cfg_text, self.device
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext["past_key_values"]
        kv_lens_cfg = cfg_img_precontext["kv_lens"]
        ropes_cfg = cfg_img_precontext["ropes"]
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )
        generation_input_cfg_img = dict_to_device(generation_input_cfg_img, self.device)

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text[
                "cfg_packed_position_ids"
            ],
            cfg_text_packed_query_indexes=generation_input_cfg_text[
                "cfg_packed_query_indexes"
            ],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text[
                "cfg_packed_key_value_indexes"
            ],
            cfg_img_packed_position_ids=generation_input_cfg_img[
                "cfg_packed_position_ids"
            ],
            cfg_img_packed_query_indexes=generation_input_cfg_img[
                "cfg_packed_query_indexes"
            ],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img[
                "cfg_packed_key_value_indexes"
            ],
        )

        unpacked_latent, unpacked_latent_llm = unpacked_latent
        image = self.decode_image(unpacked_latent[0], image_shape)
        return image


    def decode_image(self, latent, image_shape):
        # decode latent to image
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(
            1,
            h,
            w,
            self.model.latent_patch_size,
            self.model.latent_patch_size,
            self.model.latent_channel,
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1,
            self.model.latent_channel,
            h * self.model.latent_patch_size,
            w * self.model.latent_patch_size,
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(
        self,
        gen_context,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input = self.model.prepare_start_tokens(
            kv_lens, ropes, self.new_token_ids
        )
        generation_input = dict_to_device(generation_input, self.device)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            end_token_id=self.new_token_ids["eos_token_id"],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:, 0])
        output = output.split("<|im_end|>")[0].split("<|im_start|>")[1]

        # Print output if verbose_print is enabled
        if self.verbose_print:
            print(f"\033[32m[Generated Text]\n{output}\033[0m")

        return output


    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        # origin BAGEL interleave inference function

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(
                    system_prompt, cfg_img_context
                )

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(
                        input_term, cfg_img_context
                    )

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=not understanding_output
                    )

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n,
                    )
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )

                output_list.append(img)

        return output_list

    @torch.no_grad()
    def image_generation_edit(
        self,
        input_list: List[Union[str, Image.Image]],
        max_think_token_n=1024,
        do_sample=False,
        think=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        # image edit function, you can use it to generate the segmentation map or the netural image
        # the input_list should have the input image and the text prompt

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(
                    system_prompt, cfg_img_context
                )

            for input_term in input_list:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(
                        input_term, cfg_img_context
                    )

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=True
                    )
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if think:
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                gen_context = self.update_context_text(gen_text, gen_context)
                output_list.append(gen_text)

            img = self.gen_image(
                image_shapes,
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
            )

            output_list.append(img)

        return output_list

    @torch.no_grad()
    def interleave_reason_tool_condition(
        self,
        input_lists: List[Union[str, Image.Image]],
        max_inter_num=3,
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=30,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        top_p=1.0,
        output_need_vae=False,  # 控制生成图片后是否将 VAE token 插入上下文
        output_need_vit=True,   # 控制生成图片后是否将 ViT token 插入上下文
        consider_think=True,
        **kwargs,
    ) -> List[Union[str, Image.Image]]:
        # cooperative reasoning and perception generation function
        # the input_list shuould have the input image and the text prompt
        # it can generate the interleaved multimodal chain-of-thought by the model
        # but the image generation is decided by the model itself

        # Increment data count for statistics
        self.increment_data_count()

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        edit_cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            system_prompt = INTERLEAVE_REASON_SYSTEM_PROMPT
            gen_context = self.update_context_text(system_prompt, gen_context)
            edit_cfg_img_context = self.update_context_text(
                system_prompt, edit_cfg_img_context
            )
            output_list.append(system_prompt)

            answer_pattern = r"<answer>(.*?)</answer>"
            restore_pattern = RESTORE_TOKEN
            # 处理初始输入列表
            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    edit_cfg_img_context = self.update_context_text(
                        input_term, edit_cfg_img_context
                    )
                    output_list.append(input_term)
                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(input_term, gen_context)
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                    output_list.append(input_term)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            inter_num = 0
            while True:
                inter_num += 1
                # 1. 生成推理文本
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                    top_p=top_p,
                )
                output_list.append(gen_text)
                
                # 检查是否达成最终答案
                answer_match = re.search(answer_pattern, gen_text, re.DOTALL)
                if answer_match:
                    return output_list

                if inter_num >= max_inter_num:
                    break

                # 2. 检查是否需要调用图像恢复工具
                restore_match = re.search(restore_pattern, gen_text)

                if restore_match:
                    # 准备生图的 CFG 上下文
                    cfg_text_context = deepcopy(gen_context)

                    if not consider_think:
                        edit_cfg_prompt = gen_text.replace("<image_restore>", "")
                        cfg_text_context = self.update_context_text(
                            edit_cfg_prompt, cfg_text_context
                        )
                    
                    gen_context = self.update_context_text(gen_text, gen_context)
                    edit_cfg_img_context = self.update_context_text(
                        gen_text, edit_cfg_img_context
                    )
                    
                    # 3. 调用图像生成（恢复）工具
                    img = self.gen_image(
                        image_shapes,
                        gen_context=gen_context,
                        cfg_text_precontext=cfg_text_context,
                        cfg_img_precontext=edit_cfg_img_context,
                        cfg_text_scale=cfg_text_scale,
                        cfg_img_scale=cfg_img_scale,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )

                    # 4. 反馈结果
                    output_list.append(pil_img2rgb(img))

                    # 根据开关决定是否将生成的图片喂回模型 KV Cache
                    if output_need_vae or output_need_vit:
                        img_processed = self.vae_transform.resize_transform(pil_img2rgb(img))
                        gen_context = self.update_context_image(
                            img_processed, gen_context, vae=output_need_vae, vit=output_need_vit
                        )

                        # 同步更新 CFG 用的上下文
                        cfg_text_context = deepcopy(gen_context)
                        edit_cfg_img_context = self.update_context_image(
                            img_processed, edit_cfg_img_context, vae=output_need_vae, vit=output_need_vit
                        )
                else:
                    break

        return output_list


    @torch.no_grad()
    def text_reason(
        self,
        input_lists: List[Union[str, Image.Image]],
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        top_p=1.0,
        is_thinking=True,
        **kwargs,
    ) -> List[Union[str, Image.Image]]:
        # reasoning enhancement answer function
        # it will reason with the textual chain-of-thought first, then generate the answer

        output_list = []
        gen_context = self.init_gen_context()

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if is_thinking:
                system_prompt = VLM_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                output_list.append(system_prompt)
            for input_term in input_lists:
                if isinstance(input_term, str):
                    gen_context = self.update_context_text(input_term, gen_context)
                    output_list.append(input_term)
                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=False
                    )
                    output_list.append(input_term)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            gen_text = self.gen_text(
                gen_context,
                do_sample=do_sample,
                temperature=text_temperature,
                max_length=max_think_token_n,
                top_p=top_p,
            )
            output_list.append(gen_text)
        return output_list

    def __call__(
        self, image: Optional[Image.Image] = None, text: Optional[str] = None, **kargs
    ) -> Dict[str, Any]:
        output_dict = {"image": None, "text": None}

        if image is None and text is None:
            print("Please provide at least one input: either an image or text.")
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict["image"] = i
            elif isinstance(i, str):
                output_dict["text"] = i
        return output_dict
