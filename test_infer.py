import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from accelerate import (
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
    init_empty_weights,
)

# ... (保持原有的 modeling 和 data 相关 import 不变) ...
from inferencer import InterleaveInferencer
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

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




# ================= 配置区 =================
MODEL_PATH = "/root/CLEAR/models/BAGEL-7B-MoT"
CHECKPOINT_PATH = "/root/CLEAR/results/20260126_102011_mix/0003000/ema.safetensors"
DEVICE = "cuda:0"
SAVE_ROOT = "visualizations"

# 推理模式配置
# 模式 1: 'jsonl' - 从文件读取
# 模式 2: 'manual' - 代码内指定
INFERENCE_MODE = "manual" 

# JSONL 模式参数
JSONL_PATH = "/root/CLEAR/datasets/processed_dataset/sft/sft_data.jsonl"
IMAGE_DIR = "/root/CLEAR/datasets/processed_dataset/sft/corruption_images"

# MANUAL 模式参数
MANUAL_INPUTS = [
    {
        "id": "manual_test_01",
        "content": [Image.open("/root/CLEAR/datasets/processed_dataset/sft/corruption_images/chart2text_cauldron__cauldron_chart2text_images_chart2text_00002754.png.png"), 
                    "Please clarify the meaning conveyed by this graph."]
    }
]

# ================= 优化后的可视化模块 =================

class BagelVisualizer:
    def __init__(self, width=1100):
        self.width = width
        self.padding = 40
        self.inner_w = width - 2 * self.padding
        
        # 样式配置：侧重于自动识别角色
        self.styles = {
            "SYSTEM":    {"bg": "#f1f3f5", "border": "#dee2e6", "tag_bg": "#adb5bd", "text": "#666666", "font_size": 14},
            "USER_IMG":  {"bg": "#f8f9fa", "border": "#dee2e6", "tag_bg": "#495057", "text": "#333333", "font_size": 18},
            "USER_TXT":  {"bg": "#ffffff", "border": "#dee2e6", "tag_bg": "#343a40", "text": "#000000", "font_size": 18},
            "BOT_THINK": {"bg": "#fdfdfe", "border": "#e7f5ff", "tag_bg": "#74c0fc", "text": "#495057", "font_size": 16},
            "BOT_IMG":   {"bg": "#fff9db", "border": "#ffe066", "tag_bg": "#fcc419", "text": "#333333", "font_size": 18},
            "BOT_TXT":   {"bg": "#e7f5ff", "border": "#a5d8ff", "tag_bg": "#228be6", "text": "#000000", "font_size": 18},
            "BOT_TOOL":  {"bg": "#fff4e6", "border": "#ffd8a8", "tag_bg": "#fd7e14", "text": "#d9480f", "font_size": 16},
        }
        
        try:
            self.fonts = {
                14: ImageFont.truetype("arial.ttf", 14),
                16: ImageFont.truetype("arial.ttf", 16),
                18: ImageFont.truetype("arial.ttf", 18),
                "TAG": ImageFont.truetype("arial.ttf", 20, encoding="utf-8")
            }
        except:
            default = ImageFont.load_default()
            self.fonts = {14: default, 16: default, 18: default, "TAG": default}

    def wrap_text(self, text, font, max_width):
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        lines = []
        for paragraph in text.split('\n'):
            words = paragraph.split(' ')
            line = ""
            for word in words:
                test_line = line + word + " "
                w = draw.textbbox((0, 0), test_line, font=font)[2]
                if w <= max_width - 40:
                    line = test_line
                else:
                    lines.append(line)
                    line = word + " "
            lines.append(line)
        return lines

    def create_block(self, style_key, tag, content):
        style = self.styles[style_key]
        font = self.fonts[style["font_size"]]
        block = {"style": style, "tag": tag, "type": "text"}
        
        if isinstance(content, Image.Image):
            block["type"] = "image"
            w, h = content.size
            scale = (self.inner_w - 60) / w
            block["content"] = content.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            block["h"] = block["content"].size[1] + 70
        else:
            lines = self.wrap_text(content, font, self.inner_w)
            block["content"] = lines
            line_h = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), "A", font=font)[3] + 8
            block["h"] = len(lines) * line_h + 75
        return block

    def draw_summary(self, output_list, save_path):
        import re
        blocks = []
        
        # 初始标记位，用于区分 USER 输入和 BOT 生成
        # 根据代码逻辑：第一个是 System，后面跟着 input_lists 的内容，再后面是 while 循环生成的
        is_model_generating = False
        
        for idx, item in enumerate(output_list):
            # 1. 第一项必定是 System Prompt
            if idx == 0 and isinstance(item, str):
                blocks.append(self.create_block("SYSTEM", "SYSTEM INSTRUCTIONS", item))
                continue

            if isinstance(item, Image.Image):
                # 如果还没开始生成推理文本，说明是初始输入图片
                if not is_model_generating:
                    blocks.append(self.create_block("USER_IMG", "INPUT IMAGE", item))
                else:
                    blocks.append(self.create_block("BOT_IMG", "RESTORED IMAGE", item))
            
            elif isinstance(item, str):
                # 检测是否进入模型生成阶段 (通常以 <think> 开始)
                if "<think>" in item:
                    is_model_generating = True
                
                if not is_model_generating:
                    # 模型开始生成前的所有文本均视为用户问题/输入
                    blocks.append(self.create_block("USER_TXT", "QUESTION", item))
                else:
                    # 模型生成阶段的正则解析
                    think = re.search(r"<think>(.*?)</think>", item, re.DOTALL)
                    answer = re.search(r"<answer>(.*?)</answer>", item, re.DOTALL)
                    
                    if think:
                        blocks.append(self.create_block("BOT_THINK", "THOUGHT PROCESS",item.strip()))
                    
                    if "<image_restore>" in item:
                        blocks.append(self.create_block("BOT_TOOL", "TOOL ACTION", item.strip()))
                    
                    if answer:
                        blocks.append(self.create_block("BOT_TXT", "FINAL ANSWER", item.strip()))
                    
                    # 兜底：如果模型没有输出标准标签
                    if not think and not answer and "<image_restore>" not in item:
                        blocks.append(self.create_block("BOT_TXT", "MODEL RESPONSE", item.strip()))

        # 绘图逻辑
        total_h = sum(b["h"] for b in blocks) + (len(blocks) * 20) + 100
        canvas = Image.new('RGB', (self.width, total_h), "#f8f9fa")
        draw = ImageDraw.Draw(canvas)
        
        curr_y = 50
        for b in blocks:
            s = b["style"]
            draw.rectangle([self.padding, curr_y, self.width-self.padding, curr_y+b["h"]], fill=s["bg"], outline=s["border"])
            draw.rectangle([self.padding, curr_y, self.padding+10, curr_y+b["h"]], fill=s["tag_bg"])
            draw.text((self.padding+25, curr_y+15), b["tag"], font=self.fonts["TAG"], fill=s["tag_bg"])
            
            content_y = curr_y + 55
            if b["type"] == "image":
                canvas.paste(b["content"], (self.padding+35, content_y))
            else:
                for line in b["content"]:
                    draw.text((self.padding+35, content_y), line, font=self.fonts[s["font_size"]], fill=s["text"])
                    content_y += (ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), "A", font=self.fonts[s["font_size"]])[3] + 8)
            curr_y += b["h"] + 20
            
        canvas.save(save_path)

# 调用函数只需传入 output_list
def create_visual_summary(output_list, save_path):
    viz = BagelVisualizer(width=1100)
    viz.draw_summary(output_list, save_path)



# --- 模型加载逻辑 (保持原有逻辑) ---
llm_config = Qwen2Config.from_json_file(os.path.join(MODEL_PATH, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(MODEL_PATH, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

vae_model, vae_config = load_ae(local_path=os.path.join(MODEL_PATH, "ae.safetensors"))
vae_model = vae_model.to(device=DEVICE, dtype=torch.bfloat16).eval()

config = BagelConfig(
    visual_gen=True, visual_und=True, llm_config=llm_config,
    vit_config=vit_config, vae_config=vae_config,
    vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
    latent_patch_size=2, max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(518, 224, 14)

model = load_checkpoint_and_dispatch(
    model, checkpoint=CHECKPOINT_PATH,
    device_map={"": DEVICE}, offload_buffers=False, dtype=torch.bfloat16,
)
model.eval()

inferencer = InterleaveInferencer(
    model=model, vae_model=vae_model, tokenizer=tokenizer,
    vae_transform=vae_transform, vit_transform=vit_transform,
    new_token_ids=new_token_ids, device=DEVICE,
)

# --- 数据准备逻辑 ---
tasks = []
if INFERENCE_MODE == "jsonl":
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            img_path = os.path.join(IMAGE_DIR, data["image"])
            question = ""
            for msg in data["conversations"]:
                if msg["from"] == "human":
                    question = msg["value"]
                    break
            tasks.append({
                "id": data.get("id", "unknown").replace("/", "_"),
                "input_imgs": [Image.open(img_path).convert("RGB")],
                "question": question
            })
else:
    for item in MANUAL_INPUTS:
        imgs = [x for x in item["content"] if isinstance(x, Image.Image)]
        txts = [x for x in item["content"] if isinstance(x, str)]
        tasks.append({
            "id": item["id"],
            "input_imgs": imgs,
            "question": " ".join(txts)
        })

# --- 推理循环 ---
os.makedirs(SAVE_ROOT, exist_ok=True)
inference_hyper = dict(
    max_think_token_n=1000, max_new_tokns=8192, do_sample=False,
    text_temperature=1.0, max_inter_num=3, cfg_text_scale=4.0,
    cfg_img_scale=1.5, cfg_interval=[0.4, 1.0], timestep_shift=3.0,
    num_timesteps=50, cfg_renorm_min=0.0, cfg_renorm_type="global",
)

for task in tqdm(tasks):
    input_lists = []
    for img in task["input_imgs"]:
        input_lists.append(img)
    input_lists.append(task["question"])

    # 这里的 output_list 已经包含了 System Prompt, Input Image, Question 和推理结果
    output_list = inferencer.interleave_reason_tool_condition(
        input_lists=input_lists, **inference_hyper,
    )

    summary_path = os.path.join(SAVE_ROOT, f"{task['id']}_summary.png")
    # 直接传入 output_list 即可
    create_visual_summary(output_list, summary_path)

print(f"推理并可视化完成。结果保存在 {SAVE_ROOT}")