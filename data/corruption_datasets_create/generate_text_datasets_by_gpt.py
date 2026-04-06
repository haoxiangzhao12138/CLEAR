import os
import json
import base64
import time
import re
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, Tuple

from openai import OpenAI
from tqdm import tqdm

# =======================
# 0) Config
# =======================

# ---- API ----
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# 生成模型
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1")

# ---- Paths ----
# 图片根目录
IMAGE_ROOT_DIR = "./datasets/processed_dataset/sft/images"

# 输入数据 (原始的大文件)
INPUT_JSONL = "./datasets/processed_dataset/sft/sft_data.jsonl"

# 输出数据 (采样并生成CoT后的文件)
OUTPUT_JSONL = "./datasets/processed_dataset/sft/sft_pure_text.jsonl"

# ---- Sampling ----
# 需要采样的数量
SAMPLE_SIZE = 12000
# 随机种子 (固定种子以保证每次运行选中的数据尽可能一致，或者设为 None 完全随机)
RANDOM_SEED = 42

# ---- Concurrency ----
NUM_WORKERS = 128

# ---- Generation params ----
AGENT_TEMPERATURE = 0.3
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "2048"))


# =======================
# 1) Prompt
# =======================

SYSTEM_PROMPT = """You are a specialized multimodal language model designed to solve visual question answering (VQA) tasks.

Your task is to analyze the given image and answer the question through step-by-step reasoning.

# Instructions
1. First, provide your detailed reasoning process enclosed within <think> tags.
   - Carefully analyze the visual content and the question.
   - Explain the logical steps that lead to your conclusion.

2. If you are able to determine the correct answer, provide it within <answer> tags.
   - The answer should be concise and direct.
   - Briefly summarize the key reasoning that supports the answer.

# Response Format (strict)
<think>
Step-by-step reasoning process.
</think>
<answer>
Final answer with a concise justification.
</answer>
"""


# =======================
# 2) Globals / Locks
# =======================

file_write_lock = threading.Lock()
stats_lock = threading.Lock()

global_stats = {
    "total": 0,
    "passed": 0,
    "rejected": 0
}

TAG_RE_CACHE: Dict[str, re.Pattern] = {}


# =======================
# 3) Utils
# =======================

def create_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty.")
    return OpenAI(api_key=API_KEY, base_url=API_BASE)

def call_chat(
    client: OpenAI,
    messages,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int = 3,
    sleep_s: float = 2.0
) -> Optional[str]:
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            out = resp.choices[0].message.content
            if out:
                return out
        except Exception:
            time.sleep(sleep_s)
    return None

def encode_image_to_b64(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """提取 <tag>...</tag> 之间的内容"""
    if not text:
        return None
    if tag not in TAG_RE_CACHE:
        TAG_RE_CACHE[tag] = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    
    m = TAG_RE_CACHE[tag].search(text)
    if m:
        return m.group(1).strip()
    return None

def resolve_image_path(image_filename: str) -> Optional[str]:
    basename = os.path.basename(image_filename)
    # 尝试直接拼接
    full_path = os.path.join(IMAGE_ROOT_DIR, basename)
    if os.path.exists(full_path):
        return full_path
    
    # 尝试原始路径
    full_path_2 = os.path.join(IMAGE_ROOT_DIR, image_filename)
    if os.path.exists(full_path_2):
        return full_path_2

    return None


# =======================
# 4) Per-item processing
# =======================

def process_single_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    agent_client = create_client()

    conversations = item.get("conversations", [])
    if len(conversations) < 2:
        return None

    image_filename = item.get("image", "")
    item_id = item.get("id", None)
    if item_id is None or not image_filename:
        return None

    question = conversations[0]["value"].replace("<image>", "").strip()
    # 保留原始 GT 用于参考
    gt_answer = conversations[1]["value"].strip()

    # 处理图片
    image_path = resolve_image_path(image_filename)
    if not image_path:
        return None 

    img_b64 = encode_image_to_b64(image_path)
    if not img_b64:
        return None

    # --- Construct Agent Prompt ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}
    ]

    # --- Generation ---
    response_text = call_chat(
        agent_client, 
        messages, 
        MODEL_NAME, 
        AGENT_TEMPERATURE, 
        AGENT_MAX_TOKENS
    )
    if not response_text:
        return None

    # --- Basic Format Check ---
    # 只要包含了 <answer> 标签，我们就认为数据格式可用
    pred_answer = extract_tag_content(response_text, "answer")
    is_valid_format = bool(pred_answer)

    result_status = "passed" if is_valid_format else "rejected"

    save_item = None
    if is_valid_format:
        save_item = dict(item)
        # 用新生成的 CoT 回答替换旧的回答
        save_item["conversations"] = [
            conversations[0],
            {"from": "gpt", "value": response_text}
        ]
        # 保存原始 GT (可选)
        save_item["original_gt"] = gt_answer

    return {
        "status": result_status,
        "item": save_item
    }


# =======================
# 5) Main Loop
# =======================

def load_processed_ids(output_jsonl: str) -> set:
    processed = set()
    if not os.path.exists(output_jsonl):
        return processed
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                processed.add(obj.get("id"))
            except:
                pass
    return processed


def load_and_sample_items(input_jsonl: str, skip_ids: set, sample_size: int):
    """
    读取所有数据，排除已处理的，然后进行采样
    """
    print("Reading input file...")
    candidates = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if obj.get("id") not in skip_ids:
                    candidates.append(obj)
            except:
                continue
    
    total_candidates = len(candidates)
    print(f"Total available candidates (unprocessed): {total_candidates}")
    
    if total_candidates <= sample_size:
        print(f"Candidate count ({total_candidates}) <= Sample size ({sample_size}). Using all candidates.")
        return candidates
    
    print(f"Sampling {sample_size} items from {total_candidates} candidates...")
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    
    sampled_items = random.sample(candidates, sample_size)
    return sampled_items


def process_dataset():
    # 1. 加载断点（如果之前跑过一部分）
    processed_ids = load_processed_ids(OUTPUT_JSONL)
    print(f"Resuming: {len(processed_ids)} items already processed in output file.")

    # 2. 读取并采样
    # 注意：如果这只是想跑满 12000 个，这里可以传 SAMPLE_SIZE - len(processed_ids)
    # 但为了简单起见，这里假设每次运行都是为了凑齐 SAMPLE_SIZE，如果文件里已有，会先跳过
    remaining_quota = max(0, SAMPLE_SIZE - len(processed_ids))
    
    if remaining_quota == 0:
        print("Target sample size already reached in output file.")
        return

    items_to_process = load_and_sample_items(INPUT_JSONL, processed_ids, remaining_quota)
    print(f"Tasks scheduled for this run: {len(items_to_process)}")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    f_out = open(OUTPUT_JSONL, "a", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {ex.submit(process_single_item, it): it.get("id") for it in items_to_process}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Generating CoT")

            for fut in pbar:
                try:
                    res = fut.result()
                    if not res:
                        continue

                    status = res["status"]
                    
                    with stats_lock:
                        global_stats["total"] += 1
                        if status == "passed":
                            global_stats["passed"] += 1
                        else:
                            global_stats["rejected"] += 1
                        
                        rate = 100.0 * global_stats["passed"] / max(1, global_stats["total"])
                        pbar.set_postfix({
                            "Pass": global_stats["passed"],
                            "Rej": global_stats["rejected"],
                            "Rate": f"{rate:.1f}%"
                        })

                    if status == "passed" and res.get("item"):
                        line = json.dumps(res["item"], ensure_ascii=False) + "\n"
                        with file_write_lock:
                            f_out.write(line)
                            f_out.flush()

                except Exception as e:
                    print(f"[Thread Error] {e}")

    finally:
        f_out.close()

    print("\nDone.")
    print(json.dumps(global_stats, indent=2))
    print(f"Dataset saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    process_dataset()