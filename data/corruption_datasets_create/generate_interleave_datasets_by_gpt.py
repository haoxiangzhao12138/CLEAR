import os
import json
import base64
import time
import re
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, Tuple

from openai import OpenAI
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =======================
# 0) Config
# =======================

# ---- API ----
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "gpt-4.1")

# ---- Paths ----
CLEAN_IMAGE_DIR = "./datasets/processed_dataset/sft/images"
CORRUPTED_IMAGE_DIR = "./datasets/processed_dataset/sft/corruption_images"
INPUT_JSONL = "./datasets/processed_dataset/sft/sft_data.jsonl"
OUTPUT_JSONL = "./datasets/processed_dataset/sft/agent_interleave_data_filtered.jsonl"
OUTPUT_JSONL_TOOL = "./datasets/processed_dataset/sft/agent_interleave_data_filtered_tool.jsonl"
OUTPUT_JSONL_NO_TOOL = "./datasets/processed_dataset/sft/agent_interleave_data_filtered_no_tool.jsonl"

# ---- Concurrency ----
NUM_WORKERS = 190

# ---- Generation params ----
AGENT_TEMPERATURE = 0.3
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "1024"))

# ---- Judge params ----
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 128

# ---- Visualization ----
VISUALIZE_MODE = False
VISUALIZE_DIR = "./visualizations/debug_output_interleave"
MAX_VISUALIZE_COUNT = 20  # -1 for unlimited

# =======================
# 1) Prompt
# =======================

SYSTEM_PROMPT = """You are a multimodal reasoning agent. You will be given an image (possibly degraded) and a question.
Your goal is to produce a correct answer grounded in visual evidence.

You may optionally request a restored version of the image before answering.

# Action Token
To request a restored image, output the single token:
<image_restore>

After you output <image_restore>, the next message will contain a restored image.
Then you must continue reasoning based on that restored image and provide the final answer.

# Output Format (STRICT)
- All reasoning must be enclosed strictly within <think>...</think>.
- All final answers must be enclosed strictly within <answer>...</answer>.
- Do NOT put reasoning inside <answer>.
- Do NOT output any content outside these tags, except the single token <image_restore>.
- Do NOT mention prompts, policies, tools, APIs, or capability comparisons.

# Valid Patterns (ONLY these two)

Pattern A (no restoration):
<think> ... </think>
<answer> ... </answer>

Pattern B (restore then answer):
<think> ... </think>
<image_restore>
<think> ... </think>
<answer> ... </answer>

# Special rule for restoration
- If you request restoration, output <image_restore> on its own line.
- In that turn, do NOT output <answer>.
- Do NOT request restoration more than once.

# Requirements for <think>
Your <think> must be specific to the given image and question and include:
1) Image condition diagnosis: what degradations affect the relevant region(s) for this question.
2) Evidence requirement: what visual evidence is needed (text/digits/count/attributes/relations).
3) Reasoning steps as needed to reach the answer (even for clear images).
4) A decision: request restoration only if key evidence is unclear/ambiguous.

# No Guessing
If required evidence is unclear, do not guess.
If evidence remains unclear even after receiving a restored image, provide the best supported partial answer and state what cannot be determined.
"""


# =======================
# 2) Globals / Locks / Stats
# =======================

file_write_lock = threading.Lock()
stats_lock = threading.Lock()
visualize_lock = threading.Lock()

global_stats = {
    "total": 0,
    "passed": 0,
    "rejected": 0,
    "tool_used": 0,
    "vis_saved": 0
}

TAG_RE_CACHE: Dict[str, re.Pattern] = {}


# =======================
# 3) Utils
# =======================

def sanitize_filename(name: str, max_len: int = 180) -> str:
    """
    Make a safe filename:
    - remove path separators and illegal characters
    - keep it short to avoid OS/path limits
    """
    if name is None:
        name = "none"
    name = str(name)

    # Replace path separators explicitly
    name = name.replace("/", "_").replace("\\", "_")

    # Replace other problematic chars (including commas) with underscore
    name = re.sub(r'[^0-9a-zA-Z._-]+', "_", name)

    # Trim repeated underscores
    name = re.sub(r"_+", "_", name).strip("_")

    # Keep tail for uniqueness if too long
    if len(name) > max_len:
        name = name[-max_len:]

    return name or "empty"

def create_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty. Please export OPENAI_API_KEY before running.")
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
    if not text:
        return None
    if tag not in TAG_RE_CACHE:
        TAG_RE_CACHE[tag] = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    m = TAG_RE_CACHE[tag].search(text)
    return m.group(1).strip() if m else None


def has_image_restore_token(text: str) -> bool:
    if not text:
        return False
    return re.search(r'(^|\n)\s*<image_restore>\s*(\n|$)', text) is not None


def resolve_paths(image_filename: str) -> Tuple[Optional[str], Optional[str], str]:
    basename = os.path.basename(image_filename)

    clean = os.path.join(CLEAN_IMAGE_DIR, basename)
    corr = os.path.join(CORRUPTED_IMAGE_DIR, basename)
    if os.path.exists(clean) and os.path.exists(corr):
        return clean, corr, basename

    clean = os.path.join(CLEAN_IMAGE_DIR, image_filename)
    corr = os.path.join(CORRUPTED_IMAGE_DIR, image_filename)
    if os.path.exists(clean) and os.path.exists(corr):
        return clean, corr, basename

    return None, None, basename


def evaluate_consistency(
    client: OpenAI,
    question: str,
    gt_answer: str,
    pred_answer: str
) -> Tuple[bool, str]:
    if not pred_answer:
        return False, "No <answer> content"

    prompt = (
        "You are an objective judge.\n"
        f'Question: "{question}"\n'
        f'Ground Truth Answer: "{gt_answer}"\n'
        f'Model Prediction: "{pred_answer}"\n\n'
        "Task: Determine if the Model Prediction is semantically consistent with the Ground Truth Answer. "
        "Ignore minor phrasing differences.\n\n"
        'Respond ONLY with a JSON object: {"consistent": true} or {"consistent": false}.'
    )

    messages = [{"role": "user", "content": prompt}]
    resp = call_chat(
        client,
        messages,
        model=JUDGE_MODEL_NAME,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
        retries=3,
    )
    if not resp:
        return False, "Judge no response"

    cleaned = resp.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(cleaned)
        return bool(obj.get("consistent", False)), cleaned
    except Exception:
        return False, cleaned


def save_visualization_big(
    item_id: Any,
    corr_path: str,
    clean_path: str,
    question: str,
    turn1: str,
    turn2: Optional[str],
    gt_answer: str,
    pred_answer: str,
    passed: bool
):
    if not VISUALIZE_MODE:
        return

    os.makedirs(VISUALIZE_DIR, exist_ok=True)

    # matplotlib can crash under multithreading, use a lock
    with visualize_lock:
        fig = plt.figure(figsize=(26, 14))

        # images
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(Image.open(corr_path))
        ax1.set_title("Input (Corrupted)", fontsize=14)
        ax1.axis("off")

        ax2 = fig.add_subplot(2, 2, 3)
        ax2.imshow(Image.open(clean_path))
        ax2.set_title("Reference (Clean / Simulated Restore Result)", fontsize=14)
        ax2.axis("off")

        # text panel
        ax3 = fig.add_subplot(1, 2, 2)
        ax3.axis("off")

        lines = []
        lines.append(f"ID: {item_id}   STATUS: {'PASSED' if passed else 'REJECTED'}")
        lines.append("=" * 80)
        lines.append("QUESTION:")
        lines.append(question)
        lines.append("-" * 80)
        lines.append("TURN 1 OUTPUT:")
        lines.append(turn1)
        if turn2:
            lines.append("-" * 80)
            lines.append("TURN 2 OUTPUT:")
            lines.append(turn2)
        lines.append("=" * 80)
        lines.append("GROUND TRUTH:")
        lines.append(gt_answer)
        lines.append("-" * 80)
        lines.append("PREDICTION (<answer>):")
        lines.append(pred_answer if pred_answer else "N/A")

        full_text = "\n".join(lines)

        wrapper = textwrap.TextWrapper(width=110, replace_whitespace=False, drop_whitespace=False)
        wrapped = []
        for ln in full_text.split("\n"):
            if ln.startswith("=") or ln.startswith("-"):
                wrapped.append(ln)
            else:
                wrapped.extend(wrapper.wrap(ln) if ln else [""])
        final_text = "\n".join(wrapped)

        ax3.text(0, 1, final_text, va="top", ha="left", fontsize=10.5, family="monospace")

        fig.suptitle(f"Consistency: {passed}", fontsize=18)
        safe_id = sanitize_filename(item_id)
        out_name = f"{safe_id}_{'PASS' if passed else 'REJ'}.jpg"
        out_path = os.path.join(VISUALIZE_DIR, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close(fig)


# =======================
# 4) Per-item processing
# =======================

def process_single_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    agent_client = create_client()
    judge_client = create_client()

    conversations = item.get("conversations", [])
    if len(conversations) < 2:
        return None

    image_filename = item.get("image", "")
    item_id = item.get("id", None)
    if item_id is None or not image_filename:
        return None

    question = conversations[0]["value"].replace("<image>", "").strip()
    gt_answer = conversations[1]["value"].strip()

    clean_path, corr_path, _ = resolve_paths(image_filename)
    if not clean_path or not corr_path:
        return None

    clean_b64 = encode_image_to_b64(clean_path)
    corr_b64 = encode_image_to_b64(corr_path)
    if not clean_b64 or not corr_b64:
        return None

    # --- Turn 1 ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{corr_b64}"}}
        ]}
    ]

    resp1 = call_chat(agent_client, messages, MODEL_NAME, AGENT_TEMPERATURE, AGENT_MAX_TOKENS)
    if not resp1:
        return None

    tool_called = has_image_restore_token(resp1)

    messages.append({"role": "assistant", "content": resp1})

    resp2 = None
    final_text_for_eval = resp1

    # --- If restore requested, inject clean image as "restored" image and do Turn 2 ---
    if tool_called:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"}}
            ]
        })

        resp2 = call_chat(agent_client, messages, MODEL_NAME, AGENT_TEMPERATURE, AGENT_MAX_TOKENS)
        if not resp2:
            return None

        messages.append({"role": "assistant", "content": resp2})
        final_text_for_eval = resp2

    # --- Evaluate only by GT consistency (no extra filtering) ---
    pred_answer = extract_tag_content(final_text_for_eval, "answer") or ""
    is_consistent, judge_raw = evaluate_consistency(judge_client, question, gt_answer, pred_answer)

    if not is_consistent:
        # Visualization can also store some rejected items (if desired)
        return {
            "status": "rejected",
            "id": item_id,
            "tool_used": tool_called,
            "viz": {
                "id": item_id,
                "corr_path": corr_path,
                "clean_path": clean_path,
                "question": question,
                "turn1": resp1,
                "turn2": resp2,
                "gt": gt_answer,
                "pred": pred_answer,
                "passed": False
            } if VISUALIZE_MODE else None
        }

    # --- Merge conversation for training ---
    if tool_called and resp2:
        merged_value = f"{resp1}\n{resp2}"
    else:
        merged_value = resp1

    save_item = dict(item)
    save_item["conversations"] = [
        conversations[0],
        {"from": "gpt", "value": merged_value}
    ]
    save_item["tool_used"] = tool_called

    return {
        "status": "passed",
        "id": item_id,
        "tool_used": tool_called,
        "item": save_item,
        "viz": {
            "id": item_id,
            "corr_path": corr_path,
            "clean_path": clean_path,
            "question": question,
            "turn1": resp1,
            "turn2": resp2,
            "gt": gt_answer,
            "pred": pred_answer,
            "passed": True
        } if VISUALIZE_MODE else None
    }


# =======================
# 5) Main
# =======================

def load_processed_ids(output_jsonl: str) -> set:
    processed = set()
    if not os.path.exists(output_jsonl):
        return processed
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                processed.add(obj.get("id"))
            except Exception:
                pass
    return processed


def iter_input_items(input_jsonl: str, skip_ids: set):
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("id") not in skip_ids:
                    yield obj
            except Exception:
                continue


def process_dataset_multithreaded():
    # Recommendation: use the union of both output files for skip_ids to avoid reprocessing
    processed_ids = load_processed_ids(OUTPUT_JSONL_TOOL) | load_processed_ids(OUTPUT_JSONL_NO_TOOL)
    print(f"Resuming: already have {len(processed_ids)} items in output(s).")

    items = list(iter_input_items(INPUT_JSONL, processed_ids))
    print(f"Tasks remaining: {len(items)}")

    os.makedirs(os.path.dirname(OUTPUT_JSONL_TOOL), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSONL_NO_TOOL), exist_ok=True)
    if VISUALIZE_MODE:
        os.makedirs(VISUALIZE_DIR, exist_ok=True)

    out_tool = open(OUTPUT_JSONL_TOOL, "a", encoding="utf-8")
    out_no_tool = open(OUTPUT_JSONL_NO_TOOL, "a", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {ex.submit(process_single_item, it): it.get("id") for it in items}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Building")

            for fut in pbar:
                try:
                    res = fut.result()
                    if not res:
                        continue

                    # update stats (unchanged)
                    with stats_lock:
                        global_stats["total"] += 1
                        if res["status"] == "passed":
                            global_stats["passed"] += 1
                        else:
                            global_stats["rejected"] += 1
                        if res.get("tool_used"):
                            global_stats["tool_used"] += 1

                        rate = 100.0 * global_stats["passed"] / max(1, global_stats["total"])
                        tool_rate = 100.0 * global_stats["tool_used"] / max(1, global_stats["total"])
                        pbar.set_postfix({
                            "Pass": global_stats["passed"],
                            "Rej": global_stats["rejected"],
                            "PassRate": f"{rate:.1f}%",
                            "ToolRate": f"{tool_rate:.1f}%"
                        })

                    # Write passed item: split by tool_used
                    if res["status"] == "passed" and res.get("item"):
                        line = json.dumps(res["item"], ensure_ascii=False) + "\n"
                        with file_write_lock:
                            if res.get("tool_used"):
                                out_tool.write(line)
                                out_tool.flush()
                            else:
                                out_no_tool.write(line)
                                out_no_tool.flush()

                    # visualization (unchanged)
                    if VISUALIZE_MODE and res.get("viz"):
                        should_viz = False
                        with stats_lock:
                            if MAX_VISUALIZE_COUNT == -1 or global_stats["vis_saved"] < MAX_VISUALIZE_COUNT:
                                global_stats["vis_saved"] += 1
                                should_viz = True

                        if should_viz:
                            v = res["viz"]
                            save_visualization_big(
                                item_id=v["id"],
                                corr_path=v["corr_path"],
                                clean_path=v["clean_path"],
                                question=v["question"],
                                turn1=v["turn1"],
                                turn2=v["turn2"],
                                gt_answer=v["gt"],
                                pred_answer=v["pred"],
                                passed=v["passed"]
                            )

                except Exception as e:
                    print(f"[Thread Error] {e}")

    finally:
        out_tool.close()
        out_no_tool.close()

    print("\nDone.")
    print(json.dumps(global_stats, indent=2, ensure_ascii=False))
    if VISUALIZE_MODE:
        print(f"Visualizations saved to: {VISUALIZE_DIR}")
    print(f"Passed(tool) saved to: {OUTPUT_JSONL_TOOL}")
    print(f"Passed(no-tool) saved to: {OUTPUT_JSONL_NO_TOOL}")


if __name__ == "__main__":
    process_dataset_multithreaded()