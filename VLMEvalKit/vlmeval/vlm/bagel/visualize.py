"""Inference process visualization for Bagel model.

Creates per-sample directories containing:
- original_XXX.png  : copies of input images
- generated_XXX.png : images produced during inference
- metadata.json     : paths, question, reasoning, answer
- summary.html      : visual summary page
"""

import os
import json
import html as html_module
from PIL import Image


def visualize_inference(messages, output_list, response, vis_dir, sample_idx=0):
    """
    Visualize a single inference sample.

    Args:
        messages: original message dicts (type="image"/"text", value=...)
        output_list: list of str / PIL.Image.Image from the inferencer
        response: final response string
        vis_dir: base directory for all visualizations
        sample_idx: running sample counter
    Returns:
        save_dir: path to this sample's visualization folder
    """
    save_dir = os.path.join(vis_dir, f"sample_{sample_idx:06d}")
    os.makedirs(save_dir, exist_ok=True)

    # ---- collect from input messages ----
    orig_image_paths = []
    question_parts = []
    for msg in messages:
        if msg["type"] == "image":
            orig_image_paths.append(msg["value"])
        elif msg["type"] == "text":
            question_parts.append(msg["value"])

    n_input_images = len(orig_image_paths)
    question_text = "\n".join(question_parts)

    # ---- copy original images ----
    saved_orig_paths = []
    for i, path in enumerate(orig_image_paths):
        try:
            img = Image.open(path).convert("RGB")
            dest = os.path.join(save_dir, f"original_{i:03d}.png")
            img.save(dest)
            saved_orig_paths.append(dest)
        except Exception:
            saved_orig_paths.append(None)

    # ---- extract generated images & reasoning from output_list ----
    generated_image_paths = []
    reasoning_texts = []
    img_counter = 0
    for item in output_list:
        if isinstance(item, Image.Image):
            if img_counter >= n_input_images:
                img_name = f"generated_{len(generated_image_paths):03d}.png"
                img_path = os.path.join(save_dir, img_name)
                item.save(img_path)
                generated_image_paths.append(img_path)
            img_counter += 1
        elif isinstance(item, str):
            reasoning_texts.append(item)

    # ---- save metadata ----
    metadata = {
        "sample_idx": sample_idx,
        "original_image_paths": orig_image_paths,
        "saved_original_copies": [p for p in saved_orig_paths if p],
        "question": question_text,
        "generated_image_paths": generated_image_paths,
        "reasoning_texts": reasoning_texts,
        "response": response,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ---- create HTML summary ----
    _create_html_summary(
        save_dir=save_dir,
        orig_image_paths=orig_image_paths,
        saved_orig_paths=saved_orig_paths,
        question_text=question_text,
        generated_image_paths=generated_image_paths,
        reasoning_texts=reasoning_texts,
        response=response,
        sample_idx=sample_idx,
    )

    return save_dir


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """\
body { font-family: "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
.section { background: #fff; border-radius: 8px; padding: 20px; margin: 16px 0;
           box-shadow: 0 2px 4px rgba(0,0,0,.1); }
.section h2 { color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 8px; }
.image-grid { display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; }
.image-card { text-align: center; }
.image-card img { max-width: 512px; max-height: 512px; border: 1px solid #ddd; border-radius: 4px; }
.image-card .path { font-size: 12px; color: #666; word-break: break-all; max-width: 512px; margin-top: 4px; }
.question { background: #E3F2FD; padding: 16px; border-radius: 8px; font-size: 15px; white-space: pre-wrap; }
.answer   { background: #E8F5E9; padding: 16px; border-radius: 8px; font-size: 15px; white-space: pre-wrap; }
.reasoning { background: #FFF3E0; padding: 16px; border-radius: 8px; font-size: 13px;
             white-space: pre-wrap; max-height: 500px; overflow-y: auto; }
h1 { color: #1565C0; }
"""


def _esc(text):
    return html_module.escape(text or "")


def _create_html_summary(
    save_dir,
    orig_image_paths,
    saved_orig_paths,
    question_text,
    generated_image_paths,
    reasoning_texts,
    response,
    sample_idx,
):
    parts = []
    parts.append(
        f'<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
        f'<title>Sample {sample_idx}</title>'
        f'<style>{_CSS}</style></head><body>'
        f'<h1>Inference Visualization &mdash; Sample {sample_idx}</h1>'
    )

    # -- Question --
    parts.append(
        f'<div class="section"><h2>Question</h2>'
        f'<div class="question">{_esc(question_text)}</div></div>'
    )

    # -- Original images --
    if orig_image_paths:
        parts.append('<div class="section"><h2>Original Images</h2><div class="image-grid">')
        for i, (orig_path, saved) in enumerate(zip(orig_image_paths, saved_orig_paths)):
            rel = os.path.basename(saved) if saved else ""
            parts.append(
                f'<div class="image-card">'
                f'<img src="{rel}" alt="Original {i}">'
                f'<div class="path">Path: {_esc(orig_path)}</div>'
                f'</div>'
            )
        parts.append('</div></div>')

    # -- Generated images --
    if generated_image_paths:
        parts.append('<div class="section"><h2>Generated Images</h2><div class="image-grid">')
        for i, gp in enumerate(generated_image_paths):
            rel = os.path.basename(gp)
            parts.append(
                f'<div class="image-card">'
                f'<img src="{rel}" alt="Generated {i}">'
                f'<div class="path">{rel}</div>'
                f'</div>'
            )
        parts.append('</div></div>')

    # -- Reasoning --
    # Filter out very short system-prompt-like strings
    display_texts = [t for t in reasoning_texts if len(t) > 30]
    if display_texts:
        combined = "\n\n--- step ---\n\n".join(display_texts)
        parts.append(
            f'<div class="section"><h2>Reasoning Process</h2>'
            f'<div class="reasoning">{_esc(combined)}</div></div>'
        )

    # -- Answer --
    parts.append(
        f'<div class="section"><h2>Answer</h2>'
        f'<div class="answer">{_esc(response) if response else "(No answer)"}</div></div>'
    )

    parts.append('</body></html>')

    with open(os.path.join(save_dir, "summary.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
