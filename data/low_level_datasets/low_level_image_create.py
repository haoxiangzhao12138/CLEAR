#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Configuration (High Level Only)
# -----------------------------
CONFIG = {
    "motion_blur": {"ksize": 15},                # High
    "pepper":      0.15,                         # High
    "mask":        {"block_size": 12, "mask_ratio": 0.30}, # High
    "darken":      0.40                          # High (retain 40% brightness)
}

# -----------------------------
# Utils
# -----------------------------
def load_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def save_image(path: Path, img_uint8: np.ndarray, skip_exist: bool = True):
    if skip_exist and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_uint8).save(str(path))

def list_images(root: Path, exts):
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext.lower()}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(files))

# -----------------------------
# Degradation Ops
# -----------------------------

def _motion_kernel(ksize, angle_deg):
    ker = np.zeros((ksize, ksize), np.float32)
    c = ksize // 2
    ang = math.radians(angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    for i in range(ksize):
        t = i - c
        x = c + int(round(t * ca))
        y = c + int(round(t * sa))
        if 0 <= x < ksize and 0 <= y < ksize:
            ker[y, x] = 1.0
    s = ker.sum()
    ker /= (s if s > 0 else 1.0)
    if s == 0: ker[c, c] = 1.0
    return ker

def add_motion_blur(img_u8, ksize):
    if ksize <= 1:
        return img_u8.copy()
    angle = random.uniform(0, 360)
    kernel = _motion_kernel(int(ksize), angle)
    return cv2.filter2D(img_u8, -1, kernel, borderType=cv2.BORDER_REFLECT101)

def add_pepper_noise(img_u8, p):
    h, w, _ = img_u8.shape
    n = int(p * h * w)
    if n <= 0: return img_u8.copy()
    out = img_u8.copy()
    ys = np.random.randint(0, h, size=n)
    xs = np.random.randint(0, w, size=n)
    out[ys, xs] = 0
    return out

def random_mask_blocks(img_u8, block_size=4, mask_ratio=0.1):
    h, w, _ = img_u8.shape
    out = img_u8.copy()
    bh = (h + block_size - 1)//block_size
    bw = (w + block_size - 1)//block_size
    total = bh * bw
    m = int(mask_ratio * total)
    if m <= 0: return out
    idxs = np.random.choice(total, size=m, replace=False)
    for idx in idxs:
        by, bx = divmod(idx, bw)
        y0, x0 = by*block_size, bx*block_size
        y1, x1 = min(y0+block_size, h), min(x0+block_size, w)
        out[y0:y1, x0:x1] = 0
    return out

def add_darken(img_u8, factor):
    f = float(factor)
    out = np.clip((img_u8.astype(np.float32) / 255.0) * f, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)

# -----------------------------
# Processing Pipeline
# -----------------------------
def process_one(img_path: Path, in_root: Path, out_root: Path, skip_exist: bool):
    rel = img_path.relative_to(in_root) if in_root else Path(img_path.name)
    save_path = out_root / rel
    
    if skip_exist and save_path.exists():
        return

    img = load_image(img_path)
    if img is None:
        return

    # -------------------------------------------------
    # 随机选择一种退化 (Random Choice)
    # -------------------------------------------------
    options = ["motion_blur", "darken", "pepper", "mask"]
    choice = random.choice(options)

    if choice == "motion_blur":
        img = add_motion_blur(img, CONFIG["motion_blur"]["ksize"])
    
    elif choice == "darken":
        img = add_darken(img, CONFIG["darken"])
    
    elif choice == "pepper":
        img = add_pepper_noise(img, CONFIG["pepper"])
        
    elif choice == "mask":
        img = random_mask_blocks(img, **CONFIG["mask"])

    # 保存
    save_image(save_path, img, skip_exist=False)

# -----------------------------
# Worker
# -----------------------------
def _worker(p_str, in_root_str, out_root_str, seed=None, skip_exist=True):
    try:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        if seed is not None:
            # 这里的 random.seed 保证了同一张图片如果多次运行（且种子相同），
            # 每次都会随机选中同一种退化方式，便于实验复现。
            random.seed(seed ^ (hash(p_str) & 0xFFFFFFFF))
            np.random.seed(seed ^ (hash((p_str, "np")) & 0xFFFFFFFF))
            
        in_root  = Path(in_root_str) if in_root_str else None
        out_root = Path(out_root_str)
        
        process_one(Path(p_str), in_root, out_root, skip_exist=skip_exist)
        return (True, p_str, "")
    except Exception as e:
        return (False, p_str, str(e))

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Generate Randomly Selected High-Level Degraded Images")
    ap.add_argument("--input_dir", type=str, default="/home/haoxiangzhao/Bagel/datasets/mme/origin",
                    help="Input origin directory")
    ap.add_argument("--output", type=str, default="/home/haoxiangzhao/Bagel/datasets/mme/mixed_random_high",
                    help="Output directory")
    ap.add_argument("--exts", type=str, nargs="+", default=[".jpg",".jpeg",".png"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--skip_exist", action="store_true", default=True)
    args = ap.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output)
    
    if not in_root.exists():
        raise FileNotFoundError(f"input_dir not found: {in_root}")
    
    out_root.mkdir(parents=True, exist_ok=True)

    files = list_images(in_root, args.exts)
    if not files:
        print("[WARN] no images found in", in_root)
        return

    num_workers = os.cpu_count() if args.workers in (-1, 0, None) else max(1, int(args.workers))

    print(f"Start processing {len(files)} images...")
    print(f"Mode: Randomly select ONE from [Motion, Darken, Pepper, Mask] (High Level)")
    print(f"Output: {out_root}")

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = [ex.submit(_worker, str(p), str(in_root), str(out_root), args.seed, args.skip_exist) for p in files]
        
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"Processing"):
            ok, path_str, err = f.result()
            if not ok:
                print(f"[WARN] Failed on {path_str}: {err}")

    print(f"[DONE] Output saved to {out_root}")

if __name__ == "__main__":
    main()