import os
import argparse
import base64
import io
import math
import random
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import csv
import ast

# -----------------------------
# 1. 图像处理配置
# -----------------------------
CONFIG = {
    "motion_blur": {"ksize": 15},                
    "pepper":      0.15,                         
    "mask":        {"block_size": 12, "mask_ratio": 0.30}, 
    "darken":      0.30                          
}

# 增加 CSV 字段长度限制，防止超大 Base64 报错
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)

# -----------------------------
# 2. 基础编解码工具
# -----------------------------
def pil_to_b64(img, quality=95):
    """将 PIL 图片转为 Base64 字符串"""
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG', quality=quality)
    return base64.b64encode(img_buffer.getvalue()).decode('utf-8')

def b64_to_pil(b64_str):
    """将单个 Base64 字符串转为 PIL 图片"""
    try:
        if not b64_str or pd.isna(b64_str): return None
        b64_str = str(b64_str).strip()
        # 容错处理：如果单个字符串里意外混入了引号
        if b64_str.startswith("'") and b64_str.endswith("'"):
            b64_str = b64_str[1:-1]
        
        image_data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception:
        return None

# -----------------------------
# 3. 图像退化算法
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
    if ksize <= 1: return img_u8.copy()
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

def apply_random_degradation(pil_img, seed=None):
    if pil_img is None: return None
    img_np = np.array(pil_img)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    options = ["motion_blur", "darken", "pepper", "mask"]
    choice = random.choice(options)
    if choice == "motion_blur":
        img_np = add_motion_blur(img_np, CONFIG["motion_blur"]["ksize"])
    elif choice == "darken":
        img_np = add_darken(img_np, CONFIG["darken"])
    elif choice == "pepper":
        img_np = add_pepper_noise(img_np, CONFIG["pepper"])
    elif choice == "mask":
        img_np = random_mask_blocks(img_np, **CONFIG["mask"])
    return Image.fromarray(img_np)

# -----------------------------
# 4. 核心处理逻辑 (修正版：同步处理 image 和 image_path)
# -----------------------------
def process_row_images(index, raw_image_str, raw_path_str, seed_base):
    """
    同时处理 image (Base64) 和 image_path (文件名)
    确保两者列表长度一致，并修改文件名以防止缓存命中。
    """
    # --- 1. 解析 image 字符串 ---
    try:
        img_list = ast.literal_eval(raw_image_str)
        if not isinstance(img_list, list): img_list = [raw_image_str]
    except:
        img_list = [raw_image_str]

    # --- 2. 解析 image_path 字符串 ---
    try:
        path_list = ast.literal_eval(raw_path_str)
        if not isinstance(path_list, list): path_list = [raw_path_str]
    except:
        path_list = [raw_path_str]

    # 安全性检查：如果 path 不够用，自动补全（防止 crash，虽然理论上数据应该对齐）
    while len(path_list) < len(img_list):
        path_list.append(f"unknown_{index}_{len(path_list)}.jpg")

    new_img_list = []
    new_path_list = []

    # --- 3. 遍历处理 ---
    for i, b64_str in enumerate(img_list):
        # 3.1 处理文件名：必须改名！
        original_fname = path_list[i]
        fname_root, fname_ext = os.path.splitext(original_fname)
        # 生成新文件名，加上 _LOW 标识
        new_filename = f"{fname_root}_LOW{fname_ext}"
        new_path_list.append(new_filename)

        # 3.2 处理图片内容
        try:
            pil_img = b64_to_pil(b64_str)
            if pil_img is None:
                new_img_list.append(b64_str) # 解码失败保留原样
                continue
            
            # 独立随机种子
            item_seed = seed_base + int(index) + i if seed_base else None
            processed_pil = apply_random_degradation(pil_img, seed=item_seed)
            new_b64 = pil_to_b64(processed_pil, quality=95)
            new_img_list.append(new_b64)
            
        except Exception:
            # 任何处理异常，保留原数据，但必须使用新文件名（避免列表长度不一致）
            new_img_list.append(b64_str)

    # --- 4. 重新打包成字符串列表 ---
    return index, str(new_img_list), str(new_path_list)

# -----------------------------
# 5. 主程序
# -----------------------------
def process_tsv(tsv_path, seed):
    filename = os.path.basename(tsv_path)
    print(f"Processing: {filename} (Seed: {seed})")
    
    # 使用 dtype=object 防止 pandas 自动推断类型导致精度丢失或格式错误
    try:
        df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
    except Exception as e:
        print(f"  [Error] Cannot read {filename}: {e}")
        return

    # 检查必要列
    if 'image' not in df.columns:
        print(f"  [Skip] No 'image' column in {filename}")
        return
    
    # 兼容性处理：如果没有 image_path 列，先创建一个基于 index 的伪列
    # 这主要是为了兼容 MME 等单图数据集，虽然 MME 不强制依赖 image_path，但有了它逻辑更统一
    if 'image_path' not in df.columns:
        print("  [Info] 'image_path' column missing. Generating default paths based on index...")
        df['image_path'] = df['index'].apply(lambda x: f"{x}.jpg")

    results_img = {}
    results_path = {}
    
    print(f"  Start processing {len(df)} rows...")
    
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_row_images, row['index'], row['image'], row['image_path'], seed): row['index']
            for _, row in df.iterrows()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Degrading"):
            idx, new_imgs, new_paths = future.result()
            results_img[idx] = new_imgs
            results_path[idx] = new_paths

    # 更新 DataFrame
    new_image_col = []
    new_path_col = []
    
    for _, row in df.iterrows():
        idx = row['index']
        new_image_col.append(results_img.get(idx, row['image']))
        new_path_col.append(results_path.get(idx, row['image_path']))

    df['image'] = new_image_col
    df['image_path'] = new_path_col

    # 生成新文件名
    name_no_ext = os.path.splitext(tsv_path)[0]
    if "LOW_LEVEL" in name_no_ext:
        new_path = tsv_path # 如果已经是低质文件，覆盖
    else:
        new_path = f"{name_no_ext}_LOW_LEVEL.tsv"
    
    # QUOTE_ALL 至关重要，防止 list 字符串里的逗号破坏 tsv 结构
    df.to_csv(new_path, sep='\t', index=False, quoting=csv.QUOTE_ALL)
    print(f"  [Success] Saved to: {new_path}")
    print(f"  [Note] Ensure you clear VLMEvalKit cache or expect new images at {os.path.basename(new_path)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/root/CLEAR/LMUData/BLINK.tsv", help="Path to input .tsv file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found.")
        return

    process_tsv(args.input_file, args.seed)

if __name__ == "__main__":
    main()