import os
import argparse
import base64
import io
import math
import random
import glob
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import csv

# -----------------------------
# 1. 图像处理配置
# -----------------------------
CONFIG = {
    "motion_blur": {"ksize": 15},                
    "pepper":      0.15,                         
    "mask":        {"block_size": 12, "mask_ratio": 0.30}, 
    "darken":      0.30                          
}

# 增加 CSV 字段长度限制，防止 Base64 过长报错
csv.field_size_limit(sys.maxsize)

# -----------------------------
# 2. 编解码工具
# -----------------------------
def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    return ret

def decode_base64_to_image(base64_string, target_size=-1):
    try:
        if pd.isna(base64_string) or base64_string == "":
            return None
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        if target_size > 0:
            image.thumbnail((target_size, target_size))
        return image
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

def apply_random_degradation(pil_img, seed=None):
    if pil_img is None:
        return None
    img_np = np.array(pil_img)
    
    # 设置随机种子，保证可复现
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
# 4. Worker (纯内存处理)
# -----------------------------
def process_single_base64(index, original_b64, seed_base):
    """
    输入: index, base64字符串
    输出: (index, new_base64_string)
    """
    try:
        # 解码
        img = decode_base64_to_image(original_b64)
        if img is None:
            return index, original_b64 # 失败则返回原图

        # 处理
        item_seed = seed_base + int(index) if seed_base else None
        processed_img = apply_random_degradation(img, seed=item_seed)

        # 编码
        new_b64 = encode_image_to_base64(processed_img)
        return index, new_b64

    except Exception:
        # 如果出错，返回原始数据，保证程序不崩且数据不丢
        return index, original_b64

# -----------------------------
# 5. 主逻辑
# -----------------------------
def process_tsv(tsv_path, seed):
    filename = os.path.basename(tsv_path)
    print(f"Processing: {filename}")
    
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        print(f"  [Error] Cannot read {filename}: {e}")
        return

    if 'image' not in df.columns:
        print(f"  [Skip] No 'image' column in {filename}")
        return

    # 并行处理
    results = {}
    with ProcessPoolExecutor() as executor:
        # 提交任务
        futures = {
            executor.submit(process_single_base64, row['index'], row['image'], seed): row['index']
            for _, row in df.iterrows()
        }
        
        # 进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Rows"):
            idx, new_b64 = future.result()
            results[idx] = new_b64

    # 更新 DataFrame (只更新 image 列)
    # 保持原来的顺序
    new_image_col = []
    for _, row in df.iterrows():
        idx = row['index']
        # 获取新结果，如果没在results里(理论上不可能)则用原值
        new_image_col.append(results.get(idx, row['image']))

    df['image'] = new_image_col

    # 保存为 _LOW_LEVEL.tsv
    name_no_ext = os.path.splitext(tsv_path)[0]
    new_path = f"{name_no_ext}_LOW_LEVEL.tsv"
    
    # 关键: index=False (不加行号), sep='\t' (保持TSV)
    df.to_csv(new_path, sep='\t', index=False)
    print(f"  [Done] Saved to: {new_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/root/LMUData", help="Directory containing original .tsv files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: {args.input_dir} does not exist.")
        return

    # 扫描目录下所有的 .tsv 文件 (排除已经是 _LOW_LEVEL 的文件，防止重复处理)
    all_tsvs = glob.glob(os.path.join(args.input_dir, "*.tsv"))
    target_tsvs = [f for f in all_tsvs if "_LOW_LEVEL.tsv" not in f]

    if not target_tsvs:
        print("No .tsv files found.")
        return

    print(f"Found {len(target_tsvs)} files to process.")
    for tsv_path in target_tsvs:
        process_tsv(tsv_path, args.seed)
        print("-" * 40)

if __name__ == "__main__":
    main()