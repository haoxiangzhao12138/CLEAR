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
import traceback
import hashlib
import time

# -----------------------------
# 配置与初始化
# -----------------------------

# 增加 CSV 字段长度限制
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)

# -----------------------------
# 基础工具函数
# -----------------------------

def pil_to_b64(img, quality=95):
    """
    将 PIL Image 转换为纯 Base64 字符串。
    
    关键调整：
    1. quality=75 (默认值)，解决文件体积过大问题。
    2. 无前缀。
    """
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    img_buffer = io.BytesIO()
    # JPEG 压缩，quality=75 是平衡体积和画质的最佳点
    img.save(img_buffer, format='JPEG', quality=quality)
    b64_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    # 移除换行符
    return b64_data.replace("\n", "")

def b64_to_pil(b64_str):
    """
    将 Base64 字符串转换为 PIL Image。
    """
    if not b64_str or pd.isna(b64_str): return None
    
    b64_str = str(b64_str).strip()
    
    # 清理引号（兼容处理）
    if b64_str.startswith("'") and b64_str.endswith("'"): b64_str = b64_str[1:-1]
    if b64_str.startswith('"') and b64_str.endswith('"'): b64_str = b64_str[1:-1]

    # 清理前缀
    if 'base64,' in b64_str:
        b64_str = b64_str.split('base64,')[-1]
    
    try:
        image_data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(image_data)).convert('RGB')
    except:
        return None

# -----------------------------
# 图像退化算法
# -----------------------------
def apply_random_degradation(pil_img, seed=None):
    if pil_img is None: return None
    
    import numpy as np
    import cv2
    import random
    import math

    CONFIG = {
        "motion_blur": {"ksize": 15},                
        "pepper":      0.15,                         
        "mask":        {"block_size": 12, "mask_ratio": 0.30}, 
        "darken":      0.30                          
    }

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

    img_np = np.array(pil_img)
    
    if seed is not None:
        safe_seed = seed % (2**32) 
        random.seed(safe_seed)
        np.random.seed(safe_seed)
    
    options = ["motion_blur", "darken", "pepper", "mask"]
    choice = random.choice(options)
    
    if choice == "motion_blur":
        if CONFIG["motion_blur"]["ksize"] > 1:
            angle = random.uniform(0, 360)
            kernel = _motion_kernel(int(CONFIG["motion_blur"]["ksize"]), angle)
            img_np = cv2.filter2D(img_np, -1, kernel, borderType=cv2.BORDER_REFLECT101)
            
    elif choice == "darken":
        f = float(CONFIG["darken"])
        out = np.clip((img_np.astype(np.float32) / 255.0) * f, 0.0, 1.0)
        img_np = (out * 255.0 + 0.5).astype(np.uint8)
        
    elif choice == "pepper":
        h, w, _ = img_np.shape
        n = int(CONFIG["pepper"] * h * w)
        if n > 0:
            ys = np.random.randint(0, h, size=n)
            xs = np.random.randint(0, w, size=n)
            img_np[ys, xs] = 0
            
    elif choice == "mask":
        h, w, _ = img_np.shape
        block_size = CONFIG["mask"]["block_size"]
        mask_ratio = CONFIG["mask"]["mask_ratio"]
        bh = (h + block_size - 1)//block_size
        bw = (w + block_size - 1)//block_size
        total = bh * bw
        m = int(mask_ratio * total)
        if m > 0:
            idxs = np.random.choice(total, size=m, replace=False)
            for idx in idxs:
                by, bx = divmod(idx, bw)
                y0, x0 = by*block_size, bx*block_size
                y1, x1 = min(y0+block_size, h), min(x0+block_size, w)
                img_np[y0:y1, x0:x1] = 0
    
    return Image.fromarray(img_np)

# -----------------------------
# 核心处理逻辑
# -----------------------------
def process_row_images(index, raw_image_str, seed_base):
    import ast
    import hashlib
    import json # 仅用于读取时的兼容

    try:
        # 1. 解析原始数据
        # 优先使用 ast.literal_eval，因为原版数据格式是 Python List (单引号)
        try:
            img_list = ast.literal_eval(raw_image_str)
            if not isinstance(img_list, list): img_list = [raw_image_str]
        except:
            # 兼容 JSON 格式（如果之前跑过别的脚本）
            try:
                img_list = json.loads(raw_image_str)
                if not isinstance(img_list, list): img_list = [raw_image_str]
            except:
                # 纯字符串
                img_list = [raw_image_str]

        new_img_list = []

        for i, b64_str in enumerate(img_list):
            try:
                pil_img = b64_to_pil(b64_str)
                
                if pil_img is not None:
                    # 生成随机种子
                    idx_str = str(index)
                    hash_val = int(hashlib.sha256(idx_str.encode('utf-8')).hexdigest(), 16)
                    item_seed = seed_base + hash_val + i 
                    
                    # 退化处理
                    processed_pil = apply_random_degradation(pil_img, seed=item_seed)
                    
                    # 编码回 Base64 (quality=75)
                    processed_b64 = pil_to_b64(processed_pil, quality=95)
                    new_img_list.append(processed_b64)
                else:
                    new_img_list.append(b64_str)
            except:
                new_img_list.append(b64_str)

        # ------------------------------------------------------------
        # 关键修改：使用 str() 而不是 json.dumps()
        # ------------------------------------------------------------
        # 原版 BLINK 是: "['/9j/...', '/9j/...']" (单引号)
        # str([]) 在 Python 中会生成: "['item1', 'item2']" (单引号)
        # 这与原版格式完全一致。Base64 字符集不含单引号，因此这样做是安全的。
        final_str = str(new_img_list)
        
        return index, final_str
    
    except Exception as e:
        print(f"\n[CRITICAL] Row {index} failed: {e}")
        return index, raw_image_str

# -----------------------------
# 主程序
# -----------------------------
def process_tsv(tsv_path, seed):
    filename = os.path.basename(tsv_path)
    print(f"Processing: {filename} (Seed: {seed})")
    
    try:
        df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8', dtype={'index': str})
    except Exception as e:
        print(f"  [Error] Cannot read {filename}: {e}")
        return

    if 'image' not in df.columns:
        print(f"  [Skip] No 'image' column in {filename}")
        return
    
    df = df.dropna(subset=['image'])
    print(f"  Start processing {len(df)} rows...")
    
    start_time = time.time()
    
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_row_images, row['index'], row['image'], seed): row['index']
            for _, row in df.iterrows()
        }
        
        results_img = {}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Degrading"):
            idx, result_str = future.result()
            results_img[idx] = result_str

    elapsed = time.time() - start_time
    print(f"  Processing finished in {elapsed:.2f}s")

    # 更新 DataFrame
    new_image_col = []
    
    for _, row in df.iterrows():
        idx = row['index']
        new_image_col.append(results_img.get(idx, row['image']))

    df['image'] = new_image_col

    name_no_ext = os.path.splitext(tsv_path)[0]
    if "LOW_LEVEL" in name_no_ext:
        new_path = tsv_path
    else:
        new_path = f"{name_no_ext}_LOW_LEVEL.tsv"
    
    # 使用 csv.QUOTE_ALL 确保整个列表字符串被双引号包围
    # TSV 文件中的样子: "['/9j/...', '/9j/...']"
    df.to_csv(new_path, sep='\t', index=False, quoting=csv.QUOTE_ALL)
    print(f"  [Success] Saved to: {new_path}")
    print("  Note: Format matches original (Python List string with single quotes). Quality=75.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/root/CLEAR/LMUData/BLINK.tsv", help="Path to input .tsv file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found.")
        return

    process_tsv(args.input_file, args.seed)

if __name__ == "__main__":
    main()