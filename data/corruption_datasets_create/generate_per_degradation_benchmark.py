import add_degradation
import cv2
import os
import numpy as np
import argparse
import pandas as pd
import base64
import sys
import csv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 防止 CSV 读取大字段报错
csv.field_size_limit(sys.maxsize)

# 16 degradation methods
ALL_METHODS = [
    'lens_blur', 'lens_flare', 'motion_blur', 'dirty_lens', 'hsv_saturation',
    'jpeg_compression', 'block_exchange', 'mean_shift', 'scan_lines',
    'dark_illumination', 'atmospheric_turbulence', 'gaussian_noise', 'color_diffusion',
    'sharpness_change', 'graffiti', 'watermark_damage'
]

# 6 source benchmarks
TARGET_FILES = [
    "MMBench_DEV_EN_V11.tsv",
    "MMVet.tsv",
    "MMVP.tsv",
    "CV-Bench-2D.tsv",
    "MMStar.tsv",
    "RealWorldQA.tsv",
]

INTENSITY = 0.9


def base64_to_cv2(b64_str):
    try:
        img_data = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def cv2_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return b64_str


def apply_degradation(image, method_name, intensity):
    try:
        degradation_func = getattr(add_degradation, method_name)
        return degradation_func(image.copy(), intensity)
    except Exception:
        return image


def process_single_row(args):
    """处理单个图片：对指定 method 施加退化"""
    idx, original_b64, method_name = args

    if pd.isna(original_b64) or str(original_b64).strip() == "":
        return idx, original_b64

    image = base64_to_cv2(original_b64)
    if image is None:
        return idx, original_b64

    degraded_img = apply_degradation(image, method_name, INTENSITY)
    return idx, cv2_to_base64(degraded_img)


def main():
    parser = argparse.ArgumentParser(description='Per-degradation benchmark generation')
    parser.add_argument('--input_dir', type=str, default='/root/LMUData',
                        help='Directory containing the original TSV files')
    parser.add_argument('--output_dir', type=str, default='/root/LMUData',
                        help='Directory to save the processed TSV files')
    parser.add_argument('--workers', type=int, default=128,
                        help='Number of threads to use')
    parser.add_argument('--methods', type=str, nargs='*', default=None,
                        help='Specific methods to generate (default: all 16)')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    max_workers = args.workers
    methods = args.methods if args.methods else ALL_METHODS

    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting per-degradation benchmark generation with {max_workers} threads.")
    print(f"Input/Output Directory: {input_dir}")
    print(f"Methods: {methods}")
    print(f"Intensity: {INTENSITY}")
    print("-" * 50)

    for filename in TARGET_FILES:
        file_path = os.path.join(input_dir, filename)

        if not os.path.exists(file_path):
            print(f"Skipping {filename}: File not found in {input_dir}.")
            continue

        print(f"Reading {filename}...")
        try:
            df = pd.read_csv(file_path, sep='\t', engine='python', encoding='utf-8')
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        if 'image' not in df.columns:
            print(f"Skipping {filename}: No 'image' column.")
            continue

        base_name = os.path.splitext(filename)[0]

        for method_name in methods:
            output_filename = f"{base_name}_{method_name}.tsv"
            output_path = os.path.join(output_dir, output_filename)

            if os.path.exists(output_path):
                print(f"Skipping {output_filename}: already exists.")
                continue

            # 准备任务
            tasks = [(idx, img_b64, method_name)
                     for idx, img_b64 in enumerate(df['image'])]

            processed_images = [None] * len(df)

            print(f"  [{method_name}] Processing {len(tasks)} images...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(
                    executor.map(process_single_row, tasks),
                    total=len(tasks), unit="img",
                    desc=f"  {method_name}"
                ))

            for idx, b64_val in results:
                processed_images[idx] = b64_val

            new_df = df.copy()
            new_df['image'] = processed_images
            new_df.to_csv(output_path, sep='\t', index=False)
            print(f"  Saved {output_filename}")

        print(f"Done with {filename}.\n")

    print("All processing completed!")


if __name__ == '__main__':
    main()
