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

# Prevent CSV field size limit error when reading large fields
csv.field_size_limit(sys.maxsize)

DEGRADATION_CONFIG = {
    'capture': {
        'lens_blur': {'weight': 20},
        'lens_flare': {'weight': 20},
        'motion_blur': {'weight': 20},
        'dirty_lens': {'weight': 20},
        'hsv_saturation': {'weight': 20}
    },
    'transmission': {
        'jpeg_compression': {'weight': 25},
        'block_exchange': {'weight': 25},
        'mean_shift': {'weight': 25},
        'scan_lines': {'weight': 25}
    },
    'environment': {
        'dark_illumination': {'weight': 25},
        'atmospheric_turbulence': {'weight': 25},
        'gaussian_noise': {'weight': 25},
        'color_diffusion': {'weight': 25}
    },
    'postprocessing': {
        'sharpness_change': {'weight': 33},
        'graffiti': {'weight': 33},
        'watermark_damage': {'weight': 34}
    }
}

TARGET_FILES = [
    "MMVet.tsv",
    "MMBench_DEV_EN_V11.tsv",
    "MMStar.tsv",
    "MMVP.tsv",
    "CV-Bench-2D.tsv",
    "MME.tsv",
    "MathVista_MINI.tsv",
    "RealWorldQA.tsv"
]

# Modified: new intensity mapping and naming suffixes
INTENSITY_MAP = {
    0.9: '_LOW_LEVEL_HIGH',
    0.45: '_LOW_LEVEL_MID',
    0.23: '_LOW_LEVEL_LOW'
}

# Pre-compute weights
ALL_METHODS_WITH_WEIGHTS = []
for category, methods in DEGRADATION_CONFIG.items():
    for method_name, details in methods.items():
        ALL_METHODS_WITH_WEIGHTS.append((method_name, details['weight']))

METHOD_NAMES = [item[0] for item in ALL_METHODS_WITH_WEIGHTS]
WEIGHTS = [item[1] for item in ALL_METHODS_WITH_WEIGHTS]
TOTAL_WEIGHT = sum(WEIGHTS)
PROBABILITIES = [w / TOTAL_WEIGHT for w in WEIGHTS]

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

def apply_degradation_Benchmark(image, method_name, intensity):
    try:
        degradation_func = getattr(add_degradation, method_name)
        return degradation_func(image.copy(), intensity)
    except Exception:
        return image

def process_single_row(args):
    """
    Single image processing function, used by the thread pool
    """
    idx, original_b64 = args
    
    # Result container, default value is the original image
    row_result = {
        '_LOW_LEVEL_HIGH': original_b64,
        '_LOW_LEVEL_MID': original_b64,
        '_LOW_LEVEL_LOW': original_b64
    }

    # 1. Check data validity
    if pd.isna(original_b64) or str(original_b64).strip() == "":
        return idx, row_result

    # 2. Decode
    image = base64_to_cv2(original_b64)
    if image is None:
        return idx, row_result

    # 3. Randomly select method
    selected_method_name = np.random.choice(METHOD_NAMES, p=PROBABILITIES)

    # 4. Generate three intensity levels
    for intensity, suffix in INTENSITY_MAP.items():
        degraded_img = apply_degradation_Benchmark(image, selected_method_name, intensity)
        row_result[suffix] = cv2_to_base64(degraded_img)
    
    return idx, row_result

def main():
    parser = argparse.ArgumentParser(description='VLMEvalKit TSV Degradation Pipeline')
    # Default path for LMUData
    parser.add_argument('--input_dir', type=str,
                       default='./LMUData',
                       help='Directory containing the original TSV files')
    parser.add_argument('--output_dir', type=str,
                       default='./LMUData',
                       help='Directory to save the processed TSV files')
    parser.add_argument('--workers', type=int,
                       default=128,
                       help='Number of threads to use')
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    max_workers = args.workers
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting processing with {max_workers} threads.")
    print(f"Input/Output Directory: {input_dir}")
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

        # Prepare tasks
        tasks = []
        for idx, img_b64 in enumerate(df['image']):
            tasks.append((idx, img_b64))
        
        # Prepare result containers
        processed_data = {
            '_LOW_LEVEL_HIGH': [None] * len(df),
            '_LOW_LEVEL_MID': [None] * len(df),
            '_LOW_LEVEL_LOW': [None] * len(df)
        }

        print(f"Processing {len(tasks)} images in {filename}...")
        
        # Thread pool execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_single_row, tasks), total=len(tasks), unit="img"))

        # Fill back data
        for idx, row_res in results:
            for suffix, b64_val in row_res.items():
                processed_data[suffix][idx] = b64_val

        # Save files
        print(f"Saving outputs for {filename}...")
        base_name_no_ext = os.path.splitext(filename)[0]
        
        # Modified: corresponding list of save suffixes
        target_suffixes = ['_LOW_LEVEL_HIGH', '_LOW_LEVEL_MID', '_LOW_LEVEL_LOW']
        
        for suffix in target_suffixes:
            new_df = df.copy()
            new_df['image'] = processed_data[suffix]
            
            output_filename = f"{base_name_no_ext}{suffix}.tsv"
            output_path = os.path.join(output_dir, output_filename)
            
            new_df.to_csv(output_path, sep='\t', index=False)
            # print(f"Saved {output_filename}") # Reduce log spam, optional to enable
        
        print(f"Done with {filename}.\n")

    print("All processing completed!")

if __name__ == '__main__':
    main()