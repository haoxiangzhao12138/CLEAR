import add_degradation
import cv2
import os
import numpy as np
import argparse
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# ================= Configuration (difficulty adjusted) =================

DEGRADATION_CONFIG = {
    'capture': {
        'lens_blur': {'weight': 30},       # [Hard]
        'lens_flare': {'weight': 15},
        'motion_blur': {'weight': 30},     # [Hard]
        'dirty_lens': {'weight': 15},
        'hsv_saturation': {'weight': 10}
    },
    'transmission': {
        'jpeg_compression': {'weight': 15},
        'block_exchange': {'weight': 40},   # [Very Hard]
        'mean_shift': {'weight': 15},
        'scan_lines': {'weight': 30}        # [Hard]
    },
    'environment': {
        'dark_illumination': {'weight': 40},    # [Very Hard]
        'atmospheric_turbulence': {'weight': 30}, # [Hard]
        'gaussian_noise': {'weight': 15},
        'color_diffusion': {'weight': 15}
    },
    'postprocessing': {
        'sharpness_change': {'weight': 10},
        'graffiti': {'weight': 45},             # [Very Hard]
        'watermark_damage': {'weight': 45}      # [Very Hard]
    }
}

# Pre-compute probability distribution
ALL_METHODS = []
for category, methods in DEGRADATION_CONFIG.items():
    for method_name, details in methods.items():
        ALL_METHODS.append((method_name, details['weight']))

METHOD_NAMES = [item[0] for item in ALL_METHODS]
WEIGHTS = [item[1] for item in ALL_METHODS]
TOTAL_WEIGHT = sum(WEIGHTS)
PROBABILITIES = [w / TOTAL_WEIGHT for w in WEIGHTS]

# Number of worker processes
NUM_WORKERS = max(1, cpu_count() - 2)

# ===========================================

def apply_degradation_Benchmark(image, method_name, intensity):
    degradation_func = getattr(add_degradation, method_name)
    degraded_img = degradation_func(image, intensity)
    return degraded_img

def process_single_image(filename, folder_path, output_dir):
    """
    1. Randomly select a method based on new weights (hard methods have higher probability)
    2. Select intensity in a 2:3:5 ratio [0.23, 0.45, 0.9]
    3. Save
    """
    np.random.seed()
    random.seed()
    
    try:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            return False, f"Could not read image: {filename}"

        # 1. Randomly select a degradation method
        selected_method_name = np.random.choice(METHOD_NAMES, p=PROBABILITIES)

        # 2. Select intensity by ratio (hard:medium:easy = 5:3:2)
        intensity_options = [0.23, 0.45, 0.9]
        intensity_probs   = [0.2,  0.3,  0.5]
        
        intensity = np.random.choice(intensity_options, p=intensity_probs)

        # 3. Apply degradation
        degraded_img = apply_degradation_Benchmark(image, selected_method_name, intensity)
        
        # 4. Save
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, degraded_img)
        
        return True, None

    except Exception as e:
        return False, f"{filename}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Image degradation pipeline - Hard Mode')
    parser.add_argument('--input_dir', type=str,
                       default="./datasets/processed_dataset/rl/images",
                       help='Input image directory path')
    parser.add_argument('--output_dir', type=str,
                       default="./datasets/processed_dataset/rl/corruption_images",
                       help='Output directory')
    
    args = parser.parse_args()
    
    folder_path = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.exists(folder_path):
        raise ValueError(f"Input directory does not exist: {folder_path}")

    os.makedirs(output_dir, exist_ok=True)

    print("Scanning and filtering image files...")
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]
    
    total_files = len(image_files)
    print(f"Found {total_files} images in total, output directory: {output_dir}")
    print(f"Strategy: increased weight for hard degradations + intensity biased toward 0.9 (50%)")
    
    process_func = partial(
        process_single_image, 
        folder_path=folder_path, 
        output_dir=output_dir
    )

    error_count = 0
    
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.imap_unordered(process_func, image_files, chunksize=10)
        pbar = tqdm(results, total=total_files, desc="Processing Images", unit="img")
        
        for success, error_msg in pbar:
            if not success:
                error_count += 1
                tqdm.write(f"[Error] {error_msg}")
            pbar.set_postfix(errors=error_count)

    print("\n" + "="*30)
    print("Processing completed!")
    print(f"Total Images: {total_files}")
    print(f"Total Errors: {error_count}")
    print("="*30)

if __name__ == '__main__':
    main()