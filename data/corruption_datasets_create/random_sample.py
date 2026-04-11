import os
import json
import random
import re
import shutil
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm
from PIL import Image

# ================= Configuration =================

LOCAL_DATASET_PATH = "./datasets/LLaVA-OneVision-Data"
OUTPUT_ROOT = "./datasets/processed_dataset"
TEMP_DIR = os.path.join(OUTPUT_ROOT, "temp_jsonl")  # Temporary directory

# Sampling ratios
SFT_RATIO = 0.05
RL_RATIO  = 0.01

# Image filtering thresholds
MIN_RESOLUTION = 64
MAX_ASPECT_RATIO = 5.0

# Number of worker processes (defaults to CPU core count - 2, to avoid freezing the machine)
NUM_WORKERS = max(1, cpu_count() - 2)

# ===========================================

def check_contains_chinese(text):
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def check_image_quality(image_obj):
    if image_obj is None:
        return False, "is_none"
    try:
        w, h = image_obj.size
        if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
            return False, "too_small"
        if min(w, h) == 0:
            return False, "zero_dim"
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > MAX_ASPECT_RATIO:
            return False, "extreme_aspect_ratio"
        return True, "ok"
    except Exception:
        return False, "corrupt_file"

def setup_directories():
    """Run in main process: create directories"""
    dirs = {
        "sft": os.path.join(OUTPUT_ROOT, "sft", "images"),
        "rl": os.path.join(OUTPUT_ROOT, "rl", "images"),
        "temp": TEMP_DIR
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs

def append_to_temp_jsonl(data, subset_name, split_type):
    """Write to a process-independent temporary file"""
    # Filename format: temp_sft_subsetName.jsonl
    filename = f"temp_{split_type}_{subset_name}.jsonl"
    file_path = os.path.join(TEMP_DIR, filename)
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def process_single_subset(config_name):
    """
    [Worker process] Process a single subset
    """
    # Re-seed random to ensure different randomness across processes
    random.seed()
    
    # Local statistics
    local_stats = {
        "sft": 0, "rl": 0, 
        "skip_random": 0, "skip_chinese": 0, "skip_bad_image": 0,
        "error": 0
    }
    
    try:
        # Must reload dataset within the process; streaming mode cannot be pickled across processes
        ds = load_dataset(
            LOCAL_DATASET_PATH, 
            config_name if config_name != 'default' else None, 
            split="train",
            streaming=True
        )
        
        # Safe config name (used for filenames)
        safe_config = str(config_name).replace("/", "_").replace("(", "_").replace(")", "_")

        for sample in ds:
            try:
                # 1. Language detection
                conversations = sample.get("conversations", [])
                full_text = ""
                if isinstance(conversations, list):
                    for turn in conversations:
                        full_text += turn.get("value", "")
                
                if check_contains_chinese(full_text):
                    local_stats["skip_chinese"] += 1
                    continue

                # 2. Random sampling
                r = random.random()
                target_split = None
                if r < SFT_RATIO:
                    target_split = "sft"
                elif r < (SFT_RATIO + RL_RATIO):
                    target_split = "rl"
                else:
                    local_stats["skip_random"] += 1
                    continue 

                # 3. Image quality check
                image_obj = sample.get("image")
                is_valid_img, _ = check_image_quality(image_obj)
                
                if not is_valid_img:
                    local_stats["skip_bad_image"] += 1
                    continue

                # 4. Save processing
                raw_id = str(sample.get('id', 'unknown'))
                safe_id = raw_id.replace("/", "_").replace("\\", "_")
                
                # Format handling
                orig_format = image_obj.format
                if orig_format == "PNG":
                    ext = ".png"
                    save_format = "PNG"
                else:
                    ext = ".jpg"
                    save_format = "JPEG"
                    if image_obj.mode != "RGB":
                        image_obj = image_obj.convert("RGB")
                
                # Build paths
                img_filename = f"{safe_config}_{safe_id}{ext}"
                target_img_dir = os.path.join(OUTPUT_ROOT, target_split, "images")
                save_path = os.path.join(target_img_dir, img_filename)
                
                image_obj.save(save_path, format=save_format)
                
                # Write to temporary JSONL
                json_entry = {
                    "id": raw_id,
                    "subset": config_name,
                    "image": img_filename,
                    "conversations": conversations,
                    "split_type": target_split
                }
                
                # No lock needed here because the filename includes config_name, providing natural isolation
                append_to_temp_jsonl(json_entry, safe_config, target_split)
                local_stats[target_split] += 1
                
            except Exception as e:
                # Print single-item error but do not interrupt the entire subset
                # print(f"[Error in {config_name}] Item {sample.get('id')}: {e}")
                local_stats["error"] += 1
                continue

    except Exception as e:
        print(f"\n[Fatal Error] Subset {config_name} failed: {e}")
        return local_stats

    return local_stats

def merge_results():
    """
    [Post-processing] Merge scattered temporary JSONL files into the final file
    """
    print("Merging temporary files...")
    final_files = {
        "sft": os.path.join(OUTPUT_ROOT, "sft", "sft_data.jsonl"),
        "rl": os.path.join(OUTPUT_ROOT, "rl", "rl_data.jsonl")
    }
    
    # Clear old main files
    for fpath in final_files.values():
        if os.path.exists(fpath):
            os.remove(fpath)

    # Iterate over temporary directory
    temp_files = os.listdir(TEMP_DIR)
    
    for fname in tqdm(temp_files, desc="Merging"):
        if not fname.endswith(".jsonl"): continue
        
        # Parse type (temp_sft_... -> sft)
        split_type = "sft" if "temp_sft_" in fname else "rl"
        target_file = final_files[split_type]
        
        src_path = os.path.join(TEMP_DIR, fname)
        
        # Append by streaming read/write to prevent memory overflow
        with open(src_path, 'r', encoding='utf-8') as f_in:
            with open(target_file, 'a', encoding='utf-8') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    # Clean up temporary directory
    print("Cleaning up temporary files...")
    shutil.rmtree(TEMP_DIR)

def main():
    setup_directories()
    
    print(f"Reading configuration: {LOCAL_DATASET_PATH}")
    try:
        configs = get_dataset_config_names(LOCAL_DATASET_PATH)
        print(f"Found {len(configs)} subsets in total; will use {NUM_WORKERS} processes for parallel processing.")
    except Exception as e:
        print(f"Failed to read configuration: {e}")
        configs = ['default']

    # Global statistics aggregation
    global_stats = {
        "sft": 0, "rl": 0, 
        "skip_random": 0, "skip_chinese": 0, "skip_bad_image": 0, 
        "error": 0
    }

    # Start process pool
    with Pool(processes=NUM_WORKERS) as pool:
        # Use imap_unordered to get progress in real time without waiting for all to finish
        # chunksize=1 means return as soon as one subset is done, suitable for tasks with varying durations
        iterator = pool.imap_unordered(process_single_subset, configs, chunksize=1)
        
        # Outer progress bar shows number of completed subsets
        pbar = tqdm(iterator, total=len(configs), desc="Total Progress (Subsets)")
        
        for res in pbar:
            # Aggregate statistics
            for k in global_stats:
                global_stats[k] += res.get(k, 0)
                
            # Update progress bar suffix
            pbar.set_postfix(
                sft=global_stats['sft'], 
                rl=global_stats['rl'], 
                bad=global_stats['skip_bad_image']
            )

    print("\n" + "="*30)
    print("All subprocess tasks finished, starting data merge...")
    merge_results()
    
    print("\n" + "="*30)
    print("Processing complete! (Multiprocessing)")
    print(f"SFT total data count: {global_stats['sft']}")
    print(f"RL  total data count: {global_stats['rl']}")
    print(f"Skipped (Chinese): {global_stats['skip_chinese']}")
    print(f"Skipped (bad image): {global_stats['skip_bad_image']}")
    print(f"Skipped (random): {global_stats['skip_random']}")
    print(f"Processing errors: {global_stats['error']}")
    print(f"Results saved to: {OUTPUT_ROOT}")
    print("="*30)

if __name__ == "__main__":
    # On Windows/MacOS, multiprocessing must be placed under if __name__ == "__main__":
    main()