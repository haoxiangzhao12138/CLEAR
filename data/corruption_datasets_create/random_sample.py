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

# ================= 配置区域 =================

LOCAL_DATASET_PATH = "./datasets/LLaVA-OneVision-Data"
OUTPUT_ROOT = "./datasets/processed_dataset"
TEMP_DIR = os.path.join(OUTPUT_ROOT, "temp_jsonl")  # 临时文件夹

# 采样比例
SFT_RATIO = 0.05
RL_RATIO  = 0.01

# 图片过滤阈值
MIN_RESOLUTION = 64
MAX_ASPECT_RATIO = 5.0

# 进程数控制 (默认使用 CPU 核心数 - 2，防止卡死机器)
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
    """主进程运行：创建文件夹"""
    dirs = {
        "sft": os.path.join(OUTPUT_ROOT, "sft", "images"),
        "rl": os.path.join(OUTPUT_ROOT, "rl", "images"),
        "temp": TEMP_DIR
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs

def append_to_temp_jsonl(data, subset_name, split_type):
    """写入进程独立的临时文件"""
    # 文件名如: temp_sft_subsetName.jsonl
    filename = f"temp_{split_type}_{subset_name}.jsonl"
    file_path = os.path.join(TEMP_DIR, filename)
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def process_single_subset(config_name):
    """
    【工作进程】处理单个子集
    """
    # 重新设置随机种子，确保多进程下随机性不同
    random.seed()
    
    # 局部统计
    local_stats = {
        "sft": 0, "rl": 0, 
        "skip_random": 0, "skip_chinese": 0, "skip_bad_image": 0,
        "error": 0
    }
    
    try:
        # 必须在进程内重新加载 dataset，streaming 模式无法跨进程 pickling
        ds = load_dataset(
            LOCAL_DATASET_PATH, 
            config_name if config_name != 'default' else None, 
            split="train",
            streaming=True
        )
        
        # 安全配置名 (用于文件名)
        safe_config = str(config_name).replace("/", "_").replace("(", "_").replace(")", "_")

        for sample in ds:
            try:
                # 1. 语言检测
                conversations = sample.get("conversations", [])
                full_text = ""
                if isinstance(conversations, list):
                    for turn in conversations:
                        full_text += turn.get("value", "")
                
                if check_contains_chinese(full_text):
                    local_stats["skip_chinese"] += 1
                    continue

                # 2. 随机采样
                r = random.random()
                target_split = None
                if r < SFT_RATIO:
                    target_split = "sft"
                elif r < (SFT_RATIO + RL_RATIO):
                    target_split = "rl"
                else:
                    local_stats["skip_random"] += 1
                    continue 

                # 3. 图片质量检测
                image_obj = sample.get("image")
                is_valid_img, _ = check_image_quality(image_obj)
                
                if not is_valid_img:
                    local_stats["skip_bad_image"] += 1
                    continue

                # 4. 保存处理
                raw_id = str(sample.get('id', 'unknown'))
                safe_id = raw_id.replace("/", "_").replace("\\", "_")
                
                # 格式处理
                orig_format = image_obj.format
                if orig_format == "PNG":
                    ext = ".png"
                    save_format = "PNG"
                else:
                    ext = ".jpg"
                    save_format = "JPEG"
                    if image_obj.mode != "RGB":
                        image_obj = image_obj.convert("RGB")
                
                # 构建路径
                img_filename = f"{safe_config}_{safe_id}{ext}"
                target_img_dir = os.path.join(OUTPUT_ROOT, target_split, "images")
                save_path = os.path.join(target_img_dir, img_filename)
                
                image_obj.save(save_path, format=save_format)
                
                # 写入临时 JSONL
                json_entry = {
                    "id": raw_id,
                    "subset": config_name,
                    "image": img_filename,
                    "conversations": conversations,
                    "split_type": target_split
                }
                
                # 这里不加锁，因为文件名包含 config_name，天然隔离
                append_to_temp_jsonl(json_entry, safe_config, target_split)
                local_stats[target_split] += 1
                
            except Exception as e:
                # 打印单条数据错误，但不中断整个子集
                # print(f"[Error in {config_name}] Item {sample.get('id')}: {e}")
                local_stats["error"] += 1
                continue

    except Exception as e:
        print(f"\n[Fatal Error] Subset {config_name} failed: {e}")
        return local_stats

    return local_stats

def merge_results():
    """
    [后处理] 将分散的临时 JSONL 合并为最终文件
    """
    print("正在合并临时文件...")
    final_files = {
        "sft": os.path.join(OUTPUT_ROOT, "sft", "sft_data.jsonl"),
        "rl": os.path.join(OUTPUT_ROOT, "rl", "rl_data.jsonl")
    }
    
    # 清空旧的主文件
    for fpath in final_files.values():
        if os.path.exists(fpath):
            os.remove(fpath)

    # 遍历临时目录
    temp_files = os.listdir(TEMP_DIR)
    
    for fname in tqdm(temp_files, desc="Merging"):
        if not fname.endswith(".jsonl"): continue
        
        # 解析类型 (temp_sft_... -> sft)
        split_type = "sft" if "temp_sft_" in fname else "rl"
        target_file = final_files[split_type]
        
        src_path = os.path.join(TEMP_DIR, fname)
        
        # 追加写入 (流式读写，防止内存爆炸)
        with open(src_path, 'r', encoding='utf-8') as f_in:
            with open(target_file, 'a', encoding='utf-8') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    # 清理临时目录
    print("清理临时文件...")
    shutil.rmtree(TEMP_DIR)

def main():
    setup_directories()
    
    print(f"正在读取配置: {LOCAL_DATASET_PATH}")
    try:
        configs = get_dataset_config_names(LOCAL_DATASET_PATH)
        print(f"共发现 {len(configs)} 个子集，将使用 {NUM_WORKERS} 个进程并行处理。")
    except Exception as e:
        print(f"读取配置失败: {e}")
        configs = ['default']

    # 全局统计聚合
    global_stats = {
        "sft": 0, "rl": 0, 
        "skip_random": 0, "skip_chinese": 0, "skip_bad_image": 0, 
        "error": 0
    }

    # 开启进程池
    with Pool(processes=NUM_WORKERS) as pool:
        # 使用 imap_unordered 可以实时获取进度，而不必等所有都做完
        # chunksize=1 表示只要有一个子集做完就返回，适合任务耗时差异大的情况
        iterator = pool.imap_unordered(process_single_subset, configs, chunksize=1)
        
        # 外层进度条显示已完成的子集数量
        pbar = tqdm(iterator, total=len(configs), desc="Total Progress (Subsets)")
        
        for res in pbar:
            # 聚合统计数据
            for k in global_stats:
                global_stats[k] += res.get(k, 0)
                
            # 更新进度条后缀
            pbar.set_postfix(
                sft=global_stats['sft'], 
                rl=global_stats['rl'], 
                bad=global_stats['skip_bad_image']
            )

    print("\n" + "="*30)
    print("所有子进程任务结束，开始合并数据...")
    merge_results()
    
    print("\n" + "="*30)
    print("处理完成！(Multiprocessing)")
    print(f"SFT 总数据量: {global_stats['sft']}")
    print(f"RL  总数据量: {global_stats['rl']}")
    print(f"中文跳过: {global_stats['skip_chinese']}")
    print(f"坏图跳过: {global_stats['skip_bad_image']}")
    print(f"随机跳过: {global_stats['skip_random']}")
    print(f"处理错误: {global_stats['error']}")
    print(f"结果已保存在: {OUTPUT_ROOT}")
    print("="*30)

if __name__ == "__main__":
    # Windows/MacOS 下 multiprocessing 必须放在 if __name__ == "__main__": 下
    main()