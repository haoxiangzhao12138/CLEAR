import os
import io
import base64
import pandas as pd
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# ================= 配置部分 =================
# 填写你本地 Parquet 文件的路径
# 如果是一个文件： "path/to/rbench_test.parquet"
# 如果是一个文件夹里的多个文件： ["path/to/part1.parquet", "path/to/part2.parquet"]
YOUR_LOCAL_PARQUET_PATH = "/root/CLEAR/R_Bench_All.parquet" 

# 输出目录
OUTPUT_DIR = "LMUData"
# ===========================================

def encode_image_to_base64(img_data):
    """
    将图片数据转换为 Base64 字符串。
    输入可能是 PIL Image 对象，也可能是 bytes 字典，也可能是纯 bytes。
    """
    img = None
    
    # 情况1: 输入是 PIL Image 对象 (datasets 自动解码了图片)
    if isinstance(img_data, Image.Image):
        img = img_data
    # 情况2: 输入是字典 (通常包含 'bytes' 字段)
    elif isinstance(img_data, dict) and 'bytes' in img_data:
        img = Image.open(io.BytesIO(img_data['bytes']))
    # 情况3: 输入直接是 bytes 二进制数据
    elif isinstance(img_data, bytes):
        img = Image.open(io.BytesIO(img_data))
    else:
        raise ValueError(f"Unknown image format: {type(img_data)}")

    # 转换为 RGB 并编码
    buffered = io.BytesIO()
    img = img.convert('RGB')
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading local parquet file from: {YOUR_LOCAL_PARQUET_PATH}")
    
    # 1. 使用 datasets 库加载本地 Parquet
    # split='train' 是因为直接加载文件时，默认会被分配到 'train' split，即使文件名叫 test
    try:
        ds = load_dataset("parquet", data_files={'train': YOUR_LOCAL_PARQUET_PATH}, split='train')
    except Exception as e:
        print(f"加载失败，请检查路径是否正确。错误信息: {e}")
        return

    print(f"Dataset loaded. Total samples: {len(ds)}")

    ref_rows = []
    dis_rows = []

    print("Processing images and converting to TSV format...")

    for i, item in tqdm(enumerate(ds), total=len(ds)):
        # --- 提取通用字段 ---
        # 注意：datasets 加载 parquet 后，字段名通常保持不变
        common_data = {
            'index': i,
            'question': item['question'],
            'answer': item['answer'],
            'dataset_type': item['type'], 
            'choices': item['choice'], 
            'distortion': item['distortion'],
            'strength': item['strength']
        }

        # --- 处理 Reference (原始) 图片 ---
        try:
            ref_b64 = encode_image_to_base64(item['ref_image'])
            ref_row = common_data.copy()
            ref_row['image'] = ref_b64
            ref_rows.append(ref_row)
        except Exception as e:
            print(f"Error processing ref_image at index {i}: {e}")

        # --- 处理 Distorted (干扰) 图片 ---
        try:
            dis_b64 = encode_image_to_base64(item['dis_image'])
            dis_row = common_data.copy()
            dis_row['image'] = dis_b64
            dis_rows.append(dis_row)
        except Exception as e:
            print(f"Error processing dis_image at index {i}: {e}")

    # 2. 保存为 TSV
    # 只需要生成这两个文件，VLMEvalKit 就能识别
    columns_order = ['index', 'image', 'question', 'answer', 'dataset_type', 'choices', 'distortion', 'strength']
    
    df_ref = pd.DataFrame(ref_rows)
    # 过滤掉不在 dataframe 中的列（防止某些列全空导致报错）
    final_cols = [c for c in columns_order if c in df_ref.columns]
    
    ref_path = os.path.join(OUTPUT_DIR, 'RBench_Ref_all.tsv')
    print(f"Saving Reference split to {ref_path}...")
    df_ref[final_cols].to_csv(ref_path, sep='\t', index=False)

    df_dis = pd.DataFrame(dis_rows)
    dis_path = os.path.join(OUTPUT_DIR, 'RBench_Dis_all.tsv')
    print(f"Saving Distorted split to {dis_path}...")
    df_dis[final_cols].to_csv(dis_path, sep='\t', index=False)

    print("Done! Files are ready in LMUData folder.")

if __name__ == "__main__":
    main()