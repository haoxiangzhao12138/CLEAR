import pandas as pd
import sys
import csv
import json
import ast
import os

# -----------------------------
# 配置
# -----------------------------
# 原始 BLINK 文件路径
ORIGINAL_FILE = "/root/CLEAR/LMUData/BLINK.tsv"
# 你生成的 Low Level 文件路径
NEW_FILE = "/root/CLEAR/LMUData/BLINK_LOW_LEVEL.tsv"

# 增加 CSV 读取限制
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)

def get_file_size(path):
    size_bytes = os.path.getsize(path)
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def analyze_tsv(path, label):
    print(f"\n{'='*20} 分析文件: {label} {'='*20}")
    print(f"文件路径: {path}")
    
    if not os.path.exists(path):
        print("❌ 文件不存在")
        return None

    print(f"文件大小: {get_file_size(path)}")

    try:
        # 只读取前 5 行以快速检查格式，避免内存爆炸
        df_head = pd.read_csv(path, sep='\t', dtype={'index': str}, nrows=5)
        # 读取完整 shape 稍微慢点，但为了对比行数
        # df_full = pd.read_csv(path, sep='\t', dtype={'index': str}, usecols=['index'])
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return None

    if 'image' not in df_head.columns:
        print("❌ 缺少 'image' 列")
        return None
    
    # 提取第一行有效数据
    sample_row = df_head.iloc[0]
    raw_img_str = sample_row['image']
    
    print(f"\n--- 样本数据 (第 1 行) ---")
    print(f"原始字符串类型: {type(raw_img_str)}")
    print(f"原始字符串前 100 字符: {str(raw_img_str)[:100]} ...")
    print(f"原始字符串后 50 字符: ... {str(raw_img_str)[-50:]}")
    
    # 尝试解析结构
    parsed_data = None
    parse_method = "Unknown"
    
    # 1. 尝试 JSON 解析
    try:
        parsed_data = json.loads(raw_img_str)
        parse_method = "JSON (标准)"
    except:
        # 2. 尝试 AST (Python Literal) 解析
        try:
            parsed_data = ast.literal_eval(raw_img_str)
            parse_method = "AST (Python List - 单引号)"
        except:
            parse_method = "无法解析 (可能是纯字符串或格式错误)"
            parsed_data = raw_img_str

    print(f"\n--- 格式分析 ---")
    print(f"解析方式: {parse_method}")
    
    if isinstance(parsed_data, list):
        print(f"数据结构: List (列表)")
        print(f"列表长度: {len(parsed_data)}")
        
        if len(parsed_data) > 0:
            img_item = parsed_data[0]
            print(f"列表项类型: {type(img_item)}")
            
            if isinstance(img_item, str):
                length = len(img_item)
                print(f"单张图片 Base64 长度: {length}")
                
                # 检查前缀
                if img_item.startswith("data:image"):
                    print(f"⚠️  包含前缀: YES (例如 {img_item[:20]}...) -> VLMEvalKit 可能报错")
                else:
                    print(f"✅ 包含前缀: NO (纯 Base64) -> VLMEvalKit 喜欢这个")
                
                # 检查空格/换行
                if "\n" in img_item or " " in img_item:
                    print(f"⚠️  包含换行符或空格: YES")
                else:
                    print(f"✅ 包含换行符或空格: NO")
    else:
        print(f"⚠️  数据结构: {type(parsed_data)} (不是列表！BLINK 通常是列表)")

    return {
        "parse_method": parse_method,
        "is_list": isinstance(parsed_data, list),
        "has_prefix": parsed_data[0].startswith("data:") if isinstance(parsed_data, list) and len(parsed_data)>0 else False,
        "sample_length": len(parsed_data[0]) if isinstance(parsed_data, list) and len(parsed_data)>0 else 0
    }

def main():
    res_orig = analyze_tsv(ORIGINAL_FILE, "原始 BLINK")
    res_new = analyze_tsv(NEW_FILE, "新生成 BLINK_LOW_LEVEL")

    print(f"\n\n{'='*20} 对比总结 {'='*20}")
    
    if res_orig and res_new:
        # 1. 对比结构
        if res_orig['is_list'] == res_new['is_list']:
            print(f"✅ 数据结构一致 (都是 List)")
        else:
            print(f"❌ 数据结构不一致! 原版是 {res_orig['is_list']}, 新版是 {res_new['is_list']}")

        # 2. 对比解析方式
        if res_orig['parse_method'] == res_new['parse_method']:
            print(f"✅ 存储格式一致 ({res_orig['parse_method']})")
        else:
            print(f"⚠️  存储格式不同 (但这通常没问题，JSON 更好):")
            print(f"   原版: {res_orig['parse_method']}")
            print(f"   新版: {res_new['parse_method']}")

        # 3. 对比前缀
        if res_orig['has_prefix'] == res_new['has_prefix']:
            print(f"✅ 前缀一致 (Has Prefix: {res_orig['has_prefix']})")
        else:
            print(f"❌ 前缀不一致!")
            print(f"   原版有前缀? {res_orig['has_prefix']}")
            print(f"   新版有前缀? {res_new['has_prefix']}")
            if res_new['has_prefix']:
                print("   -> 建议：去除新文件中的 'data:image...' 前缀")

        # 4. 对比大小
        len_orig = res_orig['sample_length']
        len_new = res_new['sample_length']
        ratio = len_new / len_orig if len_orig > 0 else 0
        print(f"\n📉 图片数据大小对比 (单张样本):")
        print(f"   原版长度: {len_orig}")
        print(f"   新版长度: {len_new}")
        print(f"   倍数关系: {ratio:.2f}x")
        
        if ratio > 1.5:
            print("   -> 解释: 新文件显著变大。原因可能是保存时 Quality=95 (原图可能较低)，\n      或者噪点增加了图片信息熵导致压缩率下降。这是正常的，只要不超过内存限制。")

if __name__ == "__main__":
    main()