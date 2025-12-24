import pandas as pd
import glob
import os
import ast

def check_tsv_image_format(directory="."):
    search_path = os.path.join(directory, "*.tsv")
    files = glob.glob(search_path)
    
    if not files:
        print(f"在 '{directory}' 下未找到 TSV 文件。")
        return

    # 打印表头格式
    print(f"{'文件名':<35} | {'类型':<10} | {'Image格式 (示例)':<20} | {'有image_path?':<15}")
    print("-" * 90)

    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            # 只读取前 5 行，速度极快
            df = pd.read_csv(file_path, sep='\t', nrows=5)
            
            if 'image' not in df.columns:
                print(f"{file_name:<35} | {'无image列':<10} | {'-'*20} | {'-'*15}")
                continue

            # 获取第一条非空的 image 数据
            first_img = df['image'].dropna().iloc[0] if not df['image'].dropna().empty else None
            
            if first_img is None:
                print(f"{file_name:<35} | {'全空':<10} | {'None':<20} | {str('image_path' in df.columns):<15}")
                continue

            # 转换为字符串并去空格
            img_str = str(first_img).strip()
            
            # 判断逻辑
            is_list_str = img_str.startswith('[') and img_str.endswith(']')
            has_path_col = 'image_path' in df.columns
            
            # 尝试解析以确认是否真的是列表
            structure_type = "String"
            display_str = "单张图片"
            if is_list_str:
                try:
                    parsed = ast.literal_eval(img_str)
                    if isinstance(parsed, list):
                        structure_type = "LIST"
                        display_str = f"列表 (len={len(parsed)})"
                    else:
                        structure_type = "String" # 看起来像列表但解析出来不是
                except:
                    structure_type = "String (伪List)"
            
            # 加上颜色标记 (如果是在终端运行)
            # 列表类型通常需要 image_path，如果没有则是高危
            risk_flag = ""
            if structure_type == "LIST" and not has_path_col:
                risk_flag = " [DATASET ERROR?]" 

            print(f"{file_name:<35} | {structure_type:<10} | {display_str:<20} | {str(has_path_col):<15} {risk_flag}")

        except Exception as e:
            print(f"{file_name:<35} | 读取错误: {str(e)[:30]}...")

# 运行检查
# 将路径修改为你存放 BLINK 或其他数据集的目录
target_dir = "/root/CLEAR/LMUData"  # <--- 修改这里
if os.path.exists(target_dir):
    check_tsv_image_format(target_dir)
else:
    print(f"目录不存在: {target_dir}")