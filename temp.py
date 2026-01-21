import csv
from pathlib import Path

# 设置最大显示长度，超过这个长度的字段会被截断
MAX_DISPLAY_LEN = 50 

def truncate_value(value):
    """如果字符串太长，进行截断处理"""
    if len(value) > MAX_DISPLAY_LEN:
        # 保留前 N 个字符，后面加上省略号和总长度提示
        return f"{value[:MAX_DISPLAY_LEN]}...[共{len(value)}字符]"
    return value

def preview_tsv_with_truncation(folder_path):
    p = Path(folder_path)
    
    if not p.exists():
        print(f"错误：路径 '{folder_path}' 不存在。")
        return

    tsv_files = list(p.glob('*.tsv'))
    
    if not tsv_files:
        print(f"没有找到 TSV 文件。")
        return

    print(f"--- 找到 {len(tsv_files)} 个文件 ---\n")

    for file in tsv_files:
        print(f"📄 文件: {file.name}")
        
        try:
            with open(file, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                
                try:
                    # 1. 获取表头
                    header = next(reader)
                    
                    # 2. 尝试获取第一行数据
                    first_row = next(reader)
                    
                    # --- 关键步骤：对数据行中的每一列进行截断处理 ---
                    safe_row = [truncate_value(cell) for cell in first_row]
                    
                    # 打印表头
                    print(f"   表头: {header}")
                    # 打印处理过的数据
                    print(f"   数据: {safe_row}")
                    
                    # (可选) 垂直对照打印，看起来更清晰
                    # print("\n   [详细对照]:")
                    # for h, v in zip(header, safe_row):
                    #     print(f"     {h}: {v}")

                except StopIteration:
                    # 如果只能读取到header，说明没数据；如果连header都读不到，说明空文件
                    if 'header' in locals():
                        print("   状态: [仅有表头，无数据]")
                    else:
                        print("   状态: [空文件]")
                        
        except Exception as e:
            print(f"   错误: {e}")
            
        print("-" * 60)

# --- 使用示例 ---
target_folder = '/root/LMUData' 
preview_tsv_with_truncation(target_folder)