import csv

def peek_tsv(file_path, num_rows=5):
    print(f"--- 正在读取文件: {file_path} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 指定分隔符为制表符 \t
            reader = csv.reader(f, delimiter='\t')
            
            # 1. 获取并打印表头
            header = next(reader)
            print(f"\n[表头]: {header}")
            print(f"列数: {len(header)}")
            print("-" * 50)
            
            # 2. 打印前几行数据
            for i in range(num_rows):
                try:
                    row = next(reader)
                    print(f"\n[第 {i+1} 行]:")
                    # 打印每一列对应的值，方便排查
                    for col_name, val in zip(header, row):
                        # 如果是太长的内容（比如base64图片），截断显示以便阅读
                        display_val = val[:50] + "..." if len(val) > 50 else val
                        print(f"  {col_name}: {display_val}")
                except StopIteration:
                    print("\n--- 文件已结束 ---")
                    break
                    
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"发生错误: {e}")

# 使用示例 (替换为你报错的那个文件路径)
# 通常在 LMUData/MMMU/ 目录下
file_path = "/root/LMUData/MMMU_DEV_VAL.tsv" 
peek_tsv(file_path)