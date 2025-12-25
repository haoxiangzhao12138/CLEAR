from vlmeval.dataset import build_dataset

# 1. 填入你跑的数据集名字（必须和 run.py 里的 --data 参数一致）
# 例如: 'MMBench_DEV_EN', 'MME', 'MathVista_MINI', 'SEEDBench_IMG'
dataset_name = 'MathVista_MINI' 

# 2. 你的 xlsx 结果文件路径x
result_file = '/root/CLEAR/VLMEvalKit/outputs/BAGEL/T20251224_G9f092656/BAGEL_MathVista_MINI_LOW_LEVEL.xlsx'

# 3. 构建数据集对象
dataset = build_dataset(dataset_name)

# 4. 调用对象自带的评估方法
# 该方法会自动处理：提取选项 -> 匹配答案 -> 计算分数 -> (可选)调用GPT4打分
# 结果通常会打印在控制台，并返回一个字典或 pandas dataframe
res = dataset.evaluate(result_file, model='gpt-4-0125')

print(res)