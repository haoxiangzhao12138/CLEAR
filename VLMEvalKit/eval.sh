#!/bin/bash

# 记录开始时间
start_time=$(date +%s)

# 说明：
# 1. 移除了 --verbose 后面的反斜杠
# 2. 只有 MME 数据集时，不需要 --judge 参数，MME 是规则评测
torchrun \
    --nproc-per-node=8 \
    --master_port=29503 \
    run.py \
    --config ./config/test.json \
    --judge gpt-4-0125 \
    --verbose

# 记录结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "总运行时间: ${hours}小时 ${minutes}分钟 ${seconds}秒"