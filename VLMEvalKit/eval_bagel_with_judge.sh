#!/bin/bash

# export your openai api key
export OPENAI_API_KEY="sk-zSaDTLLv9cSEwRRt9oLhmMMNpFidKy4cGEtogaICub4mFw67"
export OPENAI_API_BASE="http://yy.dbh.baidu-int.com/"

# 记录开始时间
start_time=$(date +%s)

# ["chatgpt-0125", "exact_matching", "gpt-4-0125", "deepseek"]

torchrun \
    --nproc-per-node=8 \
    --master_port=29503 \
    run.py \
    --config ./eval_cfg/origin_bagel.json \
    # --judge gpt-4-0125 \
    --verbose \


# 记录结束时间
end_time=$(date +%s)

# 计算运行时间（秒）
duration=$((end_time - start_time))

# 将秒数转换为小时、分钟和秒
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

# 打印总运行时间
echo "总运行时间: ${hours}小时 ${minutes}分钟 ${seconds}秒"