#!/bin/bash

# Record start time
start_time=$(date +%s)

# Notes:
# 1. Removed the backslash after --verbose
# 2. With only the MME dataset, the --judge parameter is not needed; MME uses rule-based evaluation
torchrun \
    --nproc-per-node=8 \
    --master_port=29503 \
    run.py \
    --config ./config/MMBench_DEV_EN_V11_per_degradation.json \
    --judge gpt-4-0125 \
    --verbose

# Record end time
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"