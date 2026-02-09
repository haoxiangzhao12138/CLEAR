export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export WANDB_MODE=offline

ts=$(date +"%Y%m%d_%H%M%S")
export WANDB_PROJECT=interleaved-reasoning
export WANDB_RUN_NAME=reason_interleave_grpo_${ts}_small_data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


torchrun \
    --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    train/grpo/interleave_grpo.py \
    --deepspeed ./scripts/zero3_offload.json \
    --bf16 true \
    --output_dir results/${WANDB_RUN_NAME} \
    --save_dir ./case_out/${WANDB_RUN_NAME} \
    --jsonl_path /root/CLEAR/datasets/processed_dataset/rl/rl_data.jsonl \
    --image_root /root/CLEAR/datasets/processed_dataset/rl/corruption_images/ \
    --max_think_token_n 4096 \
    --max_completion_length 16384 \
    --output_need_vae True \
    --output_need_vit True \
    --learning_rate 3e-6 \
    --lr_scheduler_type cosine \
    --num_iterations 1 \
    --num_generations 8 \
    --beta 0.0 \
    --num_timesteps 50 \
    --mask_truncated_completions false \
    --use_liger_kernel false \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --ddp_timeout 7200 \
    --logging_steps 1 \
    --report_to wandb \
    --gradient_checkpointing false \
    --max_steps 40 \
    --run_name $WANDB_RUN_NAME \
    --save_steps 5 \
    --save_only_model true \
    --model_path ./models/BAGEL-7B-MoT \
    --model_param_path /root/CLEAR/results/20260208_203829_mix_small_data/0000600 \
    --layer_module Qwen2MoTDecoderLayer \
    --max_latent_size 64 \
    --use_flex False \
    --max_num_tokens 16384