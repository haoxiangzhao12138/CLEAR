export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export WANDB_MODE=offline

ts=$(date +"%Y%m%d_%H%M%S")
export WANDB_PROJECT=interleaved-reasoning
export WANDB_RUN_NAME=reason_interleave_grpo_${ts}_improved_standard
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
    --clean_image_root /root/CLEAR/datasets/processed_dataset/rl/images/ \
    --mse_scale 0.3 \
    --latent_reward_mode vae \
    --reward_funcs accuracy format decision latent_quality \
    --enable_reward_accuracy True \
    --enable_reward_format True \
    --enable_reward_decision True \
    --enable_reward_latent_quality True \
    --max_think_token_n 4096 \
    --max_completion_length 16384 \
    --output_need_vae True \
    --output_need_vit False \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --num_iterations 1 \
    --num_generations 8 \
    --beta 0.0 \
    --num_timesteps 30 \
    --mask_truncated_completions false \
    --use_liger_kernel false \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --ddp_timeout 7200 \
    --logging_steps 1 \
    --report_to wandb \
    --gradient_checkpointing false \
    --max_steps 200 \
    --run_name $WANDB_RUN_NAME \
    --save_steps 50\
    --save_only_model true \
    --model_path ./models/BAGEL-7B-MoT \
    --model_param_path /root/CLEAR/results/20260209_000155_mix_without_vit/0003000 \
    --layer_module Qwen2MoTDecoderLayer \
    --max_latent_size 64 \
    --use_flex False \
    --max_num_tokens 16384 \
    --use_text_grpo True \
    --use_flow_grpo True \
    --sde_sigma 1.0 \
    --num_timesteps_train 10 \
    --image_loss_weight 0.5 \
    --trajectory_selection_strategy weighted \
    --separate_image_rewards False \
    --decision_reward_smooth False