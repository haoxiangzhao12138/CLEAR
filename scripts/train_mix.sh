export NCCL_TIMEOUT=3600000  # Timeout in ms (adjust as needed)

# Timestamp for experiment ID
experiment="distill_per_layer_more_995"
ts=$(date +"%Y%m%d_%H%M%S")
run_id="${ts}_${experiment}"

# ---- Distillation config ----
# distill_mode: "final_mse" (baseline, mean-pooled MSE on last hidden state)
#               "per_layer_kl" (per-token KL divergence at intermediate layers)
# distill_weight: scaling factor for distillation loss (0 = disabled)
# distill_layer_stride: for per_layer_kl, distill every N-th layer + last layer
DISTILL_MODE="per_layer_kl"
DISTILL_LAYER_STRIDE=1

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=23457 \
  train/pretrain_unified_corruption.py \
  --dataset_config_file ./data/configs/corruption_mix.yaml \
  --output_need_vit True \
  --distill_weight 0.4 \
  --distill_mode ${DISTILL_MODE} \
  --distill_layer_stride ${DISTILL_LAYER_STRIDE} \
  --gradient_checkpointing  True \
  --checkpoint_dir "./results/${run_id}" \
  --model_path ./models/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from ./models/BAGEL-7B-MoT \
  --finetune_from_hf True \
  --auto_resume False \
  --resume-model-only True \
  --finetune-from-ema True \
  --visual_gen True \
  --visual_und True \
  --freeze_vit True \
  --freeze_vae True \
  --freeze_llm not \
  --log_every 1 \
  --vit_cond_dropout_prob 0.4 \
  --vae_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0.1 \
  --cpu_offload False \
  --use_flex True \
  --lr 1e-5 \
  --ema 0.995 \
  --weight_decay 0.01 \
  --ce_weight 1.0 \
  --num_workers 1 \
  --total_steps 8000 \
  --warmup_steps 500 \
  --save_every 2000 \
  --expected_num_tokens 30000 \
  --max_num_tokens 32768 \
  --max_num_tokens_per_sample 32768 \
  --wandb_project clear_with_distill \
  --wandb_offline True \
  --wandb_name "${run_id}" \
  --wandb_runid "${run_id}"
