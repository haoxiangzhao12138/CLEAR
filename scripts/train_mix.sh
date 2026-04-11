export NCCL_TIMEOUT=3600000  # Timeout in ms (adjust as needed)

# Timestamp for experiment ID
experiment="clear_sft_drop_vit"
ts=$(date +"%Y%m%d_%H%M%S")
run_id="${ts}_${experiment}"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=23457 \
  train/pretrain_unified_corruption.py \
  --dataset_config_file ./data/configs/corruption_mix.yaml \
  --output_need_vit True \
  --distill_weight 0.1 \
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
  --cpu_offload True \
  --use_flex True \
  --lr 1e-5 \
  --ema 0.995 \
  --ce_weight 1.0 \
  --num_worker 1 \
  --total_steps 600 \
  --warmup_steps 50 \
  --save_every 200 \
  --expected_num_tokens 30000 \
  --max_num_tokens 32768 \
  --max_num_tokens_per_sample 16384 \
  --wandb_project clear_with_distill \
  --wandb_offline True \
  --wandb_name "${run_id}" \
  --wandb_runid "${run_id}" \


