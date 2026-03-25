#!/bin/bash
# run_sweep.sh — Run Optuna hyperparameter sweep with wandb offline logging
#
# Usage:
#   bash run_sweep.sh                     # Run 50 trials (default)
#   bash run_sweep.sh --n_trials 100      # Custom trial count
#   bash run_sweep.sh --storage sqlite:///optuna.db  # Resumable study
#
# After completion, upload wandb results from an internet-connected env:
#   cd VLMEvalKit && wandb sync --sync-all ./wandb

set -euo pipefail
cd "$(dirname "$0")"

# Force wandb offline mode (no internet needed during evaluation)
export WANDB_MODE=offline

echo "[run_sweep] Starting Optuna sweep (wandb offline mode)"
echo "[run_sweep] Results will be saved locally. Use 'wandb sync' to upload later."
echo ""

python sweep_agent.py "$@"

echo ""
echo "[run_sweep] Done. To upload wandb results:"
echo "  cd $(pwd) && wandb sync --sync-all ./wandb"
