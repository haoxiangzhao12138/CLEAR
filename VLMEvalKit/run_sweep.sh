#!/bin/bash
# run_sweep.sh — Run wandb online sweep for BagelInference hyperparameter tuning
#
# Usage:
#   bash run_sweep.sh                                    # New sweep, 50 trials
#   bash run_sweep.sh --count 100                        # Custom trial count
#   bash run_sweep.sh --sweep_id entity/project/abc123   # Resume existing sweep
#
# The sweep dashboard (parallel coordinates, parameter importance, etc.)
# is available at wandb.ai in real time.

set -euo pipefail
cd "$(dirname "$0")"

# --- Proxy setup ---
# Configure your proxy settings if needed.
# export http_proxy=http://your-proxy:port
# export https_proxy=http://your-proxy:port
# export HTTP_PROXY=http://your-proxy:port
# export HTTPS_PROXY=http://your-proxy:port
# export no_proxy=localhost,127.0.0.1
# export NO_PROXY=localhost,127.0.0.1

# Ensure wandb is NOT in offline mode
unset WANDB_MODE 2>/dev/null || true

echo "[run_sweep] Starting wandb online sweep"
echo "[run_sweep] Proxy: $http_proxy"
echo "[run_sweep] no_proxy: $no_proxy"
echo ""

python sweep_agent.py "$@"
