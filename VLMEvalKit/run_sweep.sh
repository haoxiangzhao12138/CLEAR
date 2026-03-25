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
# wandb needs internet access through the corporate proxy.
# Evaluation subprocess (torchrun) will have proxy vars stripped automatically
# by sweep_agent.py, so it always runs on the intranet.
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export HTTP_PROXY=http://agent.baidu.com:8891
export HTTPS_PROXY=http://agent.baidu.com:8891
export no_proxy=baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com
export NO_PROXY=baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com

# Ensure wandb is NOT in offline mode
unset WANDB_MODE 2>/dev/null || true

echo "[run_sweep] Starting wandb online sweep"
echo "[run_sweep] Proxy: $http_proxy"
echo "[run_sweep] no_proxy: $no_proxy"
echo ""

python sweep_agent.py "$@"
