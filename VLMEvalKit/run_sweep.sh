#!/bin/bash
# run_sweep.sh — Initialize a wandb sweep and start the agent
#
# Wandb communicates with api.wandb.ai via HTTPS proxy, while model evaluation
# runs directly on the internal network (proxy env vars are stripped in sweep_agent.py).
#
# Usage:
#   # ============================================================
#   # STEP 0: Set your proxy (REQUIRED, edit before first use)
#   # ============================================================
#   # Option A: HTTP proxy
#   #   export HTTPS_PROXY=http://your-proxy-host:port
#   #
#   # Option B: SOCKS5 proxy (e.g. via SSH tunnel)
#   #   ssh -D 1080 -fNq user@jump-server
#   #   export HTTPS_PROXY=socks5://127.0.0.1:1080
#   #
#   # ============================================================
#   # STEP 1: Run sweep
#   # ============================================================
#   bash run_sweep.sh                  # Create new sweep + start agent
#   bash run_sweep.sh <sweep_id>       # Attach agent to existing sweep

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------
# Proxy configuration for wandb (Baidu internal proxy)
# ---------------------------------------------------------------
export http_proxy="http://agent.baidu.com:8891"
export https_proxy="http://agent.baidu.com:8891"
export HTTP_PROXY="http://agent.baidu.com:8891"
export HTTPS_PROXY="http://agent.baidu.com:8891"
export no_proxy="baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com"
export NO_PROXY="baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com"

echo "[run_sweep] Proxy: $HTTPS_PROXY"
echo "[run_sweep] no_proxy: $no_proxy"

if [ $# -ge 1 ]; then
    SWEEP_ID="$1"
    echo "[run_sweep] Using existing sweep: $SWEEP_ID"
else
    echo "[run_sweep] Creating new sweep from sweep_config.yaml ..."
    SWEEP_ID=$(wandb sweep sweep_config.yaml 2>&1 | grep -oP 'wandb agent \K\S+')
    echo "[run_sweep] Created sweep: $SWEEP_ID"
fi

echo "[run_sweep] Starting agent ..."
echo "[run_sweep] (wandb uses proxy; torchrun runs on internal network)"
echo ""
wandb agent "$SWEEP_ID"
