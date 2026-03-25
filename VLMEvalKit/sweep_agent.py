#!/usr/bin/env python
"""Wandb Sweep Agent for BagelInference hyperparameter tuning.

This script is invoked by `wandb agent`. Each run:
1. Reads hyperparameters from wandb.config
2. Generates a temporary config JSON (based on config/test.json)
3. Runs evaluation via torchrun (no proxy needed, runs locally)
4. Parses benchmark scores from output CSVs
5. Reports avg_score (and per-benchmark scores) to wandb (via proxy)

Proxy setup:
  wandb communicates with api.wandb.ai via HTTPS_PROXY set in run_sweep.sh.
  The torchrun evaluation subprocess clears proxy env vars so model inference
  stays on the internal network without going through the proxy.
"""

import copy
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "test.json")
WORK_DIR = os.path.join(SCRIPT_DIR, "outputs")
JUDGE = "gpt-4-0125"

# Proxy env vars to strip from torchrun subprocess so model inference
# stays on internal network
PROXY_ENV_KEYS = [
    "HTTP_PROXY", "http_proxy",
    "HTTPS_PROXY", "https_proxy",
    "ALL_PROXY", "all_proxy",
]

# Benchmark type mapping: dataset_name -> "mcq" or "mmvet"
BENCHMARK_TYPES = {
    "MMBench_DEV_EN_V11_LOW_LEVEL_HIGH": "mcq",
    "MMVet_LOW_LEVEL_HIGH": "mmvet",
    "MMVP_LOW_LEVEL_HIGH": "mcq",
    "CV-Bench-2D_LOW_LEVEL_HIGH": "mcq",
    "MMStar_LOW_LEVEL_HIGH": "mcq",
    "RealWorldQA_LOW_LEVEL_HIGH": "mcq",
    "R-Bench-Dis": "mcq",
}

# Hyperparameters that map directly into the model config dict
MODEL_HYPERPARAMS = [
    "text_temperature",
    "do_sample",
    "repetition_penalty",
    "max_think_token_n",
    "max_new_tokns",
    "is_thinking",
    "cfg_text_scale",
    "cfg_img_scale",
    "timestep_shift",
    "num_timesteps",
    "cfg_renorm_min",
    "consider_think",
    "output_need_vae",
    "output_need_vit",
    "max_inter_num",
]


def build_config(wandb_config, run_id: str) -> tuple[dict, str]:
    """Build an evaluation config dict from the base template + wandb hyperparams."""
    with open(BASE_CONFIG_PATH, "r") as f:
        base = json.load(f)

    # Use a unique model name to avoid output dir conflicts between sweep trials
    model_name = f"BAGEL_sweep_{run_id}"

    # Deep-copy the original model entry (assumed to be the only one: "BAGEL")
    orig_model_cfg = list(base["model"].values())[0]
    model_cfg = copy.deepcopy(orig_model_cfg)

    # Override with sweep hyperparams
    for key in MODEL_HYPERPARAMS:
        if key in wandb_config:
            model_cfg[key] = wandb_config[key]

    # Handle cfg_interval (split into low / high for wandb, combined as list for config)
    if "cfg_interval_low" in wandb_config and "cfg_interval_high" in wandb_config:
        model_cfg["cfg_interval"] = [
            wandb_config["cfg_interval_low"],
            wandb_config["cfg_interval_high"],
        ]

    config = {
        "model": {model_name: model_cfg},
        "data": base["data"],
    }
    return config, model_name


def run_evaluation(config_path: str) -> int:
    """Launch the evaluation via torchrun and return the process exit code.

    Strips proxy env vars from the subprocess so that model inference and
    internal network access are not affected by the wandb proxy.
    """
    cmd = [
        "torchrun",
        "--nproc-per-node=8",
        "--master_port=29503",
        "run.py",
        "--config", config_path,
        "--judge", JUDGE,
        "--verbose",
    ]
    # Build a clean env without proxy settings for the evaluation subprocess
    clean_env = {k: v for k, v in os.environ.items() if k not in PROXY_ENV_KEYS}

    print(f"[sweep_agent] Running: {' '.join(cmd)}")
    print(f"[sweep_agent] (proxy env vars stripped for torchrun subprocess)")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, env=clean_env)
    return result.returncode


def parse_scores(model_name: str) -> dict[str, float]:
    """Parse benchmark scores from output CSVs.

    Returns a dict of {benchmark_name: score} where score is 0-100 scale.
    """
    scores = {}
    model_dir = os.path.join(WORK_DIR, model_name)

    if not os.path.isdir(model_dir):
        print(f"[sweep_agent] WARNING: output dir not found: {model_dir}")
        return scores

    for dataset_name, bench_type in BENCHMARK_TYPES.items():
        try:
            if bench_type == "mcq":
                # MCQ benchmarks: {model}_{dataset}_acc.csv
                acc_file = os.path.join(
                    model_dir, f"{model_name}_{dataset_name}_acc.csv"
                )
                if not os.path.exists(acc_file):
                    print(f"[sweep_agent] WARNING: missing {acc_file}, skipping")
                    continue
                df = pd.read_csv(acc_file)
                # Overall column is 0-1 scale, convert to 0-100
                score = float(df["Overall"].iloc[0]) * 100
                scores[dataset_name] = score

            elif bench_type == "mmvet":
                # MMVet: {model}_{dataset}_{judge}_score.csv
                score_file = os.path.join(
                    model_dir,
                    f"{model_name}_{dataset_name}_{JUDGE}_score.csv",
                )
                if not os.path.exists(score_file):
                    print(f"[sweep_agent] WARNING: missing {score_file}, skipping")
                    continue
                df = pd.read_csv(score_file)
                row = df[df["Category"] == "Overall"]
                # acc column is already 0-100 scale
                score = float(row["acc"].iloc[0])
                scores[dataset_name] = score

        except Exception as e:
            print(f"[sweep_agent] ERROR parsing {dataset_name}: {e}")
            continue

    return scores


def main():
    # Initialize wandb run (communicates with server via HTTPS_PROXY)
    run = wandb.init()
    config = dict(wandb.config)
    run_id = run.id

    print(f"[sweep_agent] Starting sweep trial {run_id}")
    print(f"[sweep_agent] Hyperparameters: {json.dumps(config, indent=2, default=str)}")

    # Build temporary config file
    eval_config, model_name = build_config(config, run_id)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="sweep_config_", dir=SCRIPT_DIR, delete=False
    ) as tmp:
        json.dump(eval_config, tmp, indent=4)
        tmp_config_path = tmp.name

    print(f"[sweep_agent] Temp config: {tmp_config_path}")
    print(f"[sweep_agent] Model name: {model_name}")

    try:
        # Run evaluation (proxy stripped, uses internal network)
        exit_code = run_evaluation(tmp_config_path)
        if exit_code != 0:
            print(f"[sweep_agent] WARNING: evaluation exited with code {exit_code}")

        # Parse scores
        scores = parse_scores(model_name)
        print(f"[sweep_agent] Parsed scores: {scores}")

        if scores:
            avg_score = sum(scores.values()) / len(scores)
        else:
            avg_score = 0.0
            print("[sweep_agent] WARNING: no scores parsed, reporting avg_score=0")

        # Log to wandb (goes through proxy to wandb server)
        log_dict = {"avg_score": avg_score}
        for dataset_name, score in scores.items():
            log_dict[dataset_name] = score
        log_dict["num_benchmarks_parsed"] = len(scores)

        wandb.log(log_dict)
        print(f"[sweep_agent] Reported avg_score={avg_score:.2f} "
              f"({len(scores)}/{len(BENCHMARK_TYPES)} benchmarks)")

    finally:
        # Clean up temp config
        if os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)
            print(f"[sweep_agent] Removed temp config: {tmp_config_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
