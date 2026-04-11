#!/usr/bin/env python
"""Wandb online sweep agent for BagelInference hyperparameter tuning.

Uses wandb sweep (online mode) for Bayesian optimization and experiment tracking.
The sweep dashboard (parallel coordinates, parameter importance, etc.) is
available at wandb.ai in real time.

Proxy strategy:
    - The parent process keeps http(s)_proxy set so that wandb can reach the
      internet through the corporate proxy.
    - no_proxy covers all intranet domains/IPs so that local model inference
      is not affected.
    - The evaluation subprocess (torchrun) inherits a *clean* environment
      with proxy vars removed, ensuring it always uses the intranet directly.

Usage:
    # Create a new sweep and run 50 trials
    python sweep_agent.py

    # Custom trial count
    python sweep_agent.py --count 100

    # Resume an existing sweep (grab the sweep ID from wandb UI)
    python sweep_agent.py --sweep_id <ENTITY/PROJECT/SWEEP_ID>
"""

import argparse
import copy
import json
import os
import subprocess
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
WANDB_PROJECT = "bagel-sweep"

# Proxy for wandb to reach the internet
PROXY_URL = os.getenv("HTTP_PROXY", "")
NO_PROXY = os.getenv("NO_PROXY", "localhost,127.0.0.1")

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
    "repetition_penalty",
    "cfg_text_scale",
    "cfg_img_scale",
    "timestep_shift",
    "num_timesteps",
    "cfg_renorm_min",
    "consider_think",
]

# Env vars related to proxy
PROXY_ENV_KEYS = [
    "http_proxy", "HTTP_PROXY",
    "https_proxy", "HTTPS_PROXY",
    "no_proxy", "NO_PROXY",
]


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------

def setup_proxy():
    """Set proxy env vars so wandb (and other libs) can reach the internet."""
    os.environ["http_proxy"] = PROXY_URL
    os.environ["https_proxy"] = PROXY_URL
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    os.environ["no_proxy"] = NO_PROXY
    os.environ["NO_PROXY"] = NO_PROXY


def make_clean_env():
    """Return a copy of os.environ with all proxy vars removed.

    Used for the evaluation subprocess so it runs purely on the intranet.
    """
    env = os.environ.copy()
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    return env


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------

def build_config(params: dict, trial_number: int) -> tuple[dict, str]:
    """Build an evaluation config dict from the base template + trial params."""
    with open(BASE_CONFIG_PATH, "r") as f:
        base = json.load(f)

    model_name = f"BAGEL_sweep_trial{trial_number}"

    orig_model_cfg = list(base["model"].values())[0]
    model_cfg = copy.deepcopy(orig_model_cfg)

    for key in MODEL_HYPERPARAMS:
        if key in params:
            model_cfg[key] = params[key]

    if "cfg_interval_low" in params and "cfg_interval_high" in params:
        model_cfg["cfg_interval"] = [
            params["cfg_interval_low"],
            params["cfg_interval_high"],
        ]

    config = {
        "model": {model_name: model_cfg},
        "data": base["data"],
    }
    return config, model_name


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(config_path: str) -> int:
    """Launch the evaluation via torchrun and return the process exit code.

    The subprocess runs with proxy env vars stripped so that it stays on the
    intranet for model inference.
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
    print(f"[sweep] Running: {' '.join(cmd)}")
    clean_env = make_clean_env()
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, env=clean_env)
    return result.returncode


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def parse_scores(model_name: str) -> dict[str, float]:
    """Parse benchmark scores from output CSVs."""
    scores = {}
    model_dir = os.path.join(WORK_DIR, model_name)

    if not os.path.isdir(model_dir):
        print(f"[sweep] WARNING: output dir not found: {model_dir}")
        return scores

    for dataset_name, bench_type in BENCHMARK_TYPES.items():
        try:
            if bench_type == "mcq":
                acc_file = os.path.join(
                    model_dir, f"{model_name}_{dataset_name}_acc.csv"
                )
                if not os.path.exists(acc_file):
                    print(f"[sweep] WARNING: missing {acc_file}, skipping")
                    continue
                df = pd.read_csv(acc_file)
                score = float(df["Overall"].iloc[0]) * 100
                scores[dataset_name] = score

            elif bench_type == "mmvet":
                score_file = os.path.join(
                    model_dir,
                    f"{model_name}_{dataset_name}_{JUDGE}_score.csv",
                )
                if not os.path.exists(score_file):
                    print(f"[sweep] WARNING: missing {score_file}, skipping")
                    continue
                df = pd.read_csv(score_file)
                row = df[df["Category"] == "Overall"]
                score = float(row["acc"].iloc[0])
                scores[dataset_name] = score

        except Exception as e:
            print(f"[sweep] ERROR parsing {dataset_name}: {e}")
            continue

    return scores


# ---------------------------------------------------------------------------
# Wandb sweep train function
# ---------------------------------------------------------------------------

# Global trial counter (wandb.agent calls train() repeatedly)
_trial_counter = 0


def train():
    """Single trial function called by wandb.agent().

    wandb.agent() handles parameter sampling via the sweep controller.
    This function:
      1. Reads params from wandb.config (provided by the sweep controller)
      2. Builds eval config and runs evaluation (on the intranet, no proxy)
      3. Logs scores back to wandb (through the proxy)
    """
    global _trial_counter
    trial_number = _trial_counter
    _trial_counter += 1

    # wandb.init() is called by wandb.agent() before entering train(),
    # but we call it explicitly to set the run name.
    run = wandb.init(
        name=f"trial-{trial_number}",
        reinit=True,
    )

    # Read hyperparams from the sweep controller
    params = dict(wandb.config)
    print(f"\n{'='*60}")
    print(f"[sweep] Trial {trial_number}")
    print(f"[sweep] Params: {json.dumps(params, indent=2, default=str)}")
    print(f"{'='*60}\n")

    # Build temporary config file
    eval_config, model_name = build_config(params, trial_number)

    tmp_config_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=f"sweep_trial{trial_number}_",
            dir=SCRIPT_DIR, delete=False,
        ) as tmp:
            json.dump(eval_config, tmp, indent=4)
            tmp_config_path = tmp.name

        # Run evaluation (subprocess with proxy stripped)
        exit_code = run_evaluation(tmp_config_path)
        if exit_code != 0:
            print(f"[sweep] WARNING: evaluation exited with code {exit_code}")

        # Parse scores
        scores = parse_scores(model_name)
        print(f"[sweep] Parsed scores: {scores}")

        if scores:
            avg_score = sum(scores.values()) / len(scores)
        else:
            avg_score = 0.0
            print("[sweep] WARNING: no scores parsed, returning avg_score=0")

        # Log to wandb (proxy is set in parent process, wandb can reach internet)
        log_dict = {"avg_score": avg_score}
        for dataset_name, score in scores.items():
            log_dict[dataset_name] = score
        log_dict["num_benchmarks_parsed"] = len(scores)
        wandb.log(log_dict)

        print(f"[sweep] Trial {trial_number}: avg_score={avg_score:.2f} "
              f"({len(scores)}/{len(BENCHMARK_TYPES)} benchmarks)")

    finally:
        if tmp_config_path and os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)

    wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wandb online sweep for BagelInference hyperparameter tuning"
    )
    parser.add_argument(
        "--count", type=int, default=50,
        help="Number of trials to run in this agent (default: 50)",
    )
    parser.add_argument(
        "--sweep_id", type=str, default=None,
        help="Existing wandb sweep ID to resume (e.g. entity/project/sweep_id). "
             "If not provided, a new sweep is created from sweep_config.yaml.",
    )
    parser.add_argument(
        "--entity", type=str, default=None,
        help="Wandb entity (team or username). Uses default if not specified.",
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(SCRIPT_DIR, "sweep_config.yaml"),
        help="Path to sweep_config.yaml (default: ./sweep_config.yaml)",
    )
    args = parser.parse_args()

    # Set up proxy so wandb can reach the internet
    setup_proxy()
    # Remove offline mode if previously set
    os.environ.pop("WANDB_MODE", None)

    if args.sweep_id:
        # Resume an existing sweep
        sweep_id = args.sweep_id
        print(f"[sweep] Resuming existing sweep: {sweep_id}")
    else:
        # Create a new sweep from config
        import yaml
        with open(args.config, "r") as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=WANDB_PROJECT,
            entity=args.entity,
        )
        print(f"[sweep] Created new sweep: {sweep_id}")

    print(f"[sweep] Running {args.count} trials")
    print(f"[sweep] Proxy: {PROXY_URL} (no_proxy: {NO_PROXY})")
    print(f"[sweep] Evaluation subprocess runs WITHOUT proxy (intranet only)")
    print(f"[sweep] View sweep dashboard at: https://wandb.ai/sweep/{sweep_id}")

    # Start the agent — this blocks until `count` trials are done
    wandb.agent(sweep_id, function=train, count=args.count)

    print(f"\n{'='*60}")
    print("[sweep] All trials complete!")
    print(f"[sweep] View results at: https://wandb.ai/sweep/{sweep_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
