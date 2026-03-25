#!/usr/bin/env python
"""Optuna + Wandb offline sweep agent for BagelInference hyperparameter tuning.

Uses Optuna for local Bayesian optimization (TPE sampler) and wandb in offline
mode for experiment tracking. After all trials complete, run `wandb sync` from
an internet-connected environment to upload results.

Usage:
    # Run with defaults (50 trials)
    python sweep_agent.py

    # Custom number of trials
    python sweep_agent.py --n_trials 100

    # Resume from a previous Optuna study
    python sweep_agent.py --study_name my_study --storage sqlite:///optuna.db
"""

import argparse
import copy
import json
import os
import subprocess
import tempfile

import optuna
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


# ---------------------------------------------------------------------------
# Optuna parameter sampling
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial) -> dict:
    """Sample hyperparameters using Optuna's Bayesian (TPE) sampler."""
    params = {
        "text_temperature": trial.suggest_float("text_temperature", 0.1, 1.0),
        "do_sample": trial.suggest_categorical("do_sample", [True, False]),
        "repetition_penalty": trial.suggest_float("repetition_penalty", 1.0, 1.5),
        "max_think_token_n": trial.suggest_int("max_think_token_n", 2048, 8192),
        "max_new_tokns": trial.suggest_int("max_new_tokns", 512, 2048),
        "is_thinking": trial.suggest_categorical("is_thinking", [True, False]),
        "cfg_text_scale": trial.suggest_float("cfg_text_scale", 1.0, 7.0),
        "cfg_img_scale": trial.suggest_float("cfg_img_scale", 1.0, 3.0),
        "cfg_interval_low": trial.suggest_float("cfg_interval_low", 0.0, 0.6),
        "cfg_interval_high": trial.suggest_float("cfg_interval_high", 0.6, 1.0),
        "timestep_shift": trial.suggest_float("timestep_shift", 1.0, 6.0),
        "num_timesteps": trial.suggest_int("num_timesteps", 20, 100),
        "cfg_renorm_min": trial.suggest_float("cfg_renorm_min", 0.0, 1.0),
        "consider_think": trial.suggest_categorical("consider_think", [True, False]),
        "output_need_vae": trial.suggest_categorical("output_need_vae", [True, False]),
        "output_need_vit": trial.suggest_categorical("output_need_vit", [True, False]),
        "max_inter_num": trial.suggest_int("max_inter_num", 1, 5),
    }
    return params


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------

def build_config(params: dict, trial_number: int) -> tuple[dict, str]:
    """Build an evaluation config dict from the base template + trial params."""
    with open(BASE_CONFIG_PATH, "r") as f:
        base = json.load(f)

    # Use a unique model name to avoid output dir conflicts between trials
    model_name = f"BAGEL_sweep_trial{trial_number}"

    # Deep-copy the original model entry
    orig_model_cfg = list(base["model"].values())[0]
    model_cfg = copy.deepcopy(orig_model_cfg)

    # Override with trial hyperparams
    for key in MODEL_HYPERPARAMS:
        if key in params:
            model_cfg[key] = params[key]

    # Handle cfg_interval (split into low/high for search, combined as list for config)
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
    """Launch the evaluation via torchrun and return the process exit code."""
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
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def parse_scores(model_name: str) -> dict[str, float]:
    """Parse benchmark scores from output CSVs.

    Returns a dict of {benchmark_name: score} where score is 0-100 scale.
    """
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
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function: run one evaluation trial and return avg_score."""
    params = suggest_params(trial)
    trial_number = trial.number

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

        # Initialize wandb run in offline mode
        os.environ["WANDB_MODE"] = "offline"
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"trial-{trial_number}",
            config=params,
            reinit=True,
        )

        # Run evaluation
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

        # Log to wandb (offline)
        log_dict = {"avg_score": avg_score}
        for dataset_name, score in scores.items():
            log_dict[dataset_name] = score
        log_dict["num_benchmarks_parsed"] = len(scores)
        wandb.log(log_dict)

        print(f"[sweep] Trial {trial_number}: avg_score={avg_score:.2f} "
              f"({len(scores)}/{len(BENCHMARK_TYPES)} benchmarks)")

        wandb.finish()
        return avg_score

    finally:
        if tmp_config_path and os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna + wandb offline sweep")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of optimization trials (default: 50)")
    parser.add_argument("--study_name", type=str, default="bagel_sweep",
                        help="Optuna study name (for resuming)")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL, e.g. sqlite:///optuna.db "
                             "(default: in-memory, not resumable)")
    args = parser.parse_args()

    # Force wandb offline
    os.environ["WANDB_MODE"] = "offline"

    # Create or load Optuna study with TPE (Bayesian) sampler
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True,
    )

    print(f"[sweep] Starting Optuna study '{args.study_name}' "
          f"with {args.n_trials} trials")
    print(f"[sweep] Wandb mode: offline (use `wandb sync` to upload later)")

    study.optimize(objective, n_trials=args.n_trials)

    # Print summary
    print(f"\n{'='*60}")
    print("[sweep] Optimization complete!")
    print(f"[sweep] Best trial: #{study.best_trial.number}")
    print(f"[sweep] Best avg_score: {study.best_value:.2f}")
    print(f"[sweep] Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}")

    # Save best params to JSON
    best_params_path = os.path.join(SCRIPT_DIR, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_avg_score": study.best_value,
            "best_params": study.best_params,
        }, f, indent=4)
    print(f"[sweep] Best params saved to {best_params_path}")

    # Remind about wandb sync
    wandb_dir = os.path.join(SCRIPT_DIR, "wandb")
    print(f"\n[sweep] To upload results to wandb, run from an internet-connected env:")
    print(f"  cd {SCRIPT_DIR}")
    print(f"  wandb sync --sync-all {wandb_dir}")


if __name__ == "__main__":
    main()
