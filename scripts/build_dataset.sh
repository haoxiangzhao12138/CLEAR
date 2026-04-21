#!/usr/bin/env bash
# =============================================================================
# CLEAR Dataset Construction Pipeline
# =============================================================================
# Orchestrates the full data build from raw LLaVA-OneVision data to
# training-ready JSONL + corrupted images, and optionally generates
# degraded VLMEvalKit benchmarks.
#
# Usage:
#   bash scripts/build_dataset.sh                  # run all stages
#   bash scripts/build_dataset.sh --stage 2        # run a single stage
#   bash scripts/build_dataset.sh --stage 2,3      # run specific stages
#   bash scripts/build_dataset.sh --dry-run        # print what would run
#
# Stages:
#   1  Sample from LLaVA-OneVision  (random_sample.py)
#   2  Apply image degradation      (process_degradation_images.py)
#   3a Generate interleave data     (generate_interleave_datasets_by_gpt.py)
#   3b Generate text CoT data       (generate_text_datasets_by_gpt.py)
#   4  Generate eval benchmarks     (generate_degradation_benchmark.py)
#   5  Generate per-degradation benchmarks (generate_per_degradation_benchmark.py)
# =============================================================================

set -euo pipefail

# ======================== Configurable paths ========================

# Source dataset (HuggingFace format on disk)
LLAVA_DATASET_PATH="./datasets/LLaVA-OneVision-Data"

# Output root
OUTPUT_ROOT="./datasets/processed_dataset"

# SFT paths (derived)
SFT_IMAGE_DIR="${OUTPUT_ROOT}/sft/images"
SFT_CORRUPTED_DIR="${OUTPUT_ROOT}/sft/corruption_images"
SFT_METADATA_DIR="${OUTPUT_ROOT}/sft/degradation_metadata"
SFT_DATA_JSONL="${OUTPUT_ROOT}/sft/sft_data.jsonl"

# RL paths (derived)
RL_IMAGE_DIR="${OUTPUT_ROOT}/rl/images"
RL_CORRUPTED_DIR="${OUTPUT_ROOT}/rl/corruption_images"
RL_METADATA_DIR="${OUTPUT_ROOT}/rl/degradation_metadata"

# GPT annotation outputs
INTERLEAVE_TOOL_JSONL="${OUTPUT_ROOT}/sft/agent_interleave_data_filtered_tool.jsonl"
INTERLEAVE_NO_TOOL_JSONL="${OUTPUT_ROOT}/sft/agent_interleave_data_filtered_no_tool.jsonl"
TEXT_COT_JSONL="${OUTPUT_ROOT}/sft/sft_pure_text.jsonl"

# Eval benchmark paths
LMUDATA_DIR="./LMUData"

# ======================== API config ========================

# Set these via environment or edit here
export OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_API_KEY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export MODEL_NAME="${MODEL_NAME:-gpt-4.1}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gpt-4.1}"

# ======================== Concurrency ========================

# GPT annotation workers (adjust based on your API rate limit)
GPT_INTERLEAVE_WORKERS=64
GPT_TEXT_WORKERS=64

# Text CoT sample size
TEXT_SAMPLE_SIZE=12000

# Benchmark degradation workers
BENCHMARK_WORKERS=64

# ======================== Script directory ========================

SCRIPT_DIR="data/corruption_datasets_create"

# ======================== Argument parsing ========================

DRY_RUN=false
STAGES="1,2,3a,3b,4,5"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGES="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --api-key)
            export OPENAI_API_KEY="$2"; shift 2 ;;
        --api-base)
            export OPENAI_API_BASE="$2"; shift 2 ;;
        --model)
            export MODEL_NAME="$2"; shift 2 ;;
        --llava-path)
            LLAVA_DATASET_PATH="$2"; shift 2 ;;
        --output-root)
            OUTPUT_ROOT="$2"; shift 2 ;;
        --help|-h)
            head -25 "$0" | tail -22
            exit 0 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ======================== Helpers ========================

run_stage() {
    local label="$1"
    shift
    echo ""
    echo "================================================================"
    echo "  [Stage ${label}] $*"
    echo "================================================================"
    if $DRY_RUN; then
        echo "  (dry-run, skipping)"
        return 0
    fi
}

should_run() {
    [[ "$STAGES" == *"$1"* ]]
}

elapsed() {
    local start=$1
    local end=$(date +%s)
    echo "  Completed in $(( end - start ))s"
}

# ======================== Stage 1: Sample ========================

if should_run "1"; then
    run_stage "1" "Sample from LLaVA-OneVision → SFT/RL splits"
    T0=$(date +%s)

    python "${SCRIPT_DIR}/random_sample.py" \
        2>&1 | tee "${OUTPUT_ROOT}/stage1_sample.log"

    elapsed $T0
    echo "  SFT images: $(ls "${SFT_IMAGE_DIR}" 2>/dev/null | wc -l)"
    echo "  RL  images: $(ls "${RL_IMAGE_DIR}" 2>/dev/null | wc -l)"
fi

# ======================== Stage 2: Degrade ========================

if should_run "2"; then
    run_stage "2" "Apply image degradation to SFT and RL images"
    T0=$(date +%s)

    echo "  [2a] Degrading SFT images..."
    python "${SCRIPT_DIR}/process_degradation_images.py" \
        --input_dir "${SFT_IMAGE_DIR}" \
        --output_dir "${SFT_CORRUPTED_DIR}" \
        --metadata_dir "${SFT_METADATA_DIR}" \
        2>&1 | tee "${OUTPUT_ROOT}/stage2a_degrade_sft.log"

    echo "  [2b] Degrading RL images..."
    python "${SCRIPT_DIR}/process_degradation_images.py" \
        --input_dir "${RL_IMAGE_DIR}" \
        --output_dir "${RL_CORRUPTED_DIR}" \
        --metadata_dir "${RL_METADATA_DIR}" \
        2>&1 | tee "${OUTPUT_ROOT}/stage2b_degrade_rl.log"

    elapsed $T0
fi

# ======================== Stage 3a: Interleave annotation ========================

if should_run "3a"; then
    run_stage "3a" "Generate interleave datasets via GPT (tool / no-tool split)"
    T0=$(date +%s)

    if [[ "${OPENAI_API_KEY}" == "YOUR_API_KEY" ]]; then
        echo "  ERROR: OPENAI_API_KEY not set. Skipping stage 3a."
        echo "  Set via: export OPENAI_API_KEY=sk-... or --api-key sk-..."
    else
        python "${SCRIPT_DIR}/generate_interleave_datasets_by_gpt.py" \
            2>&1 | tee "${OUTPUT_ROOT}/stage3a_interleave.log"

        echo "  Tool samples:    $(wc -l < "${INTERLEAVE_TOOL_JSONL}" 2>/dev/null || echo 0)"
        echo "  No-tool samples: $(wc -l < "${INTERLEAVE_NO_TOOL_JSONL}" 2>/dev/null || echo 0)"
    fi

    elapsed $T0
fi

# ======================== Stage 3b: Text CoT annotation ========================

if should_run "3b"; then
    run_stage "3b" "Generate text CoT datasets via GPT"
    T0=$(date +%s)

    if [[ "${OPENAI_API_KEY}" == "YOUR_API_KEY" ]]; then
        echo "  ERROR: OPENAI_API_KEY not set. Skipping stage 3b."
    else
        python "${SCRIPT_DIR}/generate_text_datasets_by_gpt.py" \
            2>&1 | tee "${OUTPUT_ROOT}/stage3b_text_cot.log"

        echo "  Text CoT samples: $(wc -l < "${TEXT_COT_JSONL}" 2>/dev/null || echo 0)"
    fi

    elapsed $T0
fi

# ======================== Stage 4: Eval benchmarks (mixed degradation) ========================

if should_run "4"; then
    run_stage "4" "Generate degraded VLMEvalKit benchmarks (3 intensity levels)"
    T0=$(date +%s)

    if [[ ! -d "${LMUDATA_DIR}" ]]; then
        echo "  WARNING: ${LMUDATA_DIR} not found. Skipping stage 4."
        echo "  Download LMUData first, or set LMUDATA_DIR."
    else
        python "${SCRIPT_DIR}/generate_degradation_benchmark.py" \
            --input_dir "${LMUDATA_DIR}" \
            --output_dir "${LMUDATA_DIR}" \
            --workers "${BENCHMARK_WORKERS}" \
            2>&1 | tee "${OUTPUT_ROOT}/stage4_benchmark.log"
    fi

    elapsed $T0
fi

# ======================== Stage 5: Per-degradation benchmarks ========================

if should_run "5"; then
    run_stage "5" "Generate per-degradation benchmarks (16 methods × 6 benchmarks)"
    T0=$(date +%s)

    if [[ ! -d "${LMUDATA_DIR}" ]]; then
        echo "  WARNING: ${LMUDATA_DIR} not found. Skipping stage 5."
    else
        python "${SCRIPT_DIR}/generate_per_degradation_benchmark.py" \
            --input_dir "${LMUDATA_DIR}" \
            --output_dir "${LMUDATA_DIR}" \
            --workers "${BENCHMARK_WORKERS}" \
            2>&1 | tee "${OUTPUT_ROOT}/stage5_per_degrade.log"
    fi

    elapsed $T0
fi

# ======================== Summary ========================

echo ""
echo "================================================================"
echo "  Pipeline complete!"
echo "================================================================"
echo ""
echo "  Output root: ${OUTPUT_ROOT}"
echo ""

if [[ -d "${SFT_IMAGE_DIR}" ]]; then
    echo "  SFT clean images:      $(ls "${SFT_IMAGE_DIR}" | wc -l)"
fi
if [[ -d "${SFT_CORRUPTED_DIR}" ]]; then
    echo "  SFT corrupted images:  $(ls "${SFT_CORRUPTED_DIR}" | wc -l)"
fi
if [[ -f "${INTERLEAVE_TOOL_JSONL}" ]]; then
    echo "  Interleave (tool):     $(wc -l < "${INTERLEAVE_TOOL_JSONL}") samples"
fi
if [[ -f "${INTERLEAVE_NO_TOOL_JSONL}" ]]; then
    echo "  Interleave (no-tool):  $(wc -l < "${INTERLEAVE_NO_TOOL_JSONL}") samples"
fi
if [[ -f "${TEXT_COT_JSONL}" ]]; then
    echo "  Text CoT:              $(wc -l < "${TEXT_COT_JSONL}") samples"
fi
echo ""
echo "  Training config ready at: data/configs/corruption_mix.yaml"
echo "================================================================"
