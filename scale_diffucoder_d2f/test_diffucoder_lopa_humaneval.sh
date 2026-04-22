#!/usr/bin/env bash
set -euo pipefail

# Self-contained parallel evaluation script.
# This file embeds model paths and a small hyperparameter grid. It runs generate.py
# for every combo and writes results into OUTPUT_DIR/<dataset>/<run_dir_name>.

# --- User-editable configuration (no CLI args required) ---
BASE_MODEL_PATH="/home/chenkai/data/models/DiffuCoder-7B-Instruct"
LORA_PATH="/home/chenkai/data/models/D2F_DiffuCoder_Instruct_7B_Lora"
TP=1
OUTPUT_DIR="results/diffucoder_lopa_32_0.95_0.95_0.3_new"
DEVICE="cuda:4"

# Datasets to run (must be supported by evalplus)
DATASETS=(humaneval)

# Hyperparameter grid: edit these arrays to add/remove values
BRANCH_FACTORS=(6)
BRANCH_TOPPS=(1)
SELECTION_ALPHAS=(0)
TOP_PS=("")
TOP_KS=("")
DISABLE_VERIFICATION_OPTIONS=(0)
DISABLE_BASE_COMPETITION_OPTIONS=(0)
FORCE_BASE_WINNER_OPTIONS=(0)
DISABLE_UNCERTAINTY_OPTIONS=(0)

# [NEW] Generation Hyperparameters (Not grid searched, passed directly)
BLOCK_SIZE=32
BLOCK_ADD_THRESHOLD=0.3
SKIP_THRESHOLD=0.95
DECODED_TOKEN_THRESHOLD=0.95

# Optional run tag to identify this overall experiment batch
GLOBAL_RUN_TAG="batch1"

# --- End of user-editable configuration ---

mkdir -p "${OUTPUT_DIR}"
for ds in "${DATASETS[@]}"; do
  mkdir -p "${OUTPUT_DIR}/${ds}"
done

export HF_ENDPOINT=https://hf-mirror.com
export PATH=./vllm/bin:$PATH

MODEL_COMBINED_PATH="${BASE_MODEL_PATH},${LORA_PATH}"

fmt_float() {
  python - "$1" <<'PY'
import sys
val = float(sys.argv[1])
s = ("{:.3f}".format(val)).rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")
print(s)
PY
}

run_count=0
for BF in "${BRANCH_FACTORS[@]}"; do
  for Topp in "${BRANCH_TOPPS[@]}"; do
    for Alpha in "${SELECTION_ALPHAS[@]}"; do
      for TopP in "${TOP_PS[@]}"; do
        for TopK in "${TOP_KS[@]}"; do
          for NoVer in "${DISABLE_VERIFICATION_OPTIONS[@]}"; do
            for NoBaseComp in "${DISABLE_BASE_COMPETITION_OPTIONS[@]}"; do
              for ForceBase in "${FORCE_BASE_WINNER_OPTIONS[@]}"; do
                for NoUnc in "${DISABLE_UNCERTAINTY_OPTIONS[@]}"; do

                  # Build run suffix from active options
                  suffix_parts=("bf${BF}" "topp$(fmt_float "${Topp}")" "alpha$(fmt_float "${Alpha}")")
                  if [[ -n "${TopP}" ]]; then
                    suffix_parts+=("tp$(fmt_float "${TopP}")")
                  fi
                  if [[ -n "${TopK}" ]]; then
                    suffix_parts+=("tk${TopK}")
                  fi
                  [[ ${NoVer} -eq 1 ]] && suffix_parts+=("noVer")
                  [[ ${NoBaseComp} -eq 1 ]] && suffix_parts+=("noBaseComp")
                  [[ ${ForceBase} -eq 1 ]] && suffix_parts+=("forceBase")
                  [[ ${NoUnc} -eq 1 ]] && suffix_parts+=("noUnc")
                  [[ -n "${GLOBAL_RUN_TAG}" ]] && suffix_parts+=("${GLOBAL_RUN_TAG}")

                  RUN_SUFFIX=""
                  if [[ ${#suffix_parts[@]} -gt 0 ]]; then
                    RUN_SUFFIX=_"$(IFS=_; echo "${suffix_parts[*]}")"
                  fi

                  RUN_DIR_NAME="diffucoder_parallel_chat_temp_0.0${RUN_SUFFIX}"

                  echo "=============================================="
                  echo "Run #$((run_count+1)): BF=${BF}, Topp=${Topp}, Alpha=${Alpha}, TopP=${TopP}, TopK=${TopK}, NoVer=${NoVer}, NoBaseComp=${NoBaseComp}, ForceBase=${ForceBase}, NoUnc=${NoUnc}"
                  echo "Output dir suffix: ${RUN_DIR_NAME}"

                  GENERIC_ARGS=(
                    --model_type diffucoder_parallel
                    --model_size chat
                    --model_path "${MODEL_COMBINED_PATH}"
                    --bs 1
                    --temperature 0
                    --n_samples 1
                    --greedy
                    --root "${OUTPUT_DIR}"
                    --tensor-parallel-size "${TP}"
                    --device "${DEVICE}"
                    --run-tag "${GLOBAL_RUN_TAG}"
                    --parallel-branching-factor "${BF}"
                    --parallel-branch-topp "${Topp}"
                    --parallel-selection-alpha "${Alpha}"
                    
                    # [NEW] Pass generation hyperparameters
                    --block-size "${BLOCK_SIZE}"
                    --block-add-threshold "${BLOCK_ADD_THRESHOLD}"
                    --skip-threshold "${SKIP_THRESHOLD}"
                    --decoded-token-threshold "${DECODED_TOKEN_THRESHOLD}"
                  )

                  if [[ -n "${TopP}" ]]; then
                    GENERIC_ARGS+=(--parallel-top-p "${TopP}")
                  fi
                  if [[ -n "${TopK}" ]]; then
                    GENERIC_ARGS+=(--parallel-top-k "${TopK}")
                  fi
                  if [[ ${NoVer} -eq 1 ]]; then
                    GENERIC_ARGS+=(--parallel-disable-verification)
                  fi
                  if [[ ${NoBaseComp} -eq 1 ]]; then
                    GENERIC_ARGS+=(--parallel-disable-base-competition)
                  fi
                  if [[ ${ForceBase} -eq 1 ]]; then
                    GENERIC_ARGS+=(--parallel-force-base-winner)
                  fi
                  if [[ ${NoUnc} -eq 1 ]]; then
                    GENERIC_ARGS+=(--parallel-disable-uncertainty)
                  fi

                  for DATASET in "${DATASETS[@]}"; do
                    mkdir -p "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}"
                    python generate.py \
                      "${GENERIC_ARGS[@]}" \
                      --dataset "${DATASET}" \
                      --save-dir "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}"

                    python -m evalplus.sanitize --samples "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}"

                    evalplus.evaluate \
                      --dataset "${DATASET}" \
                      --samples "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}" > "${OUTPUT_DIR}/raw_${DATASET}_${RUN_DIR_NAME}_results.txt"

                    evalplus.evaluate \
                      --dataset "${DATASET}" \
                      --samples "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}-sanitized" > "${OUTPUT_DIR}/${DATASET}_${RUN_DIR_NAME}_results.txt"
                  done

                  run_count=$((run_count+1))

                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Completed ${run_count} runs. Results under ${OUTPUT_DIR}."
