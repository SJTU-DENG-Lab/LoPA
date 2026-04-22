#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_PATH=${1:-"/home/chenkai/data/models/DiffuCoder-7B-Instruct"}
TP=${2:-1}
OUTPUT_DIR=${3:-"results/diffucoder_basic_len256"}
DEVICE=${4:-"cuda:0"}
LENGTH=${5:-256}

DATASET=humaneval
RUN_DIR_NAME="diffucoder_basic_chat_temp_0.0"

mkdir -p "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}"

export HF_ENDPOINT=https://hf-mirror.com
export PATH=./vllm/bin:$PATH

echo "EvalPlus basic DiffuCoder: base=${BASE_MODEL_PATH}, dataset=${DATASET}, length=${LENGTH}, OUTPUT_DIR=${OUTPUT_DIR}"

python generate.py \
  --model_type diffucoder_basic \
  --model_size chat \
  --model_path "${BASE_MODEL_PATH}" \
  --bs 1 \
  --temperature 0 \
  --n_samples 1 \
  --greedy \
  --root "${OUTPUT_DIR}" \
  --dataset "${DATASET}" \
  --save-dir "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}" \
  --tensor-parallel-size "${TP}" \
  --device "${DEVICE}" \
  --max-new-tokens "${LENGTH}" \
  --basic-token-per-step 1

python -m evalplus.sanitize --samples "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}"

evalplus.evaluate \
  --dataset "${DATASET}" \
  --samples "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}" > "${OUTPUT_DIR}/raw_${DATASET}_results.txt"

evalplus.evaluate \
  --dataset "${DATASET}" \
  --samples "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}-sanitized" > "${OUTPUT_DIR}/${DATASET}_results.txt"
