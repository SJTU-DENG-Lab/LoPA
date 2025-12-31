#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_PATH=${1:-"DiffuCoder-7B-Instruct"}
LORA_PATH=${2:-"SJTU-Deng-Lab/D2F_DiffuCoder_Instruct_7B_Lora"}
TP=${3:-1}
OUTPUT_DIR=${4:-"results/diffucoder_0.3_0.95_0.95_new"}
DEVICE=${5:-"cuda:0"}

# [NEW] Additional knobs inferred from output-dir naming defaults
BLOCK_SIZE=32
BLOCK_ADD_THRESHOLD=0.3
SKIP_THRESHOLD=0.95
DECODED_TOKEN_THRESHOLD=0.95

mkdir -p "${OUTPUT_DIR}"/humaneval
mkdir -p "${OUTPUT_DIR}"/mbpp

export HF_ENDPOINT=https://hf-mirror.com
export PATH=./vllm/bin:$PATH
# export PYTHONPATH=$PYTHONPATH:./eval_plus/evalplus
# pip install datamodel_code_generator anthropic mistralai google-generativeai

MODEL_COMBINED_PATH="${BASE_MODEL_PATH},${LORA_PATH}"
RUN_DIR_NAME="diffucoder_chat_temp_0.0"

echo "EvalPlus: base=${BASE_MODEL_PATH}, lora=${LORA_PATH}, OUTPUT_DIR=${OUTPUT_DIR}"

# for DATASET in humaneval mbpp; do
for DATASET in humaneval; do
  python generate.py \
    --model_type diffucoder \
    --model_size chat \
    --model_path "${MODEL_COMBINED_PATH}" \
    --bs 1 \
    --temperature 0 \
    --n_samples 1 \
    --greedy \
    --root "${OUTPUT_DIR}" \
    --dataset "${DATASET}" \
    --tensor-parallel-size "${TP}" \
    --device "${DEVICE}" \
    --block-size "${BLOCK_SIZE}" \
    --block-add-threshold "${BLOCK_ADD_THRESHOLD}" \
    --skip-threshold "${SKIP_THRESHOLD}" \
    --decoded-token-threshold "${DECODED_TOKEN_THRESHOLD}"

  python -m evalplus.sanitize --samples "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}"

  evalplus.evaluate \
    --dataset "${DATASET}" \
    --samples "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}" > "${OUTPUT_DIR}/raw_${DATASET}_results.txt"

  evalplus.evaluate \
    --dataset "${DATASET}" \
    --samples "${OUTPUT_DIR}/${DATASET}/${RUN_DIR_NAME}-sanitized" > "${OUTPUT_DIR}/${DATASET}_results.txt"
done
