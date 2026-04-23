#!/bin/bash

# ==============================================================================
# 模型路径和通用配置
# ==============================================================================
model="/home/chenkai/data/models/Dream-v0-Instruct-7B"

export HF_ALLOW_CODE_EVAL=1
# export CURL_CA_BUNDLE=""
# export REQUESTS_CA_BUNDLE=""
export HF_ENDPOINT="https://hf-mirror.com"
# export HF_HOME="/mnt/rl/xinyi/LoPA"

SCRIPT_NAME="eval_dream_lopa_earlystopnew.py"


# ==============================================================================
# HumanEval 评测配置 (批量, 无括号)
# ==============================================================================
# 注意：HumanEval 有特殊的 model_args: escape_until=true
he_modes="true true true"
he_temps="0 0 0"
he_bfs="2 3 4"
he_bbcs="true true true"
he_vfbws="false false false"
he_btopps="1 1 1"
he_alphas="0 0 0"
he_limits="10000 10000 10000"
he_dtypes="bfloat16 bfloat16 bfloat16"

he_modes="true"
he_temps="0"
he_bfs="3"
he_bbcs="true"
he_vfbws="false"
he_btopps="1"
he_alphas="0"
he_limits="10000"
he_dtypes="bfloat16"



# ==============================================================================
# 主要任务评测配置 (批量, 无括号)
# ==============================================================================
tasks="gsm8k gsm8k gsm8k minerva_math minerva_math minerva_math"
nshots="4 4 4 4 4 4"
modes="true true true true true true"
lengths="256 256 256 256 256 256"
temperatures="0 0 0 0 0 0"
limits="10000 10000 10000 10000 10000 10000"
dtypes="bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16"
bfs="2 3 4 2 3 4"
bbcs="true true true true true true"
vfbws="false false false false false false"
btopps="1 1 1 1 1 1"
alphas="0 0 0 0 0 0"

tasks="gsm8k mbpp minerva_math"
nshots="4 3 4"
modes="true true true"
lengths="256 256 256"
temperatures="0 0 0"
limits="10000 10000 10000"
dtypes="bfloat16 bfloat16 bfloat16"
bfs="3 6 3"
bbcs="true true true"
vfbws="false false false"
btopps="1 1 1"
alphas="0 0 0"

tasks="minerva_math"
nshots="4"
modes="true"
lengths="256"
temperatures="0"
limits="10000"
dtypes="bfloat16"
bfs="3"
bbcs="true"
vfbws="false"
btopps="1"
alphas="0"




# tasks="gsm8k gsm8k gsm8k gsm8k gsm8k gsm8k mbpp mbpp mbpp mbpp mbpp mbpp"
# nshots="4 4 4 4 4 4 3 3 3 3 3 3"
# modes="true true true true true true true true true true true true"
# lengths="256 256 256 256 256 256 256 256 256 256 256 256"
# temperatures="0 0 0 0 0 0 0 0 0 0 0 0"
# limits="10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000"
# dtypes="bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16"
# bfs="5 6 7 8 9 10 5 6 7 8 9 10"
# bbcs="true true true true true true true true true true true true"
# vfbws="false false false false false false false false false false false false"
# btopps="1 1 1 1 1 1 1 1 1 1 1 1"
# alphas="0 0 0 0 0 0 0 0 0 0 0 0"

# tasks="minerva_math minerva_math minerva_math minerva_math minerva_math minerva_math minerva_math minerva_math minerva_math"
# nshots="4 4 4 4 4 4 4 4 4"
# modes="true true true true true true true true true"
# lengths="256 256 256 256 256 256 256 256 256"
# temperatures="0 0 0 0 0 0 0 0 0"
# limits="10000 10000 10000 10000 10000 10000 10000 10000 10000"
# dtypes="bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16"
# bfs="2 3 4 5 6 7 8 9 10"
# bbcs="true true true true true true true true true"
# vfbws="false false false false false false false false false"
# btopps="1 1 1 1 1 1 1 1 1"
# alphas="0 0 0 0 0 0 0 0 0"

# tasks="minerva_math"
# nshots="4"
# modes="true"
# lengths="256"
# temperatures="0"
# limits="10000"
# dtypes="bfloat16"
# bfs="2"
# bbcs="true"
# vfbws="false"
# btopps="1"
# alphas="0"



# ==============================================================================
# 1. 运行 HumanEval 批量评测
# ==============================================================================
read -ra HE_MODES_ARRAY <<< "${he_modes}"
read -ra HE_TEMPS_ARRAY <<< "${he_temps}"
read -ra HE_BFS_ARRAY <<< "${he_bfs}"
read -ra HE_BBCS_ARRAY <<< "${he_bbcs}"
read -ra HE_VFBWS_ARRAY <<< "${he_vfbws}"
read -ra HE_BTOPPS_ARRAY <<< "${he_btopps}"
read -ra HE_ALPHAS_ARRAY <<< "${he_alphas}"
read -ra HE_LIMITS_ARRAY <<< "${he_limits}"
read -ra HE_DTYPES_ARRAY <<< "${he_dtypes}"

# 验证 HumanEval 数组长度一致性
if [[ ${#HE_MODES_ARRAY[@]} -ne ${#HE_TEMPS_ARRAY[@]} || ${#HE_MODES_ARRAY[@]} -ne ${#HE_BFS_ARRAY[@]} || \
      ${#HE_MODES_ARRAY[@]} -ne ${#HE_BBCS_ARRAY[@]} || ${#HE_MODES_ARRAY[@]} -ne ${#HE_VFBWS_ARRAY[@]} || \
      ${#HE_MODES_ARRAY[@]} -ne ${#HE_BTOPPS_ARRAY[@]} || ${#HE_MODES_ARRAY[@]} -ne ${#HE_ALPHAS_ARRAY[@]} || \
      ${#HE_MODES_ARRAY[@]} -ne ${#HE_LIMITS_ARRAY[@]} || ${#HE_MODES_ARRAY[@]} -ne ${#HE_DTYPES_ARRAY[@]} ]]; then
    echo "Error: HumanEval configuration arrays have different lengths!"
    exit 1
fi

echo "#################### Starting HumanEval Evaluations ####################"
for i in "${!HE_MODES_ARRAY[@]}"; do
    mode_label="multi"
    if [[ "${HE_MODES_ARRAY[$i]}" == "false" ]]; then
        mode_label="single"
    fi
    output_path="results/evals_results_instruct_block_lopa_new/humaneval-256-ns0-temp${HE_TEMPS_ARRAY[$i]}-mode_${mode_label}-bf${HE_BFS_ARRAY[$i]}-bbc${HE_BBCS_ARRAY[$i]}-vfbw${HE_VFBWS_ARRAY[$i]}-btopp${HE_BTOPPS_ARRAY[$i]}-alpha${HE_ALPHAS_ARRAY[$i]}-limit${HE_LIMITS_ARRAY[$i]}"
    
    echo "--- Running HumanEval Config $((i+1))/${#HE_MODES_ARRAY[@]} ---"
    echo "  - Mode: ${mode_label} (${HE_MODES_ARRAY[$i]}), Temp: ${HE_TEMPS_ARRAY[$i]}, BF: ${HE_BFS_ARRAY[$i]}, BBC: ${HE_BBCS_ARRAY[$i]}"
    echo "  - VFBW: ${HE_VFBWS_ARRAY[$i]}, BTopP: ${HE_BTOPPS_ARRAY[$i]}, Alpha: ${HE_ALPHAS_ARRAY[$i]}"
    echo "  - Output: $output_path"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29510 --num_processes 8 ${SCRIPT_NAME} --model dream \
        --model_args pretrained=${model},max_new_tokens=256,temperature=${HE_TEMPS_ARRAY[$i]},add_bos_token=true,escape_until=true,dtype=${HE_DTYPES_ARRAY[$i]},use_uncertainty_logic=${HE_MODES_ARRAY[$i]},branching_factor=${HE_BFS_ARRAY[$i]},base_branch_competition=${HE_BBCS_ARRAY[$i]},verification_force_base_winner=${HE_VFBWS_ARRAY[$i]},branch_topp=${HE_BTOPPS_ARRAY[$i]},selection_conf_alpha=${HE_ALPHAS_ARRAY[$i]},save_dir=${output_path} \
        --tasks humaneval \
        --num_fewshot 0 \
        --batch_size 1 \
        --limit ${HE_LIMITS_ARRAY[$i]} \
        --output_path "$output_path" \
        --log_samples \
        --confirm_run_unsafe_code
done
echo "#################### HumanEval Evaluations Finished ####################"
echo "NOTICE: Remember to postprocess humaneval results."
echo ""


# ==============================================================================
# 2. 运行主要任务批量评测
# ==============================================================================
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra MODES_ARRAY <<< "$modes"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra TEMP_ARRAY <<< "$temperatures"
read -ra LIMITS_ARRAY <<< "$limits"
read -ra DTYPES_ARRAY <<< "$dtypes"
read -ra BFS_ARRAY <<< "$bfs"
read -ra BBCS_ARRAY <<< "$bbcs"
read -ra VFBWS_ARRAY <<< "$vfbws"
read -ra BTOPPS_ARRAY <<< "$btopps"
read -ra ALPHAS_ARRAY <<< "$alphas"

# 严格验证主要任务所有数组长度一致性
if [[ ${#TASKS_ARRAY[@]} -ne ${#NSHOTS_ARRAY[@]} || ${#TASKS_ARRAY[@]} -ne ${#MODES_ARRAY[@]} || \
      ${#TASKS_ARRAY[@]} -ne ${#LENGTH_ARRAY[@]} || ${#TASKS_ARRAY[@]} -ne ${#TEMP_ARRAY[@]} || \
      ${#TASKS_ARRAY[@]} -ne ${#LIMITS_ARRAY[@]} || ${#TASKS_ARRAY[@]} -ne ${#DTYPES_ARRAY[@]} || \
      ${#TASKS_ARRAY[@]} -ne ${#BFS_ARRAY[@]} || ${#TASKS_ARRAY[@]} -ne ${#BBCS_ARRAY[@]} || \
      ${#TASKS_ARRAY[@]} -ne ${#VFBWS_ARRAY[@]} || ${#TASKS_ARRAY[@]} -ne ${#BTOPPS_ARRAY[@]} || \
      ${#TASKS_ARRAY[@]} -ne ${#ALPHAS_ARRAY[@]} ]]; then
    echo "Error: Main task configuration arrays have different lengths! Please check your settings."
    echo "Lengths: tasks=${#TASKS_ARRAY[@]}, nshots=${#NSHOTS_ARRAY[@]}, modes=${#MODES_ARRAY[@]}, lengths=${#LENGTH_ARRAY[@]}, temps=${#TEMP_ARRAY[@]}, limits=${#LIMITS_ARRAY[@]}, dtypes=${#DTYPES_ARRAY[@]}, bfs=${#BFS_ARRAY[@]}, bbcs=${#BBCS_ARRAY[@]}, vfbws=${#VFBWS_ARRAY[@]}, btopps=${#BTOPPS_ARRAY[@]}, alphas=${#ALPHAS_ARRAY[@]}"
    exit 1
fi

echo "#################### Starting Main Task Evaluations ####################"
for i in "${!TASKS_ARRAY[@]}"; do
    mode_label="multi"
    if [[ "${MODES_ARRAY[$i]}" == "false" ]]; then
        mode_label="single"
    fi
    output_path="results/evals_results_instruct_block_lopa_new/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-len${LENGTH_ARRAY[$i]}-temp${TEMP_ARRAY[$i]}-mode_${mode_label}-bf${BFS_ARRAY[$i]}-bbc${BBCS_ARRAY[$i]}-vfbw${VFBWS_ARRAY[$i]}-btopp${BTOPPS_ARRAY[$i]}-alpha${ALPHAS_ARRAY[$i]}"
    
    echo "--- Running Main Task Config $((i+1))/${#TASKS_ARRAY[@]}: ${TASKS_ARRAY[$i]} ---"
    echo "  - Mode: ${mode_label} (${MODES_ARRAY[$i]}), Shots: ${NSHOTS_ARRAY[$i]}, Length: ${LENGTH_ARRAY[$i]}, Temp: ${TEMP_ARRAY[$i]}"
    echo "  - BF: ${BFS_ARRAY[$i]}, BBC: ${BBCS_ARRAY[$i]}, VFBW: ${VFBWS_ARRAY[$i]}, BTopP: ${BTOPPS_ARRAY[$i]}, Alpha: ${ALPHAS_ARRAY[$i]}"
    echo "  - Output: $output_path"

    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29511 --num_processes 7 ${SCRIPT_NAME} --model dream \
    #     --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},dtype=${DTYPES_ARRAY[$i]},use_uncertainty_logic=${MODES_ARRAY[$i]},branching_factor=${BFS_ARRAY[$i]},base_branch_competition=${BBCS_ARRAY[$i]},verification_force_base_winner=${VFBWS_ARRAY[$i]},branch_topp=${BTOPPS_ARRAY[$i]},selection_conf_alpha=${ALPHAS_ARRAY[$i]},save_dir=${output_path} \
    #     --tasks ${TASKS_ARRAY[$i]} \
    #     --num_fewshot ${NSHOTS_ARRAY[$i]} \
    #     --batch_size 1 \
    #     --output_path "$output_path" \
    #     --log_samples \
    #     --limit ${LIMITS_ARRAY[$i]} \
    #     --confirm_run_unsafe_code
done

echo "#################### All Evaluations Finished ####################"