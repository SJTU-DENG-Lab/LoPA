#!/bin/bash
# filepath: /home/chenkai/data/Dream/eval/eval_dream_lora_para.sh

tasks="gsm8k mbpp minerva_math"
nshots="4 3 4"
lengths="512 512 512"
temperatures="0 0 0"
limits="10000 200 10000"
block_sizes="32 32 32"
skip_thresholds="0.9 0.9 0.9"
top_ps="none none none"
dtypes="bfloat16 bfloat16 bfloat16"
sampling_strategies="default default default"

# HumanEval参数配置列表
humaneval_nshots="0"
humaneval_lengths="512"
humaneval_temperatures="0"
humaneval_limits="10000"
humaneval_diffusion_steps="512"
humaneval_block_sizes="32"
humaneval_skip_thresholds="0.9"
humaneval_top_ps="none"
humaneval_dtypes="bfloat16"
humaneval_sampling_strategies="default"

# 基础模型路径
base_model=/home/chenkai/data/models/llada-8b-instruct

# 保留原脚本结构，当前实现不再真正使用 LoRA
lora_models=(
    "unused"
)

# Create arrays from space-separated strings
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra TEMP_ARRAY <<< "$temperatures"
read -ra LIMITS_ARRAY <<< "$limits"
read -ra BLOCK_SIZES_ARRAY <<< "$block_sizes"
read -ra SKIP_THRESHOLDS_ARRAY <<< "$skip_thresholds"
read -ra TOP_PS_ARRAY <<< "$top_ps"
read -ra DTYPES_ARRAY <<< "$dtypes"
read -ra SAMPLING_STRATEGIES_ARRAY <<< "$sampling_strategies"

# Create arrays for HumanEval configurations
read -ra HUMANEVAL_NSHOTS_ARRAY <<< "$humaneval_nshots"
read -ra HUMANEVAL_LENGTHS_ARRAY <<< "$humaneval_lengths"
read -ra HUMANEVAL_TEMP_ARRAY <<< "$humaneval_temperatures"
read -ra HUMANEVAL_LIMITS_ARRAY <<< "$humaneval_limits"
read -ra HUMANEVAL_DIFFUSION_STEPS_ARRAY <<< "$humaneval_diffusion_steps"
read -ra HUMANEVAL_BLOCK_SIZES_ARRAY <<< "$humaneval_block_sizes"
read -ra HUMANEVAL_SKIP_THRESHOLDS_ARRAY <<< "$humaneval_skip_thresholds"
read -ra HUMANEVAL_TOP_PS_ARRAY <<< "$humaneval_top_ps"
read -ra HUMANEVAL_DTYPES_ARRAY <<< "$humaneval_dtypes"
read -ra HUMANEVAL_SAMPLING_STRATEGIES_ARRAY <<< "$humaneval_sampling_strategies"

# 验证所有数组长度是否一致
array_length=${#TASKS_ARRAY[@]}
if [[ ${#NSHOTS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#LENGTH_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#TEMP_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#LIMITS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#BLOCK_SIZES_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#SKIP_THRESHOLDS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#TOP_PS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#SAMPLING_STRATEGIES_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#DTYPES_ARRAY[@]} -ne $array_length ]]; then
    echo "错误：所有配置数组的长度必须相同！"
    echo "Tasks: ${#TASKS_ARRAY[@]}, Nshots: ${#NSHOTS_ARRAY[@]}, Lengths: ${#LENGTH_ARRAY[@]}, Temperatures: ${#TEMP_ARRAY[@]}, Limits: ${#LIMITS_ARRAY[@]}, Block sizes: ${#BLOCK_SIZES_ARRAY[@]}, Skip thresholds: ${#SKIP_THRESHOLDS_ARRAY[@]}, Top_ps: ${#TOP_PS_ARRAY[@]}, Sampling strategies: ${#SAMPLING_STRATEGIES_ARRAY[@]}, Dtypes: ${#DTYPES_ARRAY[@]}"
    exit 1
fi

# 验证HumanEval数组长度是否一致
humaneval_array_length=${#HUMANEVAL_NSHOTS_ARRAY[@]}
if [[ ${#HUMANEVAL_LENGTHS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_TEMP_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_LIMITS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_DIFFUSION_STEPS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_BLOCK_SIZES_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_SKIP_THRESHOLDS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_TOP_PS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_DTYPES_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[@]} -ne $humaneval_array_length ]]; then
    echo "错误：所有HumanEval配置数组的长度必须相同！"
    echo "HumanEval Nshots: ${#HUMANEVAL_NSHOTS_ARRAY[@]}, Lengths: ${#HUMANEVAL_LENGTHS_ARRAY[@]}, Temperatures: ${#HUMANEVAL_TEMP_ARRAY[@]}, Limits: ${#HUMANEVAL_LIMITS_ARRAY[@]}, Diffusion steps: ${#HUMANEVAL_DIFFUSION_STEPS_ARRAY[@]}, Block sizes: ${#HUMANEVAL_BLOCK_SIZES_ARRAY[@]}, Skip thresholds: ${#HUMANEVAL_SKIP_THRESHOLDS_ARRAY[@]}, Top_ps: ${#HUMANEVAL_TOP_PS_ARRAY[@]}, Dtypes: ${#HUMANEVAL_DTYPES_ARRAY[@]}, Sampling strategies: ${#HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[@]}"
    exit 1
fi

export HF_ALLOW_CODE_EVAL=1
# 对每个LoRA模型进行评测
for lora_model in "${lora_models[@]}"; do
    model_name="$base_model"
    echo "===================================================================="
    echo "Evaluating model: $model_name"
    echo "===================================================================="

    # HumanEval评估（参数列表遍历）
    for i in "${!HUMANEVAL_NSHOTS_ARRAY[@]}"; do
        output_path="eval_llada_chat_try7_full_fix/${model_name##*/}/humaneval-ns${HUMANEVAL_NSHOTS_ARRAY[$i]}-len${HUMANEVAL_LENGTHS_ARRAY[$i]}-temp${HUMANEVAL_TEMP_ARRAY[$i]}-limit${HUMANEVAL_LIMITS_ARRAY[$i]}-diffsteps${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]}-block${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]}-skip${HUMANEVAL_SKIP_THRESHOLDS_ARRAY[$i]}-topp${HUMANEVAL_TOP_PS_ARRAY[$i]}-dtype${HUMANEVAL_DTYPES_ARRAY[$i]}-sampling${HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[$i]}"
        echo "Running HumanEval evaluation $((i+1))/${humaneval_array_length} for $model_name..."
        echo "HumanEval Config: Shots: ${HUMANEVAL_NSHOTS_ARRAY[$i]}, Length: ${HUMANEVAL_LENGTHS_ARRAY[$i]}, Temperature: ${HUMANEVAL_TEMP_ARRAY[$i]}, Limit: ${HUMANEVAL_LIMITS_ARRAY[$i]}, Diffusion Steps: ${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]}, Block Size: ${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]}, Skip Threshold: ${HUMANEVAL_SKIP_THRESHOLDS_ARRAY[$i]}, Top_p: ${HUMANEVAL_TOP_PS_ARRAY[$i]}, Sampling Strategy: ${HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[$i]}, Dtype: ${HUMANEVAL_DTYPES_ARRAY[$i]}; Output: $output_path"

        # 构建HumanEval的model_args，根据top_p是否为none来决定是否包含top_p参数
        if [[ "${HUMANEVAL_TOP_PS_ARRAY[$i]}" == "none" ]]; then
            humaneval_model_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${HUMANEVAL_LENGTHS_ARRAY[$i]},diffusion_steps=${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]},temperature=${HUMANEVAL_TEMP_ARRAY[$i]},add_bos_token=true,escape_until=true,block_size=${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]},skip_threshold=${HUMANEVAL_SKIP_THRESHOLDS_ARRAY[$i]},dtype=${HUMANEVAL_DTYPES_ARRAY[$i]},sampling_strategy=${HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[$i]},save_dir=${output_path}"
        else
            humaneval_model_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${HUMANEVAL_LENGTHS_ARRAY[$i]},diffusion_steps=${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]},temperature=${HUMANEVAL_TEMP_ARRAY[$i]},top_p=${HUMANEVAL_TOP_PS_ARRAY[$i]},add_bos_token=true,escape_until=true,block_size=${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]},skip_threshold=${HUMANEVAL_SKIP_THRESHOLDS_ARRAY[$i]},dtype=${HUMANEVAL_DTYPES_ARRAY[$i]},sampling_strategy=${HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[$i]},save_dir=${output_path}"
        fi

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29520 --num_processes 8 eval_llada_xck_blocknew.py --model dream_lora \
            --model_args $humaneval_model_args \
            --tasks humaneval \
            --limit ${HUMANEVAL_LIMITS_ARRAY[$i]} \
            --num_fewshot ${HUMANEVAL_NSHOTS_ARRAY[$i]} \
            --batch_size 1 \
            --output_path $output_path \
            --log_samples \
            --confirm_run_unsafe_code
    done

    # 其他任务的评估
    for i in "${!TASKS_ARRAY[@]}"; do
        output_path="eval_llada_chat_verify/${model_name##*/}/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-len${LENGTH_ARRAY[$i]}-temp${TEMP_ARRAY[$i]}-limit${LIMITS_ARRAY[$i]}-diffsteps${LENGTH_ARRAY[$i]}-block${BLOCK_SIZES_ARRAY[$i]}-skip${SKIP_THRESHOLDS_ARRAY[$i]}-topp${TOP_PS_ARRAY[$i]}-dtype${DTYPES_ARRAY[$i]}-sampling${SAMPLING_STRATEGIES_ARRAY[$i]}"
        echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}, Length: ${LENGTH_ARRAY[$i]}, Temperature: ${TEMP_ARRAY[$i]}, Limit: ${LIMITS_ARRAY[$i]}, Block Size: ${BLOCK_SIZES_ARRAY[$i]}, Skip Threshold: ${SKIP_THRESHOLDS_ARRAY[$i]}, Top_p: ${TOP_PS_ARRAY[$i]}, Sampling Strategy: ${SAMPLING_STRATEGIES_ARRAY[$i]}, Dtype: ${DTYPES_ARRAY[$i]}; Output: $output_path"

        # 构建model_args，根据top_p是否为none来决定是否包含top_p参数
        if [[ "${TOP_PS_ARRAY[$i]}" == "none" ]]; then
            model_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},block_size=${BLOCK_SIZES_ARRAY[$i]},skip_threshold=${SKIP_THRESHOLDS_ARRAY[$i]},dtype=${DTYPES_ARRAY[$i]},sampling_strategy=${SAMPLING_STRATEGIES_ARRAY[$i]},save_dir=${output_path}"
        else
            model_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},top_p=${TOP_PS_ARRAY[$i]},block_size=${BLOCK_SIZES_ARRAY[$i]},skip_threshold=${SKIP_THRESHOLDS_ARRAY[$i]},dtype=${DTYPES_ARRAY[$i]},sampling_strategy=${SAMPLING_STRATEGIES_ARRAY[$i]},save_dir=${output_path}"
        fi

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29520 --num_processes 8 eval_llada_xck_blocknew.py --model dream_lora \
            --model_args $model_args \
            --tasks ${TASKS_ARRAY[$i]} \
            --limit ${LIMITS_ARRAY[$i]} \
            --num_fewshot ${NSHOTS_ARRAY[$i]} \
            --batch_size 1 \
            --output_path $output_path \
            --log_samples \
            --confirm_run_unsafe_code \
            --apply_chat_template \
            --fewshot_as_multiturn
    done
done

echo "All evaluations completed!"
