#!/bin/bash
# filepath: /home/chenkai/data/Dream/eval/eval_dream_lora_para.sh

# ==========================================
# 任务配置 (General Tasks - 保持注释状态，随时可用)
# ==========================================
tasks="gsm8k mbpp minerva_math"
nshots="4 3 4"
lengths="256 256 256"  # 生成长度
temperatures="0 0 0"  # 温度参数
limits="10000 10000 10000"  # 生成限制
top_ps="0.9 0.9 0.9"  # top_p参数
dtypes="bfloat16 bfloat16 bfloat16"  # dtype参数
# --- 新增/修改的超参数 ---
block_lengths="32 32 32"  # 块大小 (原 block_sizes)
dParallels="true true true"  # dParallel参数
thresholds="0.45 0.5 0.45"  # threshold参数

# ==========================================
# HumanEval参数配置列表
# ==========================================
humaneval_nshots="0"  # HumanEval的few-shot数量
humaneval_lengths="256"  # HumanEval的生成长度
humaneval_temperatures="0"  # HumanEval的温度参数
humaneval_limits="10000"  # HumanEval的生成限制
humaneval_diffusion_steps="256"  # HumanEval的扩散步数
humaneval_top_ps="0.9"  # HumanEval的top_p参数
humaneval_dtypes="bfloat16"  # HumanEval的dtype参数
# --- 新增/修改的超参数 ---
humaneval_block_lengths="32"  # HumanEval的块大小 (原 humaneval_block_sizes)
humaneval_dParallels="true"  # HumanEval的dParallel参数
humaneval_thresholds="0.5"  # HumanEval的threshold参数

# ==========================================
# 模型路径配置
# ==========================================
# base_model=/data1/xck/models/Dream-v0-Instruct-7B
base_model=/home/chenkai/data/models/dParallel_Dream_7B_Instruct

lora_models=(
    "/home/chenkai/data/ckpt/wx_dream_base/Decoder-ddt_test-20k"
    # "/data1/xck/ckpt/my_data_block16_maskold_fp16_wx_smalllora/ddt_test/ddt_test/Decoder-ddt_test-19k"
    # "/data1/xck/ckpt/wx/dllm_block/data/dream_mask/Decoder-ddt_test-20k"
    # "/data1/xck/ckpt/jiachun/experiment/dllm_block/0606_block_attnmask_teacherlogits_blksize16_merged/denoiser-dllm_block-20k"
)

# ==========================================
# 数组转换 - General Tasks
# ==========================================
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra TEMP_ARRAY <<< "$temperatures"
read -ra LIMITS_ARRAY <<< "$limits"
read -ra TOP_PS_ARRAY <<< "$top_ps"
read -ra DTYPES_ARRAY <<< "$dtypes"
read -ra BLOCK_LENGTHS_ARRAY <<< "$block_lengths"
read -ra DPARALLELS_ARRAY <<< "$dParallels"
read -ra THRESHOLDS_ARRAY <<< "$thresholds"

# ==========================================
# 数组转换 - HumanEval
# ==========================================
read -ra HUMANEVAL_NSHOTS_ARRAY <<< "$humaneval_nshots"
read -ra HUMANEVAL_LENGTHS_ARRAY <<< "$humaneval_lengths"
read -ra HUMANEVAL_TEMP_ARRAY <<< "$humaneval_temperatures"
read -ra HUMANEVAL_LIMITS_ARRAY <<< "$humaneval_limits"
read -ra HUMANEVAL_DIFFUSION_STEPS_ARRAY <<< "$humaneval_diffusion_steps"
read -ra HUMANEVAL_TOP_PS_ARRAY <<< "$humaneval_top_ps"
read -ra HUMANEVAL_DTYPES_ARRAY <<< "$humaneval_dtypes"
read -ra HUMANEVAL_BLOCK_LENGTHS_ARRAY <<< "$humaneval_block_lengths"
read -ra HUMANEVAL_DPARALLELS_ARRAY <<< "$humaneval_dParallels"
read -ra HUMANEVAL_THRESHOLDS_ARRAY <<< "$humaneval_thresholds"

# ==========================================
# 长度校验
# ==========================================
array_length=${#TASKS_ARRAY[@]}
if [[ $array_length -gt 0 ]]; then
    if [[ ${#NSHOTS_ARRAY[@]} -ne $array_length ]] || \
       [[ ${#LENGTH_ARRAY[@]} -ne $array_length ]] || \
       [[ ${#TEMP_ARRAY[@]} -ne $array_length ]] || \
       [[ ${#LIMITS_ARRAY[@]} -ne $array_length ]] || \
       [[ ${#TOP_PS_ARRAY[@]} -ne $array_length ]] || \
       [[ ${#DTYPES_ARRAY[@]} -ne $array_length ]] || \
       [[ ${#BLOCK_LENGTHS_ARRAY[@]} -ne $array_length ]] || \
       [[ ${#DPARALLELS_ARRAY[@]} -ne $array_length ]] || \
       [[ ${#THRESHOLDS_ARRAY[@]} -ne $array_length ]]; then
        echo "错误：所有配置数组的长度必须相同！"
        exit 1
    fi
fi

humaneval_array_length=${#HUMANEVAL_NSHOTS_ARRAY[@]}
if [[ ${#HUMANEVAL_LENGTHS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_TEMP_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_LIMITS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_DIFFUSION_STEPS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_TOP_PS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_DTYPES_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_BLOCK_LENGTHS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_DPARALLELS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_THRESHOLDS_ARRAY[@]} -ne $humaneval_array_length ]]; then
    echo "错误：所有HumanEval配置数组的长度必须相同！"
    exit 1
fi

export HF_ALLOW_CODE_EVAL=1
export HF_ENDPOINT=https://hf-mirror.com

# ==========================================
# 评测主循环
# ==========================================
for lora_model in "${lora_models[@]}"; do
    lora_model_name="$lora_model"
    echo "===================================================================="
    echo "Evaluating LoRA model: $lora_model_name"
    echo "===================================================================="
    
    # HumanEval评估（参数列表遍历）
    for i in "${!HUMANEVAL_NSHOTS_ARRAY[@]}"; do
        # 简化了输出路径的命名，去掉了无用的参数标识
        output_path="evals_dream_dp_my_new${lora_model_name}/humaneval-ns${HUMANEVAL_NSHOTS_ARRAY[$i]}-len${HUMANEVAL_LENGTHS_ARRAY[$i]}-temp${HUMANEVAL_TEMP_ARRAY[$i]}-limit${HUMANEVAL_LIMITS_ARRAY[$i]}-diffsteps${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]}-blocklen${HUMANEVAL_BLOCK_LENGTHS_ARRAY[$i]}-dpara${HUMANEVAL_DPARALLELS_ARRAY[$i]}-thresh${HUMANEVAL_THRESHOLDS_ARRAY[$i]}-topp${HUMANEVAL_TOP_PS_ARRAY[$i]}-dtype${HUMANEVAL_DTYPES_ARRAY[$i]}-max_new_tokens${HUMANEVAL_LENGTHS_ARRAY[$i]}"
        echo "Running HumanEval evaluation $((i+1))/${humaneval_array_length} for $lora_model_name..."
        echo "HumanEval Config Output: $output_path"
        
        # 基础 args 配置 (移除旧参数，添加新参数)
        base_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${HUMANEVAL_LENGTHS_ARRAY[$i]},diffusion_steps=${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]},temperature=${HUMANEVAL_TEMP_ARRAY[$i]},add_bos_token=true,escape_until=true,block_length=${HUMANEVAL_BLOCK_LENGTHS_ARRAY[$i]},dParallel=${HUMANEVAL_DPARALLELS_ARRAY[$i]},threshold=${HUMANEVAL_THRESHOLDS_ARRAY[$i]},dtype=${HUMANEVAL_DTYPES_ARRAY[$i]},save_dir=${output_path}"

        # 构建HumanEval的model_args，根据top_p是否为none来决定是否包含top_p参数
        if [[ "${HUMANEVAL_TOP_PS_ARRAY[$i]}" == "none" ]]; then
            humaneval_model_args="${base_args}"
        else
            humaneval_model_args="${base_args},top_p=${HUMANEVAL_TOP_PS_ARRAY[$i]}"
        fi
        
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29520 --num_processes 8 eval_dream_dp_my.py --model dream_lora \
            --model_args $humaneval_model_args \
            --tasks humaneval \
            --num_fewshot ${HUMANEVAL_NSHOTS_ARRAY[$i]} \
            --batch_size 1 \
            --output_path $output_path \
            --log_samples \
            --confirm_run_unsafe_code
    done

    # 其他任务的评估 (注：由于外部 tasks 为注释状态，此段循环默认不执行，如需启用取消文件开头的注释即可)
    for i in "${!TASKS_ARRAY[@]}"; do
        # 简化了输出路径的命名，去掉了无用的参数标识
        output_path="evals_dream_dp_my_new${lora_model_name}/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-len${LENGTH_ARRAY[$i]}-temp${TEMP_ARRAY[$i]}-limit${LIMITS_ARRAY[$i]}-diffsteps${LENGTH_ARRAY[$i]}-blocklen${BLOCK_LENGTHS_ARRAY[$i]}-dpara${DPARALLELS_ARRAY[$i]}-thresh${THRESHOLDS_ARRAY[$i]}-topp${TOP_PS_ARRAY[$i]}-dtype${DTYPES_ARRAY[$i]}"
        echo "Running Task evaluation for ${TASKS_ARRAY[$i]}..."
        echo "Task Config Output: $output_path"
        
        base_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},block_length=${BLOCK_LENGTHS_ARRAY[$i]},dParallel=${DPARALLELS_ARRAY[$i]},threshold=${THRESHOLDS_ARRAY[$i]},dtype=${DTYPES_ARRAY[$i]},save_dir=${output_path}"

        # 构建model_args，根据top_p是否为none来决定是否包含top_p参数
        if [[ "${TOP_PS_ARRAY[$i]}" == "none" ]]; then
            model_args="${base_args}"
        else
            model_args="${base_args},top_p=${TOP_PS_ARRAY[$i]}"
        fi
        
        # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29520 --num_processes 8 eval_dream_dp_my.py --model dream_lora \
        #     --model_args $model_args \
        #     --tasks ${TASKS_ARRAY[$i]} \
        #     --limit ${LIMITS_ARRAY[$i]} \
        #     --num_fewshot ${NSHOTS_ARRAY[$i]} \
        #     --batch_size 1 \
        #     --output_path $output_path \
        #     --log_samples \
        #     --confirm_run_unsafe_code
    done
done

echo "All evaluations completed!"