#!/bin/bash
# filepath: /home/chenkai/data/Dream/eval/eval_dream_lora_para.sh

tasks="gsm8k gsm8k gsm8k mbpp mbpp mbpp"
nshots="4 4 4 3 3 3"
lengths="512 512 512 512 512 512"
temperatures="0 0 0 0 0 0"
limits="10000 10000 10000 10000 10000 10000"
block_sizes="64 64 64 32 32 32"
block_add_thresholds="0.7 0.7 0.7 0.9 0.9 0.9"
decoded_token_thresholds="0.95 0.95 0.95 0.95 0.95 0.95"
skip_thresholds="0.9 0.9 0.9 0.9 0.9 0.9"
top_ps="none none none none none none"
dtypes="bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16"
sampling_strategies="neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy"
max_branches_kepts="1 1 1 1 1 1"
branching_factors="2 3 4 2 3 4"
branch_topps="1 1 1 1 1 1"
selection_conf_alphas="0 0 0 0 0 0"
branch_verification_modes="true true true true true true"
base_branch_competitions="true true true true true true"
verification_force_base_winners="false false false false false false"

tasks="gsm8k gsm8k gsm8k"
nshots="4 4 4"
lengths="512 512 512"
temperatures="0 0 0"
limits="10000 10000 10000"
block_sizes="64 64 64"
block_add_thresholds="0.7 0.7 0.7"
decoded_token_thresholds="0.95 0.95 0.95"
skip_thresholds="0.9 0.9 0.9"
top_ps="none none none"
dtypes="bfloat16 bfloat16 bfloat16"
sampling_strategies="neg_entropy neg_entropy neg_entropy"
max_branches_kepts="1 1 1"
branching_factors="2 3 4"
branch_topps="0.5 0.5 0.5"
selection_conf_alphas="0.5 0.5 0.5"
branch_verification_modes="true true true"
base_branch_competitions="true true true"
verification_force_base_winners="false false false"

tasks="gsm8k"
nshots="4"
lengths="512"
temperatures="0"
limits="10000"
block_sizes="64"
block_add_thresholds="0.7"
decoded_token_thresholds="0.95"
skip_thresholds="0.9"
top_ps="none"
dtypes="bfloat16"
sampling_strategies="neg_entropy"
max_branches_kepts="1"
branching_factors="2"
branch_topps="1"
selection_conf_alphas="0"
branch_verification_modes="true"
base_branch_competitions="true"
verification_force_base_winners="true"

tasks="gsm8k gsm8k gsm8k gsm8k gsm8k gsm8k gsm8k gsm8k gsm8k mbpp mbpp mbpp mbpp mbpp mbpp mbpp mbpp mbpp minerva_math minerva_math minerva_math minerva_math minerva_math minerva_math minerva_math minerva_math minerva_math"
nshots="4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4"
lengths="512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512"
temperatures="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
limits="10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000"
block_sizes="64 64 64 64 64 64 64 64 64 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32"
block_add_thresholds="0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1"
decoded_token_thresholds="0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95"
skip_thresholds="0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9"
top_ps="none none none none none none none none none none none none none none none none none none none none none none none none none none none"
dtypes="bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16"
sampling_strategies="neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy"
max_branches_kepts="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
branching_factors="2 2 3 4 5 6 7 8 9 2 2 3 4 5 6 7 8 9 2 2 3 4 5 6 7 8 9"
branch_topps="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
selection_conf_alphas="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
branch_verification_modes="true true true true true true true true true true true true true true true true true true true true true true true true true true true"
base_branch_competitions="true true true true true true true true true true true true true true true true true true true true true true true true true true true"
verification_force_base_winners="true false false false false false false false false true false false false false false false false false true false false false false false false false false"

# HumanEval参数配置列表
humaneval_nshots="0 0 0 0 0 0"  # HumanEval的few-shot数量
humaneval_lengths="512 512 512 512 512 512"  # HumanEval的生成长度
humaneval_temperatures="0 0 0 0 0 0"  # HumanEval的温度参数
humaneval_limits="10000 10000 10000 10000 10000 10000"  # HumanEval的生成限制
humaneval_diffusion_steps="512 512 512 512 512 512"  # HumanEval的扩散步数
humaneval_block_sizes="32 32 32 32 32 32"  # HumanEval的块大小
humaneval_block_add_thresholds="0.1 0.1 0.1 0.1 0.1 0.1"  # HumanEval的块添加阈值
humaneval_decoded_token_thresholds="0.95 0.95 0.95 0.95 0.95 0.95"  # HumanEval的解码token阈值
humaneval_skip_thresholds="0.9 0.9 0.9 0.9 0.9 0.9"  # HumanEval的跳过阈值
humaneval_top_ps="none none none none none none"  # HumanEval的top_p参数
humaneval_dtypes="bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16"  # HumanEval的dtype参数
humaneval_sampling_strategies="neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy"  # HumanEval的采样策略参数
humaneval_max_branches_kepts="1 1 1 1 1 1"  # HumanEval的保留最大分支数
humaneval_branching_factors="2 3 4 2 3 4"  # HumanEval的分支因子
humaneval_branch_topps="1 1 1 1 1 1"  # HumanEval的分支置信度阈值参数
humaneval_selection_conf_alphas="0 0 0 0 0 0"  # HumanEval的选择置信度权重参数
humaneval_branch_verification_modes="false false false true true true"  # HumanEval的分支验证模式
humaneval_base_branch_competitions="true true true true true true"  # HumanEval的基础分支竞争
humaneval_verification_force_base_winners="false false false false false false"  # HumanEval的验证强制基础分支胜出

humaneval_nshots="0 0 0 0 0 0 0 0 0"
humaneval_lengths="512 512 512 512 512 512 512 512 512"
humaneval_temperatures="0 0 0 0 0 0 0 0 0"
humaneval_limits="10000 10000 10000 10000 10000 10000 10000 10000 10000"
humaneval_diffusion_steps="512 512 512 512 512 512 512 512 512"
humaneval_block_sizes="32 32 32 32 32 32 32 32 32"
humaneval_block_add_thresholds="0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1"
humaneval_decoded_token_thresholds="0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95"
humaneval_skip_thresholds="0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9"
humaneval_top_ps="none none none none none none none none none"
humaneval_dtypes="bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16"
humaneval_sampling_strategies="neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy neg_entropy"
humaneval_max_branches_kepts="1 1 1 1 1 1 1 1 1"
humaneval_branching_factors="2 2 3 4 5 6 7 8 9"
humaneval_branch_topps="1 1 1 1 1 1 1 1 1"
humaneval_selection_conf_alphas="0 0 0 0 0 0 0 0 0"
humaneval_branch_verification_modes="true true true true true true true true true"
humaneval_base_branch_competitions="true true true true true true true true true"
humaneval_verification_force_base_winners="true false false false false false false false false"



base_model=/mnt/rl/xinyi/models/LLaDA-8B-Instruct

# 多个LoRA模型路径，用空格分隔
lora_models=(
    "/mnt/rl/xinyi/models/D2F_LLaDA_Instruct_8B_Lora"
)

# Create arrays from space-separated strings
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra TEMP_ARRAY <<< "$temperatures"
read -ra LIMITS_ARRAY <<< "$limits"
read -ra BLOCK_SIZES_ARRAY <<< "$block_sizes"
read -ra BLOCK_ADD_THRESHOLDS_ARRAY <<< "$block_add_thresholds"
read -ra DECODED_TOKEN_THRESHOLDS_ARRAY <<< "$decoded_token_thresholds"
read -ra SKIP_THRESHOLDS_ARRAY <<< "$skip_thresholds"
read -ra TOP_PS_ARRAY <<< "$top_ps"
read -ra DTYPES_ARRAY <<< "$dtypes"
read -ra SAMPLING_STRATEGIES_ARRAY <<< "$sampling_strategies"

# Create arrays for multi-branch parameters
read -ra MAX_BRANCHES_KEPTS_ARRAY <<< "$max_branches_kepts"
read -ra BRANCHING_FACTORS_ARRAY <<< "$branching_factors"
read -ra BRANCH_TOPPS_ARRAY <<< "$branch_topps"
read -ra SELECTION_CONF_ALPHAS_ARRAY <<< "$selection_conf_alphas"
read -ra BRANCH_VERIFICATION_MODES_ARRAY <<< "$branch_verification_modes"
read -ra BASE_BRANCH_COMPETITIONS_ARRAY <<< "$base_branch_competitions"
read -ra VERIFICATION_FORCE_BASE_WINNERS_ARRAY <<< "$verification_force_base_winners"

# Create arrays for HumanEval configurations
read -ra HUMANEVAL_NSHOTS_ARRAY <<< "$humaneval_nshots"
read -ra HUMANEVAL_LENGTHS_ARRAY <<< "$humaneval_lengths"
read -ra HUMANEVAL_TEMP_ARRAY <<< "$humaneval_temperatures"
read -ra HUMANEVAL_LIMITS_ARRAY <<< "$humaneval_limits"
read -ra HUMANEVAL_DIFFUSION_STEPS_ARRAY <<< "$humaneval_diffusion_steps"
read -ra HUMANEVAL_BLOCK_SIZES_ARRAY <<< "$humaneval_block_sizes"
read -ra HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY <<< "$humaneval_block_add_thresholds"
read -ra HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY <<< "$humaneval_decoded_token_thresholds"
read -ra HUMANEVAL_SKIP_THRESHOLDS_ARRAY <<< "$humaneval_skip_thresholds"
read -ra HUMANEVAL_TOP_PS_ARRAY <<< "$humaneval_top_ps"
read -ra HUMANEVAL_DTYPES_ARRAY <<< "$humaneval_dtypes"
read -ra HUMANEVAL_SAMPLING_STRATEGIES_ARRAY <<< "$humaneval_sampling_strategies"

# Create arrays for HumanEval multi-branch parameters
read -ra HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY <<< "$humaneval_max_branches_kepts"
read -ra HUMANEVAL_BRANCHING_FACTORS_ARRAY <<< "$humaneval_branching_factors"
read -ra HUMANEVAL_BRANCH_TOPPS_ARRAY <<< "$humaneval_branch_topps"
read -ra HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY <<< "$humaneval_selection_conf_alphas"
read -ra HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY <<< "$humaneval_branch_verification_modes"
read -ra HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY <<< "$humaneval_base_branch_competitions"
read -ra HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY <<< "$humaneval_verification_force_base_winners"

# 验证所有数组长度是否一致
array_length=${#TASKS_ARRAY[@]}
if [[ ${#NSHOTS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#LENGTH_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#TEMP_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#LIMITS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#BLOCK_SIZES_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#BLOCK_ADD_THRESHOLDS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#DECODED_TOKEN_THRESHOLDS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#SKIP_THRESHOLDS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#TOP_PS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#SAMPLING_STRATEGIES_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#DTYPES_ARRAY[@]} -ne $array_length ]]; then
    echo "错误：所有配置数组的长度必须相同！"
    echo "Tasks: ${#TASKS_ARRAY[@]}, Nshots: ${#NSHOTS_ARRAY[@]}, Lengths: ${#LENGTH_ARRAY[@]}, Temperatures: ${#TEMP_ARRAY[@]}, Limits: ${#LIMITS_ARRAY[@]}, Block sizes: ${#BLOCK_SIZES_ARRAY[@]}, Block thresholds: ${#BLOCK_ADD_THRESHOLDS_ARRAY[@]}, Decoded token thresholds: ${#DECODED_TOKEN_THRESHOLDS_ARRAY[@]}, Skip thresholds: ${#SKIP_THRESHOLDS_ARRAY[@]}, Top_ps: ${#TOP_PS_ARRAY[@]}, Sampling strategies: ${#SAMPLING_STRATEGIES_ARRAY[@]}, Dtypes: ${#DTYPES_ARRAY[@]}"
    exit 1
fi

# 验证多分支参数数组长度是否一致
multibranch_array_length=${#MAX_BRANCHES_KEPTS_ARRAY[@]}
if [[ ${#BRANCHING_FACTORS_ARRAY[@]} -ne $multibranch_array_length ]] || \
   [[ ${#BRANCH_TOPPS_ARRAY[@]} -ne $multibranch_array_length ]] || \
   [[ ${#SELECTION_CONF_ALPHAS_ARRAY[@]} -ne $multibranch_array_length ]] || \
   [[ ${#BRANCH_VERIFICATION_MODES_ARRAY[@]} -ne $multibranch_array_length ]] || \
   [[ ${#BASE_BRANCH_COMPETITIONS_ARRAY[@]} -ne $multibranch_array_length ]] || \
   [[ ${#VERIFICATION_FORCE_BASE_WINNERS_ARRAY[@]} -ne $multibranch_array_length ]]; then
    echo "错误：所有多分支参数配置数组的长度必须相同！"
    echo "Max branches kept: ${#MAX_BRANCHES_KEPTS_ARRAY[@]}, Branching factors: ${#BRANCHING_FACTORS_ARRAY[@]}, Branch topps: ${#BRANCH_TOPPS_ARRAY[@]}, Selection conf alphas: ${#SELECTION_CONF_ALPHAS_ARRAY[@]}, Branch verification modes: ${#BRANCH_VERIFICATION_MODES_ARRAY[@]}, Base branch competitions: ${#BASE_BRANCH_COMPETITIONS_ARRAY[@]}, Verification force base winners: ${#VERIFICATION_FORCE_BASE_WINNERS_ARRAY[@]}"
    exit 1
fi

# 如果多分支参数只有一个配置，将其扩展到与主任务数组相同长度
if [[ $multibranch_array_length -eq 1 && $array_length -gt 1 ]]; then
    max_branches_kept_single="${MAX_BRANCHES_KEPTS_ARRAY[0]}"
    branching_factor_single="${BRANCHING_FACTORS_ARRAY[0]}"
    branch_topp_single="${BRANCH_TOPPS_ARRAY[0]}"
    selection_conf_alpha_single="${SELECTION_CONF_ALPHAS_ARRAY[0]}"
    branch_verification_mode_single="${BRANCH_VERIFICATION_MODES_ARRAY[0]}"
    base_branch_competition_single="${BASE_BRANCH_COMPETITIONS_ARRAY[0]}"
    verification_force_base_winner_single="${VERIFICATION_FORCE_BASE_WINNERS_ARRAY[0]}"
    
    MAX_BRANCHES_KEPTS_ARRAY=()
    BRANCHING_FACTORS_ARRAY=()
    BRANCH_TOPPS_ARRAY=()
    SELECTION_CONF_ALPHAS_ARRAY=()
    BRANCH_VERIFICATION_MODES_ARRAY=()
    BASE_BRANCH_COMPETITIONS_ARRAY=()
    VERIFICATION_FORCE_BASE_WINNERS_ARRAY=()
    
    for ((j=0; j<$array_length; j++)); do
        MAX_BRANCHES_KEPTS_ARRAY+=("$max_branches_kept_single")
        BRANCHING_FACTORS_ARRAY+=("$branching_factor_single")
        BRANCH_TOPPS_ARRAY+=("$branch_topp_single")
        SELECTION_CONF_ALPHAS_ARRAY+=("$selection_conf_alpha_single")
        BRANCH_VERIFICATION_MODES_ARRAY+=("$branch_verification_mode_single")
        BASE_BRANCH_COMPETITIONS_ARRAY+=("$base_branch_competition_single")
        VERIFICATION_FORCE_BASE_WINNERS_ARRAY+=("$verification_force_base_winner_single")
    done
elif [[ $multibranch_array_length -ne $array_length ]]; then
    echo "错误：多分支参数数组长度必须为1或与主任务数组长度相同！"
    echo "Main tasks length: $array_length, Multi-branch params length: $multibranch_array_length"
    exit 1
fi

# 验证HumanEval数组长度是否一致
humaneval_array_length=${#HUMANEVAL_NSHOTS_ARRAY[@]}
if [[ ${#HUMANEVAL_LENGTHS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_TEMP_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_LIMITS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_DIFFUSION_STEPS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_BLOCK_SIZES_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_SKIP_THRESHOLDS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_TOP_PS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_DTYPES_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[@]} -ne $humaneval_array_length ]]; then
    echo "错误：所有HumanEval配置数组的长度必须相同！"
    echo "HumanEval Nshots: ${#HUMANEVAL_NSHOTS_ARRAY[@]}, Lengths: ${#HUMANEVAL_LENGTHS_ARRAY[@]}, Temperatures: ${#HUMANEVAL_TEMP_ARRAY[@]}, Limits: ${#HUMANEVAL_LIMITS_ARRAY[@]}, Diffusion steps: ${#HUMANEVAL_DIFFUSION_STEPS_ARRAY[@]}, Block sizes: ${#HUMANEVAL_BLOCK_SIZES_ARRAY[@]}, Block thresholds: ${#HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY[@]}, Decoded token thresholds: ${#HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY[@]}, Skip thresholds: ${#HUMANEVAL_SKIP_THRESHOLDS_ARRAY[@]}, Top_ps: ${#HUMANEVAL_TOP_PS_ARRAY[@]}, Dtypes: ${#HUMANEVAL_DTYPES_ARRAY[@]}, Sampling strategies: ${#HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[@]}"
    exit 1
fi

# 验证HumanEval多分支参数数组长度是否一致
humaneval_multibranch_array_length=${#HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY[@]}
if [[ ${#HUMANEVAL_BRANCHING_FACTORS_ARRAY[@]} -ne $humaneval_multibranch_array_length ]] || \
   [[ ${#HUMANEVAL_BRANCH_TOPPS_ARRAY[@]} -ne $humaneval_multibranch_array_length ]] || \
   [[ ${#HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY[@]} -ne $humaneval_multibranch_array_length ]] || \
   [[ ${#HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY[@]} -ne $humaneval_multibranch_array_length ]] || \
   [[ ${#HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY[@]} -ne $humaneval_multibranch_array_length ]] || \
   [[ ${#HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY[@]} -ne $humaneval_multibranch_array_length ]]; then
    echo "错误：所有HumanEval多分支参数配置数组的长度必须相同！"
    echo "HumanEval Max branches kept: ${#HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY[@]}, Branching factors: ${#HUMANEVAL_BRANCHING_FACTORS_ARRAY[@]}, Branch topps: ${#HUMANEVAL_BRANCH_TOPPS_ARRAY[@]}, Selection conf alphas: ${#HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY[@]}, Branch verification modes: ${#HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY[@]}, Base branch competitions: ${#HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY[@]}, Verification force base winners: ${#HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY[@]}"
    exit 1
fi

# 如果HumanEval多分支参数只有一个配置，将其扩展到与HumanEval主数组相同长度
if [[ $humaneval_multibranch_array_length -eq 1 && $humaneval_array_length -gt 1 ]]; then
    humaneval_max_branches_kept_single="${HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY[0]}"
    humaneval_branching_factor_single="${HUMANEVAL_BRANCHING_FACTORS_ARRAY[0]}"
    humaneval_branch_topp_single="${HUMANEVAL_BRANCH_TOPPS_ARRAY[0]}"
    humaneval_selection_conf_alpha_single="${HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY[0]}"
    humaneval_branch_verification_mode_single="${HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY[0]}"
    humaneval_base_branch_competition_single="${HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY[0]}"
    humaneval_verification_force_base_winner_single="${HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY[0]}"
    
    HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY=()
    HUMANEVAL_BRANCHING_FACTORS_ARRAY=()
    HUMANEVAL_BRANCH_TOPPS_ARRAY=()
    HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY=()
    HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY=()
    HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY=()
    HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY=()
    
    for ((k=0; k<$humaneval_array_length; k++)); do
        HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY+=("$humaneval_max_branches_kept_single")
        HUMANEVAL_BRANCHING_FACTORS_ARRAY+=("$humaneval_branching_factor_single")
        HUMANEVAL_BRANCH_TOPPS_ARRAY+=("$humaneval_branch_topp_single")
        HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY+=("$humaneval_selection_conf_alpha_single")
        HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY+=("$humaneval_branch_verification_mode_single")
        HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY+=("$humaneval_base_branch_competition_single")
        HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY+=("$humaneval_verification_force_base_winner_single")
    done
elif [[ $humaneval_multibranch_array_length -ne $humaneval_array_length ]]; then
    echo "错误：HumanEval多分支参数数组长度必须为1或与HumanEval主数组长度相同！"
    echo "HumanEval main length: $humaneval_array_length, HumanEval multi-branch params length: $humaneval_multibranch_array_length"
    exit 1
fi

export HF_ALLOW_CODE_EVAL=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/mnt/rl/xinyi/LoPA"


# 对每个LoRA模型进行评测
for lora_model in "${lora_models[@]}"; do
# 获取LoRA模型的基本名称用于输出目录
    lora_model_name="$lora_model"
    echo "===================================================================="
    echo "Evaluating LoRA model: $lora_model_name"
    echo "===================================================================="
    
    
    
    # HumanEval评估（参数列表遍历）
    for i in "${!HUMANEVAL_NSHOTS_ARRAY[@]}"; do
        output_path="results/eval_llada_chat_lopa${lora_model_name}/humaneval-ns${HUMANEVAL_NSHOTS_ARRAY[$i]}-len${HUMANEVAL_LENGTHS_ARRAY[$i]}-temp${HUMANEVAL_TEMP_ARRAY[$i]}-limit${HUMANEVAL_LIMITS_ARRAY[$i]}-diffsteps${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]}-block${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]}-thresh${HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY[$i]}-decodethresh${HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY[$i]}-skip${HUMANEVAL_SKIP_THRESHOLDS_ARRAY[$i]}-topp${HUMANEVAL_TOP_PS_ARRAY[$i]}-dtype${HUMANEVAL_DTYPES_ARRAY[$i]}-sampling${HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[$i]}-maxbranch${HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY[$i]}-branchfactor${HUMANEVAL_BRANCHING_FACTORS_ARRAY[$i]}-branchtopp${HUMANEVAL_BRANCH_TOPPS_ARRAY[$i]}-selconfal${HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY[$i]}-branchverify${HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY[$i]}-basecompete${HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY[$i]}-forcebase${HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY[$i]}"
        echo "Running HumanEval evaluation $((i+1))/${humaneval_array_length} for $lora_model_name..."
        echo "HumanEval Config: Shots: ${HUMANEVAL_NSHOTS_ARRAY[$i]}, Length: ${HUMANEVAL_LENGTHS_ARRAY[$i]}, Temperature: ${HUMANEVAL_TEMP_ARRAY[$i]}, Limit: ${HUMANEVAL_LIMITS_ARRAY[$i]}, Diffusion Steps: ${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]}, Block Size: ${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]}, Block Add Threshold: ${HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY[$i]}, Decoded Token Threshold: ${HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY[$i]}, Skip Threshold: ${HUMANEVAL_SKIP_THRESHOLDS_ARRAY[$i]}, Top_p: ${HUMANEVAL_TOP_PS_ARRAY[$i]}, Sampling Strategy: ${HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[$i]}, Dtype: ${HUMANEVAL_DTYPES_ARRAY[$i]}, Max Branches: ${HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY[$i]}, Branching Factor: ${HUMANEVAL_BRANCHING_FACTORS_ARRAY[$i]}, Branch Topp: ${HUMANEVAL_BRANCH_TOPPS_ARRAY[$i]}, Selection Conf Alpha: ${HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY[$i]}, Branch Verification: ${HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY[$i]}, Base Competition: ${HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY[$i]}, Force Base Winner: ${HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY[$i]}; Output: $output_path"
        
        # 构建HumanEval的model_args，根据top_p是否为none来决定是否包含top_p参数，并添加多分支参数
        if [[ "${HUMANEVAL_TOP_PS_ARRAY[$i]}" == "none" ]]; then
            humaneval_model_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${HUMANEVAL_LENGTHS_ARRAY[$i]},diffusion_steps=${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]},temperature=${HUMANEVAL_TEMP_ARRAY[$i]},add_bos_token=true,escape_until=true,block_size=${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]},block_add_threshold=${HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY[$i]},skip_threshold=${HUMANEVAL_SKIP_THRESHOLDS_ARRAY[$i]},decoded_token_threshold=${HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY[$i]},dtype=${HUMANEVAL_DTYPES_ARRAY[$i]},sampling_strategy=${HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[$i]},max_branches_kept=${HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY[$i]},branching_factor=${HUMANEVAL_BRANCHING_FACTORS_ARRAY[$i]},branch_topp=${HUMANEVAL_BRANCH_TOPPS_ARRAY[$i]},selection_conf_alpha=${HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY[$i]},branch_verification_mode=${HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY[$i]},base_branch_competition=${HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY[$i]},verification_force_base_winner=${HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY[$i]},save_dir=${output_path}"
        else
            humaneval_model_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${HUMANEVAL_LENGTHS_ARRAY[$i]},diffusion_steps=${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]},temperature=${HUMANEVAL_TEMP_ARRAY[$i]},top_p=${HUMANEVAL_TOP_PS_ARRAY[$i]},add_bos_token=true,escape_until=true,block_size=${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]},block_add_threshold=${HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY[$i]},skip_threshold=${HUMANEVAL_SKIP_THRESHOLDS_ARRAY[$i]},decoded_token_threshold=${HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY[$i]},dtype=${HUMANEVAL_DTYPES_ARRAY[$i]},sampling_strategy=${HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[$i]},max_branches_kept=${HUMANEVAL_MAX_BRANCHES_KEPTS_ARRAY[$i]},branching_factor=${HUMANEVAL_BRANCHING_FACTORS_ARRAY[$i]},branch_topp=${HUMANEVAL_BRANCH_TOPPS_ARRAY[$i]},selection_conf_alpha=${HUMANEVAL_SELECTION_CONF_ALPHAS_ARRAY[$i]},branch_verification_mode=${HUMANEVAL_BRANCH_VERIFICATION_MODES_ARRAY[$i]},base_branch_competition=${HUMANEVAL_BASE_BRANCH_COMPETITIONS_ARRAY[$i]},verification_force_base_winner=${HUMANEVAL_VERIFICATION_FORCE_BASE_WINNERS_ARRAY[$i]},save_dir=${output_path}"
        fi
        
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29520 --num_processes 8 eval_llada_xck_truenew.py --model dream_lora \
            --model_args $humaneval_model_args \
            --tasks humaneval \
            --num_fewshot ${HUMANEVAL_NSHOTS_ARRAY[$i]} \
            --batch_size 1 \
            --output_path $output_path \
            --log_samples \
            --confirm_run_unsafe_code
    done

    # 其他任务的评估
    for i in "${!TASKS_ARRAY[@]}"; do
        # Create comprehensive output path with all hyperparameters including LoRA-specific info and multi-branch params
        output_path="results/eval_llada_chat_lopa${lora_model_name}/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-len${LENGTH_ARRAY[$i]}-temp${TEMP_ARRAY[$i]}-limit${LIMITS_ARRAY[$i]}-diffsteps${LENGTH_ARRAY[$i]}-block${BLOCK_SIZES_ARRAY[$i]}-thresh${BLOCK_ADD_THRESHOLDS_ARRAY[$i]}-decodethresh${DECODED_TOKEN_THRESHOLDS_ARRAY[$i]}-skip${SKIP_THRESHOLDS_ARRAY[$i]}-topp${TOP_PS_ARRAY[$i]}-dtype${DTYPES_ARRAY[$i]}-sampling${SAMPLING_STRATEGIES_ARRAY[$i]}-maxbranch${MAX_BRANCHES_KEPTS_ARRAY[$i]}-branchfactor${BRANCHING_FACTORS_ARRAY[$i]}-branchtopp${BRANCH_TOPPS_ARRAY[$i]}-selconfal${SELECTION_CONF_ALPHAS_ARRAY[$i]}-branchverify${BRANCH_VERIFICATION_MODES_ARRAY[$i]}-basecompete${BASE_BRANCH_COMPETITIONS_ARRAY[$i]}-forcebase${VERIFICATION_FORCE_BASE_WINNERS_ARRAY[$i]}"
        echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}, Length: ${LENGTH_ARRAY[$i]}, Temperature: ${TEMP_ARRAY[$i]}, Limit: ${LIMITS_ARRAY[$i]}, Block Size: ${BLOCK_SIZES_ARRAY[$i]}, Block Add Threshold: ${BLOCK_ADD_THRESHOLDS_ARRAY[$i]}, Decoded Token Threshold: ${DECODED_TOKEN_THRESHOLDS_ARRAY[$i]}, Skip Threshold: ${SKIP_THRESHOLDS_ARRAY[$i]}, Top_p: ${TOP_PS_ARRAY[$i]}, Sampling Strategy: ${SAMPLING_STRATEGIES_ARRAY[$i]}, Dtype: ${DTYPES_ARRAY[$i]}, Max Branches: ${MAX_BRANCHES_KEPTS_ARRAY[$i]}, Branching Factor: ${BRANCHING_FACTORS_ARRAY[$i]}, Branch Topp: ${BRANCH_TOPPS_ARRAY[$i]}, Selection Conf Alpha: ${SELECTION_CONF_ALPHAS_ARRAY[$i]}, Branch Verification: ${BRANCH_VERIFICATION_MODES_ARRAY[$i]}, Base Competition: ${BASE_BRANCH_COMPETITIONS_ARRAY[$i]}, Force Base Winner: ${VERIFICATION_FORCE_BASE_WINNERS_ARRAY[$i]}; Output: $output_path"
        
        # 构建model_args，根据top_p是否为none来决定是否包含top_p参数，并添加多分支参数
        if [[ "${TOP_PS_ARRAY[$i]}" == "none" ]]; then
            model_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},block_size=${BLOCK_SIZES_ARRAY[$i]},block_add_threshold=${BLOCK_ADD_THRESHOLDS_ARRAY[$i]},skip_threshold=${SKIP_THRESHOLDS_ARRAY[$i]},decoded_token_threshold=${DECODED_TOKEN_THRESHOLDS_ARRAY[$i]},dtype=${DTYPES_ARRAY[$i]},sampling_strategy=${SAMPLING_STRATEGIES_ARRAY[$i]},max_branches_kept=${MAX_BRANCHES_KEPTS_ARRAY[$i]},branching_factor=${BRANCHING_FACTORS_ARRAY[$i]},branch_topp=${BRANCH_TOPPS_ARRAY[$i]},selection_conf_alpha=${SELECTION_CONF_ALPHAS_ARRAY[$i]},branch_verification_mode=${BRANCH_VERIFICATION_MODES_ARRAY[$i]},base_branch_competition=${BASE_BRANCH_COMPETITIONS_ARRAY[$i]},verification_force_base_winner=${VERIFICATION_FORCE_BASE_WINNERS_ARRAY[$i]},save_dir=${output_path}"
        else
            model_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},top_p=${TOP_PS_ARRAY[$i]},block_size=${BLOCK_SIZES_ARRAY[$i]},block_add_threshold=${BLOCK_ADD_THRESHOLDS_ARRAY[$i]},skip_threshold=${SKIP_THRESHOLDS_ARRAY[$i]},decoded_token_threshold=${DECODED_TOKEN_THRESHOLDS_ARRAY[$i]},dtype=${DTYPES_ARRAY[$i]},sampling_strategy=${SAMPLING_STRATEGIES_ARRAY[$i]},max_branches_kept=${MAX_BRANCHES_KEPTS_ARRAY[$i]},branching_factor=${BRANCHING_FACTORS_ARRAY[$i]},branch_topp=${BRANCH_TOPPS_ARRAY[$i]},selection_conf_alpha=${SELECTION_CONF_ALPHAS_ARRAY[$i]},branch_verification_mode=${BRANCH_VERIFICATION_MODES_ARRAY[$i]},base_branch_competition=${BASE_BRANCH_COMPETITIONS_ARRAY[$i]},verification_force_base_winner=${VERIFICATION_FORCE_BASE_WINNERS_ARRAY[$i]},save_dir=${output_path}"
        fi

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29520 --num_processes 8 eval_llada_xck_truenew.py --model dream_lora \
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