#!/bin/bash

model="Dream-org/Dream-v0-Base-7B"
# 设置LoRA路径，如果不使用LoRA，留空即可
lora_path="/data1/xck/ckpt/wx_dream_base_sft"  # 例如: lora_path="/path/to/your/lora/model"

export HF_ALLOW_CODE_EVAL=1

ACCEL_CONFIG="accelerate_config.yaml"
MAIN_PORT="29510" 

# echo "Starting evaluation for gsm8k_cot"

# # --- Task Specific Parameters for gsm8k_cot ---
# TASK="gsm8k_cot"
# NUM_FEWSHOT=8     # From tasks="gsm8k_cot ...", nshots="8 ..."
# MAX_NEW_TOKENS=256 # From tasks="gsm8k_cot ...", lengths="256 ..."
# DIFFUSION_STEPS=256 # Note: based on original script (equal to max_new_tokens)
# TEMPERATURE=0.2    # From tasks="gsm8k_cot ...", temperatures="0 ..."
# TOP_P=0.95        # Constant in the original loop's model_args
# ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# # Note: original loop did NOT include escape_until=true

# OUTPUT_PATH="./eval_results_dream_sft/${TASK}_length256_steps256_prompt-1_gen-1"

# # 构建model_args，根据是否有lora_path决定是否添加lora_path参数
# if [ -n "$lora_path" ]; then
#     MODEL_ARGS="pretrained=${model},lora_path=${lora_path},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},alg=entropy,alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH}"
# else
#     MODEL_ARGS="pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},alg=entropy,alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH}"
# fi

# accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
#     --model_args ${MODEL_ARGS} \
#     --tasks ${TASK} \
#     --num_fewshot ${NUM_FEWSHOT} \
#     --batch_size 1 \
#     --output_path ${OUTPUT_PATH} \
#     --log_samples \
#     --confirm_run_unsafe_code




# # --- Task Specific Parameters for humaneval ---
# TASK="humaneval"
# NUM_FEWSHOT=0
# MAX_NEW_TOKENS=256
# DIFFUSION_STEPS=256 # Note: based on original script
# TEMPERATURE=0.2
# TOP_P=0.95
# ADD_BOS_TOKEN="true"
# ESCAPE_UNTIL="true" # Note: specific to the humaneval run in original script

# OUTPUT_PATH="./eval_results_dream_sft/${TASK}_length256_steps256_prompt-1_gen-1"

# # 构建model_args，根据是否有lora_path决定是否添加lora_path参数
# if [ -n "$lora_path" ]; then
#     MODEL_ARGS="pretrained=${model},lora_path=${lora_path},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},alg=entropy,alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH}"
# else
#     MODEL_ARGS="pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},alg=entropy,alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH}"
# fi

# accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
#     --model_args ${MODEL_ARGS} \
#     --tasks ${TASK} \
#     --num_fewshot ${NUM_FEWSHOT} \
#     --batch_size 1 \
#     --output_path ${OUTPUT_PATH} \
#     --log_samples \
#     --confirm_run_unsafe_code


# ### NOTICE: use postprocess for humaneval
# # python postprocess_code.py {the samples_xxx.jsonl file under output_path}






echo "Starting evaluation for mbpp"

# --- Task Specific Parameters for mbpp ---
TASK="mbpp"
NUM_FEWSHOT=3     # From tasks="... mbpp ...", nshots="... 3 ..."
MAX_NEW_TOKENS=256 # From tasks="... mbpp ...", lengths="... 512 ..."
DIFFUSION_STEPS=256 # Note: based on original script (equal to max_new_tokens)
TEMPERATURE=0.2   # From tasks="... mbpp ...", temperatures="... 0.2 ..."
TOP_P=0.95        # Constant in the original loop's model_args
ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# Note: original loop did NOT include escape_until=true

OUTPUT_PATH="./eval_results_dream_sft/${TASK}_length256_steps256_prompt-1_gen-1"

# 构建model_args，根据是否有lora_path决定是否添加lora_path参数
if [ -n "$lora_path" ]; then
    MODEL_ARGS="pretrained=${model},lora_path=${lora_path},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},alg=entropy,alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH}"
else
    MODEL_ARGS="pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},alg=entropy,alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH}"
fi

accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args ${MODEL_ARGS} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --confirm_run_unsafe_code






echo "Starting evaluation for minerva_math"

# --- Task Specific Parameters for minerva_math ---
TASK="minerva_math"
NUM_FEWSHOT=4     # From tasks="... minerva_math ...", nshots="... 4 ..."
MAX_NEW_TOKENS=256 # From tasks="... minerva_math ...", lengths="... 512 ..."
DIFFUSION_STEPS=256 # Note: based on original script (equal to max_new_tokens)
TEMPERATURE=0.2    # From tasks="... minerva_math ...", temperatures="... 0 ..."
TOP_P=0.95        # Constant in the original loop's model_args
ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# Note: original loop did NOT include escape_until=true

OUTPUT_PATH="./eval_results_dream_sft/${TASK}_length256_steps256_prompt-1_gen-1"

# 构建model_args，根据是否有lora_path决定是否添加lora_path参数
if [ -n "$lora_path" ]; then
    MODEL_ARGS="pretrained=${model},lora_path=${lora_path},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH}"
else
    MODEL_ARGS="pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH}"
fi

accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args ${MODEL_ARGS} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --confirm_run_unsafe_code



