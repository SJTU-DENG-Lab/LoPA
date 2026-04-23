#!/bin/bash

model="/home/chenkai/data/models/Dream-v0-Instruct-7B"

export HF_ALLOW_CODE_EVAL=1
# export CURL_CA_BUNDLE=""
# export REQUESTS_CA_BUNDLE=""
export HF_ENDPOINT="https://hf-mirror.com"
# export HF_HOME="/mnt/rl/xinyi/LoPA"

ACCEL_CONFIG="accelerate_config.yaml"
MAIN_PORT="29510"

# root output directory for all tasks
OUTPUT_ROOT="./eval_results_dream_instruct_t02p095"
mkdir -p "${OUTPUT_ROOT}"

echo "Starting evaluation for gsm8k"

# --- Task Specific Parameters for gsm8k ---
# TASK="gsm8k"
# NUM_FEWSHOT=4     # From tasks="gsm8k ...", nshots="8 ..."
# MAX_NEW_TOKENS=256 # From tasks="gsm8k ...", lengths="256 ..."
# DIFFUSION_STEPS=256 # Note: based on original script (equal to max_new_tokens)
# TEMPERATURE=0.2    # From tasks="gsm8k_cot ...", temperatures="0 ..."
# TOP_P=0.95        # Constant in the original loop's model_args
# ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# # Note: original loop did NOT include escape_until=true

# OUTPUT_PATH="${OUTPUT_ROOT}/${TASK}_log"

# accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="entropy",alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
#     --tasks ${TASK} \
#     --num_fewshot ${NUM_FEWSHOT} \
#     --batch_size 1 \
#     --output_path ${OUTPUT_PATH} \
#     --log_samples \
#     --confirm_run_unsafe_code


# echo "Completed evaluation for ${TASK}"


echo "Starting evaluation for humaneval"

# --- Task Specific Parameters for humaneval ---
TASK="humaneval"
NUM_FEWSHOT=0
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256 # Note: based on original script
TEMPERATURE=0.2
TOP_P=0.95
ADD_BOS_TOKEN="true"
ESCAPE_UNTIL="true" # Note: specific to the humaneval run in original script

OUTPUT_PATH="${OUTPUT_ROOT}/${TASK}_log_256"

accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="entropy",alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --confirm_run_unsafe_code



echo "Completed evaluation for ${TASK}"





# # --- Task Specific Parameters for mbpp ---
# TASK="mbpp"
# NUM_FEWSHOT=3     # From tasks="... mbpp ...", nshots="... 3 ..."
# MAX_NEW_TOKENS=256 # From tasks="... mbpp ...", lengths="... 512 ..."
# DIFFUSION_STEPS=256 # Note: based on original script (equal to max_new_tokens)
# TEMPERATURE=0.2    # From tasks="... mbpp ...", temperatures="... 0.2 ..."
# TOP_P=0.95        # Constant in the original loop's model_args
# ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# # Note: original loop did NOT include escape_until=true


# OUTPUT_PATH="${OUTPUT_ROOT}/${TASK}_log"

# accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="entropy",alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,add_bos_token=${ADD_BOS_TOKEN},save_dir=${OUTPUT_PATH} \
#     --tasks ${TASK} \
#     --num_fewshot ${NUM_FEWSHOT} \
#     --batch_size 1 \
#     --output_path ${OUTPUT_PATH} \
#     --log_samples \
#     --confirm_run_unsafe_code



# echo "Completed evaluation for ${TASK}"






# # --- Task Specific Parameters for minerva_math ---
# TASK="minerva_math"
# NUM_FEWSHOT=4     # From tasks="... minerva_math ...", nshots="... 4 ..."
# MAX_NEW_TOKENS=256 # From tasks="... minerva_math ...", lengths="... 512 ..."
# DIFFUSION_STEPS=256 # Note: based on original script (equal to max_new_tokens)
# TEMPERATURE=0.2    # From tasks="... minerva_math ...", temperatures="... 0 ..."
# TOP_P=0.95        # Constant in the original loop's model_args
# ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# # Note: original loop did NOT include escape_until=true


# OUTPUT_PATH="${OUTPUT_ROOT}/${TASK}_log"

# accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="entropy",alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,add_bos_token=${ADD_BOS_TOKEN},save_dir=${OUTPUT_PATH} \
#     --tasks ${TASK} \
#     --num_fewshot ${NUM_FEWSHOT} \
#     --batch_size 1 \
#     --output_path ${OUTPUT_PATH} \
#     --log_samples \
#     --confirm_run_unsafe_code


# echo "Completed evaluation for ${TASK}"


