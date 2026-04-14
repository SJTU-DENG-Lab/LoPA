#!/bin/bash

model="/home/chenkai/data/models/Dream-v0-Base-7B"

export HF_ALLOW_CODE_EVAL=1

ACCEL_CONFIG="accelerate_config.yaml"
MAIN_PORT="29510" 

echo "Starting evaluation for gsm8k_cot"



# --- Task Specific Parameters for mbpp ---
TASK="gsm8k_cot"
NUM_FEWSHOT=8
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256 # Note: based on original script
TEMPERATURE=0.2
TOP_P=0.95
LIMIT=10000
SEED=1
ADD_BOS_TOKEN="true"
ESCAPE_UNTIL="false" # Note: specific to the mbpp run in original script

OUTPUT_PATH="./eval_results_dream_${SEED}/${TASK}_length256_steps256_prompt-1_gen-1"

accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},alg="entropy",alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --limit ${LIMIT} \
    --seed ${SEED} \
    --confirm_run_unsafe_code

OUTPUT_PATH="./eval_results_dream_${SEED}/${TASK}_length256_steps256_prompt5_gen1"
accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},alg="entropy",alg_temp=0.0,prompt_interval_steps=5,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --limit ${LIMIT} \
    --seed ${SEED} \
    --confirm_run_unsafe_code

echo "Completed evaluation for ${TASK}"

# ### NOTICE: use postprocess for mbpp
# # python postprocess_code.py {the samples_xxx.jsonl file under output_path}


TASK="gsm8k_cot"
NUM_FEWSHOT=8
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256 # Note: based on original script
TEMPERATURE=0.2
TOP_P=0.95
LIMIT=10000
SEED=2
ADD_BOS_TOKEN="true"
ESCAPE_UNTIL="false" # Note: specific to the mbpp run in original script

OUTPUT_PATH="./eval_results_dream_${SEED}/${TASK}_length256_steps256_prompt-1_gen-1"

accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},alg="entropy",alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --limit ${LIMIT} \
    --seed ${SEED} \
    --confirm_run_unsafe_code

OUTPUT_PATH="./eval_results_dream_${SEED}/${TASK}_length256_steps256_prompt5_gen1"
accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},alg="entropy",alg_temp=0.0,prompt_interval_steps=5,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --limit ${LIMIT} \
    --seed ${SEED} \
    --confirm_run_unsafe_code

echo "Completed evaluation for ${TASK}"

# ### NOTICE: use postprocess for mbpp
# # python postprocess_code.py {the samples_xxx.jsonl file under output_path}



TASK="gsm8k_cot"
NUM_FEWSHOT=8
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256 # Note: based on original script
TEMPERATURE=0.2
TOP_P=0.95
LIMIT=10000
SEED=3
ADD_BOS_TOKEN="true"
ESCAPE_UNTIL="false" # Note: specific to the mbpp run in original script

OUTPUT_PATH="./eval_results_dream_${SEED}/${TASK}_length256_steps256_prompt-1_gen-1"

accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},alg="entropy",alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --limit ${LIMIT} \
    --seed ${SEED} \
    --confirm_run_unsafe_code

OUTPUT_PATH="./eval_results_dream_${SEED}/${TASK}_length256_steps256_prompt5_gen1"
accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},alg="entropy",alg_temp=0.0,prompt_interval_steps=5,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --limit ${LIMIT} \
    --seed ${SEED} \
    --confirm_run_unsafe_code

echo "Completed evaluation for ${TASK}"

# ### NOTICE: use postprocess for mbpp
# # python postprocess_code.py {the samples_xxx.jsonl file under output_path}



TASK="gsm8k_cot"
NUM_FEWSHOT=8
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256 # Note: based on original script
TEMPERATURE=0.2
TOP_P=0.95
LIMIT=10000
SEED=4
ADD_BOS_TOKEN="true"
ESCAPE_UNTIL="false" # Note: specific to the mbpp run in original script

OUTPUT_PATH="./eval_results_dream_${SEED}/${TASK}_length256_steps256_prompt-1_gen-1"

accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},alg="entropy",alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --limit ${LIMIT} \
    --seed ${SEED} \
    --confirm_run_unsafe_code

OUTPUT_PATH="./eval_results_dream_${SEED}/${TASK}_length256_steps256_prompt5_gen1"
accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},alg="entropy",alg_temp=0.0,prompt_interval_steps=5,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --limit ${LIMIT} \
    --seed ${SEED} \
    --confirm_run_unsafe_code

echo "Completed evaluation for ${TASK}"

# ### NOTICE: use postprocess for mbpp
# # python postprocess_code.py {the samples_xxx.jsonl file under output_path}



TASK="gsm8k_cot"
NUM_FEWSHOT=8
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256 # Note: based on original script
TEMPERATURE=0.2
TOP_P=0.95
LIMIT=10000
SEED=5
ADD_BOS_TOKEN="true"
ESCAPE_UNTIL="false" # Note: specific to the mbpp run in original script

OUTPUT_PATH="./eval_results_dream_${SEED}/${TASK}_length256_steps256_prompt-1_gen-1"

accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},alg="entropy",alg_temp=0.0,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --limit ${LIMIT} \
    --seed ${SEED} \
    --confirm_run_unsafe_code

OUTPUT_PATH="./eval_results_dream_${SEED}/${TASK}_length256_steps256_prompt5_gen1"
accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},escape_until=${ESCAPE_UNTIL},alg="entropy",alg_temp=0.0,prompt_interval_steps=5,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --limit ${LIMIT} \
    --seed ${SEED} \
    --confirm_run_unsafe_code

echo "Completed evaluation for ${TASK}"

# ### NOTICE: use postprocess for mbpp
# # python postprocess_code.py {the samples_xxx.jsonl file under output_path}






