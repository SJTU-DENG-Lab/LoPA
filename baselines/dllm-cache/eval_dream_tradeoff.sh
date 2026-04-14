 # Number of processes (should match the number of GPUs you're using)

export HF_ALLOW_CODE_EVAL=1
export CUDA_VISIBLE_DEVICES="4,5,6,7"
NUM_PROCESSES=4 


model="/home/chenkai/data/models/Dream-v0-Base-7B"


ACCEL_CONFIG="accelerate_config.yaml"
MAIN_PORT="29510" 

echo "Starting evaluation for gsm8k_cot"

# --- Task Specific Parameters for gsm8k ---
TASK="gsm8k"
NUM_FEWSHOT=4     # From tasks="gsm8k ...", nshots="4 ..."
MAX_NEW_TOKENS=512 # From tasks="gsm8k ...", lengths="256 ..."
DIFFUSION_STEPS=512 # Note: based on original script (equal to max_new_tokens)
TEMPERATURE=0.2    # From tasks="gsm8k   ...", temperatures="0 ..."
TOP_P=0.95        # Constant in the original loop's model_args
ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# Note: original loop did NOT include escape_until=true

OUTPUT_PATH="./eval_results_dream_tradeoff/${TASK}_length${MAX_NEW_TOKENS}_steps${DIFFUSION_STEPS}_prompt-1_gen-1"


OUTPUT_PATH="./eval_results_dream_tradeoff/${TASK}_length256_steps256_prompt100_gen8"
accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} --num_processes ${NUM_PROCESSES} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},alg="entropy",alg_temp=0.0,prompt_interval_steps=100,gen_interval_steps=8,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --confirm_run_unsafe_code

echo "Completed evaluation for ${TASK}"




# --- Task Specific Parameters for gsm8k ---
TASK="gsm8k"
NUM_FEWSHOT=4     # From tasks="gsm8k ...", nshots="4 ..."
MAX_NEW_TOKENS=512 # From tasks="gsm8k ...", lengths="256 ..."
DIFFUSION_STEPS=256 # Note: based on original script (equal to max_new_tokens)
TEMPERATURE=0.2    # From tasks="gsm8k   ...", temperatures="0 ..."
TOP_P=0.95        # Constant in the original loop's model_args
ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# Note: original loop did NOT include escape_until=true

OUTPUT_PATH="./eval_results_dream_tradeoff/${TASK}_length${MAX_NEW_TOKENS}_steps${DIFFUSION_STEPS}_prompt-1_gen-1"


OUTPUT_PATH="./eval_results_dream_tradeoff/${TASK}_length256_steps256_prompt100_gen8"
accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} --num_processes ${NUM_PROCESSES} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},alg="entropy",alg_temp=0.0,prompt_interval_steps=100,gen_interval_steps=8,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --confirm_run_unsafe_code

echo "Completed evaluation for ${TASK}"



# --- Task Specific Parameters for gsm8k ---
TASK="gsm8k"
NUM_FEWSHOT=4     # From tasks="gsm8k ...", nshots="4 ..."
MAX_NEW_TOKENS=512 # From tasks="gsm8k ...", lengths="256 ..."
DIFFUSION_STEPS=128 # Note: based on original script (equal to max_new_tokens)
TEMPERATURE=0.2    # From tasks="gsm8k   ...", temperatures="0 ..."
TOP_P=0.95        # Constant in the original loop's model_args
ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# Note: original loop did NOT include escape_until=true

OUTPUT_PATH="./eval_results_dream_tradeoff/${TASK}_length${MAX_NEW_TOKENS}_steps${DIFFUSION_STEPS}_prompt-1_gen-1"


OUTPUT_PATH="./eval_results_dream_tradeoff/${TASK}_length256_steps256_prompt100_gen8"
accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} --num_processes ${NUM_PROCESSES} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},alg="entropy",alg_temp=0.0,prompt_interval_steps=100,gen_interval_steps=8,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False,save_dir=${OUTPUT_PATH} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --confirm_run_unsafe_code

echo "Completed evaluation for ${TASK}"






