export HF_ALLOW_CODE_EVAL=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/mnt/rl/xinyi/LoPA"


accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=/mnt/rl/xinyi/models/LLaDA-8B-Instruct,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0,cache_order=0,is_feature_cache=False,is_cfg_cache=False,save_dir=./eval_results_llada/gsm8k_log_block32_len256_steps256_prompt-1_gen-1" \
--gen_kwargs "block_length=32,gen_length=256,steps=256,cfg_scale=0.0"  \
--num_fewshot 4  \
--output_path ./eval_results_llada/gsm8k_log_block32_len256_steps256_prompt-1_gen-1 \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \



accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks humaneval --batch_size 1 \
--model_args "pretrained=/mnt/rl/xinyi/models/LLaDA-8B-Instruct,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0,cache_order=0,is_feature_cache=False,is_cfg_cache=False,save_dir=./eval_results_llada/humaneval_log_block32_len256_steps256_prompt-1_gen-1" \
--gen_kwargs "block_length=32,gen_length=256,steps=256,cfg_scale=0.0"  \
--output_path ./eval_results_llada/humaneval_log_block32_len256_steps256_prompt-1_gen-1 \
--log_samples \
--confirm_run_unsafe_code \




accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks mbpp --batch_size 1 \
--model_args "pretrained=/mnt/rl/xinyi/models/LLaDA-8B-Instruct,prompt_interval_steps=-1,gen_interval_steps=-1,cache_order=0,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=./eval_results_llada/mbpp_log_block32_len256_steps256_prompt-1_gen-1" \
--gen_kwargs "block_length=32,gen_length=256,steps=256,cfg_scale=0.0,remasking="low_confidence""  \
--num_fewshot 3  \
--output_path ./eval_results_llada/mbpp_log_block32_len256_steps256_prompt-1_gen-1 \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \
--confirm_run_unsafe_code \



accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks minerva_math --batch_size 1 \
--model_args "pretrained=/mnt/rl/xinyi/models/LLaDA-8B-Instruct,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=./eval_results_llada/minerva_math_log_block32_len256_steps256_prompt-1_gen-1" \
--gen_kwargs "block_length=32,gen_length=256,steps=256,cfg_scale=0.0 "  \
--num_fewshot 4  \
--output_path ./eval_results_llada/minerva_math_log_block32_len256_steps256_prompt-1_gen-1 \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \