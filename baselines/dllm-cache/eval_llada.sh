# accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=/data1/xck/models/llada-8b-instruct,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0,cache_order=0,is_feature_cache=False,is_cfg_cache=False,save_dir=./eval_results_llada/gsm8k_log_block8_len256_steps256_prompt-1_gen-1" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0"  \
# --num_fewshot 4  \
# --output_path ./eval_results_llada/gsm8k_log_block8_len256_steps256_prompt-1_gen-1 \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \

# accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,prompt_interval_steps=50,gen_interval_steps=7,transfer_ratio=0.25,cache_order=0,is_feature_cache=True,is_cfg_cache=False,save_dir=./eval_results_llada/gsm8k_log_block8_len256_steps256_prompt50_gen7" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0"  \
# --num_fewshot 4  \
# --output_path ./eval_results_llada/gsm8k_log_block8_len256_steps256_prompt50_gen7 \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \


# accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0,cache_order=0,is_feature_cache=False,is_cfg_cache=False,save_dir=./eval_results_llada/humaneval_log_block32_len512_steps512_prompt-1_gen-1" \
# --gen_kwargs "block_length=32,gen_length=512,steps=512,cfg_scale=0.0"  \
# --output_path ./eval_results_llada/humaneval_log_block32_len512_steps512_prompt-1_gen-1 \
# --log_samples \
# --confirm_run_unsafe_code \

# accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,prompt_interval_steps=50,gen_interval_steps=8,transfer_ratio=0.25,cache_order=0,is_feature_cache=True,is_cfg_cache=False,save_dir=./eval_results_llada/humaneval_log_block32_len512_steps512_prompt50_gen8" \
# --gen_kwargs "block_length=32,gen_length=512,steps=512,cfg_scale=0.0"  \
# --output_path ./eval_results_llada/humaneval_log_block32_len512_steps512_prompt50_gen8 \
# --log_samples \
# --confirm_run_unsafe_code \


# accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,prompt_interval_steps=25,gen_interval_steps=5,transfer_ratio=0.25,cache_order=0,is_feature_cache=True,is_cfg_cache=False,save_dir=./eval_results_llada/humaneval_log_block32_len512_steps512_prompt25_gen5" \
# --gen_kwargs "block_length=32,gen_length=512,steps=512,cfg_scale=0.0"  \
# --output_path ./eval_results_llada/humaneval_log_block32_len512_steps512_prompt25_gen5 \
# --log_samples \
# --confirm_run_unsafe_code \


# accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks mbpp --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,prompt_interval_steps=-1,gen_interval_steps=-1,cache_order=0,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=./eval_results_llada/mbpp_log_block32_len512_steps512_prompt-1_gen-1" \
# --gen_kwargs "block_length=32,gen_length=512,steps=512,cfg_scale=0.0,remasking="low_confidence""  \
# --num_fewshot 3  \
# --output_path ./eval_results_llada/mbpp_log_block32_len512_steps512_prompt-1_gen-1 \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \
# --confirm_run_unsafe_code \

# accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks mbpp --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,prompt_interval_steps=100,gen_interval_steps=5,cache_order=0,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False,save_dir=./eval_results_llada/mbpp_log_block32_len512_steps512_prompt100_gen5" \
# --gen_kwargs "block_length=32,gen_length=512,steps=512,cfg_scale=0.0,remasking="low_confidence""  \
# --num_fewshot 3  \
# --output_path ./eval_results_llada/mbpp_log_block32_len512_steps512_prompt100_gen5 \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \
# --confirm_run_unsafe_code \



# accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks minerva_math --batch_size 1 \
# --model_args "pretrained=/data1/xck/models/llada-8b-instruct,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False,save_dir=./eval_results_llada/minerva_math_log_block256_len256_steps256_prompt-1_gen-1" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./eval_results_llada/minerva_math_log_block256_len256_steps256_prompt-1_gen-1 \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \

accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLaDA --tasks minerva_math --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,prompt_interval_steps=50,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False,save_dir=./eval_results_llada/minerva_math_log_block256_len256_steps256_prompt50_gen1" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./eval_results_llada/minerva_math_log_block256_len256_steps256_prompt50_gen1 \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \

