############################################### gsm8k evaluations ###############################################

export HF_ENDPOINT=https://hf-mirror.com
ACCEL_CONFIG="accelerate_config.yaml"
export HF_ALLOW_CODE_EVAL=1

# our dParallel
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="/home/chenkai/data/models/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,threshold=0.45,stats_save_path="output_reproduce_fix/gsm8k/generation_stats" \
#     --tasks gsm8k \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 4 \
#     --output_path output_reproduce_fix/gsm8k \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

############################################## minerva_math evaluations ###############################################


# our dParallel



############################################### humaneval evaluations ###############################################


## our dParallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained="/home/chenkai/data/models/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=512,diffusion_steps=512,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,threshold=0.5,escape_until=True,stats_save_path="output_reproduce_fix/humaneval/generation_stats" \
    --tasks humaneval \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce_fix/humaneval \
    --log_samples --confirm_run_unsafe_code \
    # --apply_chat_template



############################################### mbpp evaluations ###############################################


## our dParallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained="/home/chenkai/data/models/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,threshold=0.5,stats_save_path="output_reproduce_fix/mbpp/generation_stats" \
    --tasks mbpp \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce_fix/mbpp \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template





CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained="/home/chenkai/data/models/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,threshold=0.45,stats_save_path="output_reproduce_fix/math/generation_stats" \
    --tasks minerva_math \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 4 \
    --output_path output_reproduce_fix/math \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template