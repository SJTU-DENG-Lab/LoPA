############################################### gsm8k evaluations ###############################################
export HF_ENDPOINT=https://hf-mirror.com
ACCEL_CONFIG="accelerate_config.yaml"
export HF_ALLOW_CODE_EVAL=1
MODEL_PATH="/home/chenkai/data/models/Dream-v0-Instruct-7B"
## Original dllm
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True,max_new_tokens=256,diffusion_steps=128,dtype="bfloat16",temperature=0.2,top_p=0.95,alg="entropy" \
    --tasks gsm8k \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 4 \
    --output_path output_reproduce_baseline_2t/gsm8k \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template



############################################### minerva_math evaluations ###############################################




############################################### humaneval evaluations ###############################################

## Original dllm
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True,max_new_tokens=512,diffusion_steps=256,dtype="bfloat16",temperature=0.2,top_p=0.95,alg="entropy" \
    --tasks humaneval \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce_baseline_2t/humaneval \
    --log_samples --confirm_run_unsafe_code \
    # --apply_chat_template





############################################### mbpp evaluations ###############################################

## Original dllm
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True,max_new_tokens=256,diffusion_steps=128,dtype="bfloat16",temperature=0.2,top_p=0.95,alg="entropy" \
    --tasks mbpp \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 3 \
    --output_path output_reproduce_baseline_2t/mbpp \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template


## Original dllm
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True,max_new_tokens=256,diffusion_steps=128,dtype="bfloat16",temperature=0.2,top_p=0.95,alg="entropy" \
    --tasks minerva_math \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 4 \
    --output_path output_reproduce_baseline_2t/math \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template



