# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
ACCEL_CONFIG="accelerate_config.yaml"


############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=4
steps=256
output_path=evals_results/parallel/gsm8k-ns${num_fewshot}-${length}


# dParallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29601 --config_file ${ACCEL_CONFIG} eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='/home/chenkai/data/models/dParallel_LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="gsm8k",save_dir="${output_path}" \
--output_path ${output_path} --log_samples








############################################### humaneval evaluations ###############################################
task=humaneval
length=256
block_length=32
num_fewshot=0
steps=256
output_path=evals_results/parallel/humaneval-ns${num_fewshot}-${length}


# dparallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29601 --config_file ${ACCEL_CONFIG} eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='/home/chenkai/data/models/dParallel_LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.5,show_speed=True,task="humaneval",save_dir="${output_path}" \
--output_path ${output_path} --log_samples

## NOTICE: use postprocess for humaneval
# python postprocess_code_humaneval.py {the samples_xxx.jsonl file under output_path}





############################################### mbpp evaluations ###############################################
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=256
output_path=evals_results/parallel/mbpp-ns${num_fewshot}-${length}


# parallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29601 --config_file ${ACCEL_CONFIG} eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='/home/chenkai/data/models/dParallel_LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.45,show_speed=True,task="mbpp",save_dir="${output_path}" \
--output_path ${output_path} --log_samples

## NOTICE: use postprocess for mbpp
# python postprocess_code_mbpp.py {the samples_xxx.jsonl file under output_path}


############################################### minerva_math evaluations ###############################################
task=minerva_math
length=256
block_length=32
num_fewshot=4
steps=256
output_path=evals_results/parallel/minerva_math-ns${num_fewshot}-${length}


# dParallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29601 --config_file ${ACCEL_CONFIG} eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='/home/chenkai/data/models/dParallel_LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="minerva_math",save_dir="${output_path}" \
--output_path ${output_path} --log_samples