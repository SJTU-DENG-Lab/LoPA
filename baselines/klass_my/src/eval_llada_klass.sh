# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/mnt/rl/xinyi/LoPA"


############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=64
num_fewshot=4
steps=256
alg=klass
unmask_strategy=all
conf_threshold=0.6
kl_threshold=0.015
history_length=2
output_path=evals_results/klass-earlystop/gsm8k-ns${num_fewshot}-${length}


# dParallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29601 --num_processes 8 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='/mnt/rl/xinyi/models/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="gsm8k",save_dir="${output_path}",alg="${alg}",unmask_strategy="${unmask_strategy}",conf_threshold=${conf_threshold},kl_threshold=${kl_threshold},history_length=${history_length} \
--output_path ${output_path} --log_samples




############################################### minerva_math evaluations ###############################################
task=minerva_math
length=256
block_length=64
num_fewshot=4
steps=256
alg=klass
unmask_strategy=all
conf_threshold=0.6
kl_threshold=0.01
history_length=2
output_path=evals_results/klass-earlystop/minerva_math-ns${num_fewshot}-${length}


# dParallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29601 --num_processes 8 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='/mnt/rl/xinyi/models/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="minerva_math",save_dir="${output_path}",alg="${alg}",unmask_strategy="${unmask_strategy}",conf_threshold=${conf_threshold},kl_threshold=${kl_threshold},history_length=${history_length} \
--output_path ${output_path} --log_samples



############################################### humaneval evaluations ###############################################
task=humaneval
length=256
block_length=64
num_fewshot=0
steps=256
alg=klass
unmask_strategy=all
conf_threshold=0.9
kl_threshold=0.01
history_length=2
output_path=evals_results/klass-earlystop/humaneval-ns${num_fewshot}-${length}


# dparallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29601 --num_processes 8 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='/mnt/rl/xinyi/models/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.5,show_speed=True,task="humaneval",save_dir="${output_path}",alg="${alg}",unmask_strategy="${unmask_strategy}",conf_threshold=${conf_threshold},kl_threshold=${kl_threshold},history_length=${history_length} \
--output_path ${output_path} --log_samples

## NOTICE: use postprocess for humaneval
# python postprocess_code_humaneval.py {the samples_xxx.jsonl file under output_path}





############################################### mbpp evaluations ###############################################
task=mbpp
length=256
block_length=64
num_fewshot=3
steps=256
alg=klass
unmask_strategy=all
conf_threshold=0.7
kl_threshold=0.01
history_length=2
output_path=evals_results/klass-earlystop/mbpp-ns${num_fewshot}-${length}


# parallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29601 --num_processes 8 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='/mnt/rl/xinyi/models/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.45,show_speed=True,task="mbpp",save_dir="${output_path}",alg="${alg}",unmask_strategy="${unmask_strategy}",conf_threshold=${conf_threshold},kl_threshold=${kl_threshold},history_length=${history_length} \
--output_path ${output_path} --log_samples

## NOTICE: use postprocess for mbpp
# python postprocess_code_mbpp.py {the samples_xxx.jsonl file under output_path}
