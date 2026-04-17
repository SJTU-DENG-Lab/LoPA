# tasks="gsm8k_cot mbpp minerva_math bbh"
# nshots="8 3 4 3"
# lengths="256 512 512 512"
# temperatures="0 0.2 0 0"
# limits="100 100 100 100"
# tasks="gsm8k_cot mbpp minerva_math"
# nshots="8 3 4"
# lengths="256 512 512"
# temperatures="0.2 0.2 0.2"
# limits="200 200 200"
# dtypes="float32 float32 float32"

# tasks="gsm8k_cot gsm8k_cot mbpp mbpp minerva_math minerva_math"
# nshots="5 5 3 3 4 4"
# lengths="256 256 512 512 512 512"
# temperatures="0 0.2 0 0.2 0 0.2"
# limits="200 200 200 200 200 200"
# dtypes="bfloat16 bfloat16 bfloat16 bfloat16 bfloat16 bfloat16"

tasks="gsm8k mbpp minerva_math"
nshots="4 3 4"
lengths="256 256 256"
temperatures="0.2 0.2 0"
limits="10000 10000 10000"
dtypes="bfloat16 bfloat16 bfloat16"

tasks="gsm8k"
nshots="4"
lengths="256"
temperatures="0.2"
limits="100"
dtypes="bfloat16"
decode_strategies="left_to_right right_to_left confidence_max random oracle"

model=/home/chenkai/data/models/Dream-v0-Instruct-7B
# Create arrays from space-separated strings
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra TEMP_ARRAY <<< "$temperatures"
read -ra LIMITS_ARRAY <<< "$limits"
read -ra DTYPES_ARRAY <<< "$dtypes"
read -ra DECODE_ARRAY <<< "$decode_strategies"

# 验证数组长度一致性
if [[ ${#TASKS_ARRAY[@]} != ${#NSHOTS_ARRAY[@]} || ${#TASKS_ARRAY[@]} != ${#LENGTH_ARRAY[@]} || 
      ${#TASKS_ARRAY[@]} != ${#TEMP_ARRAY[@]} || ${#TASKS_ARRAY[@]} != ${#LIMITS_ARRAY[@]} || 
      ${#TASKS_ARRAY[@]} != ${#DTYPES_ARRAY[@]} ]]; then
    echo "Error: Arrays have different lengths!"
    echo "Tasks: ${#TASKS_ARRAY[@]}, Shots: ${#NSHOTS_ARRAY[@]}, Lengths: ${#LENGTH_ARRAY[@]}, Temps: ${#TEMP_ARRAY[@]}, Limits: ${#LIMITS_ARRAY[@]}, Dtypes: ${#DTYPES_ARRAY[@]}"
    exit 1
fi

export HF_ALLOW_CODE_EVAL=1
# CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 accelerate launch --main_process_port 29510 --num_processes 7 eval_inline_diffusion.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=512,diffusion_steps=512,temperature=0.2,top_p=0.95,add_bos_token=true,escape_until=true,dtype=${DTYPES_ARRAY[0]},save_dir=evals_results_instruct/humaneval-ns0-512-dtype${DTYPES_ARRAY[0]}-limit${LIMITS_ARRAY[0]}-temp0.2 \
#     --tasks humaneval \
#     --num_fewshot 0 \
#     --batch_size 1 \
#     --limit ${LIMITS_ARRAY[0]} \
#     --output_path evals_results_instruct/humaneval-ns0-512-dtype${DTYPES_ARRAY[0]}-limit${LIMITS_ARRAY[0]}-temp0.2 \
#     --log_samples \
#     --confirm_run_unsafe_code 
# NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}

# Iterate through the arrays
for i in "${!TASKS_ARRAY[@]}"; do
    for decode_strategy in "${DECODE_ARRAY[@]}"; do
        output_path=evals_results_instruct/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-len${LENGTH_ARRAY[$i]}-temp${TEMP_ARRAY[$i]}-limit${LIMITS_ARRAY[$i]}-diffsteps${LENGTH_ARRAY[$i]}-dtype${DTYPES_ARRAY[$i]}-topp09-${decode_strategy}
        save_dir=$output_path
        echo "Task: ${TASKS_ARRAY[$i]}, Decode: ${decode_strategy}, Shots: ${NSHOTS_ARRAY[$i]}, Length: ${LENGTH_ARRAY[$i]}, Temperature: ${TEMP_ARRAY[$i]}, Limit: ${LIMITS_ARRAY[$i]}, Dtype: ${DTYPES_ARRAY[$i]}; Output: $output_path"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 eval_inline_diffusion.py --model dream \
            --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},top_p=0.95,dtype=${DTYPES_ARRAY[$i]},save_dir=${save_dir},decode_strategy=${decode_strategy} \
            --tasks ${TASKS_ARRAY[$i]} \
            --num_fewshot ${NSHOTS_ARRAY[$i]} \
            --batch_size 1 \
            --output_path $output_path \
            --log_samples \
            --limit ${LIMITS_ARRAY[$i]} \
            --confirm_run_unsafe_code
    done
done
# 
