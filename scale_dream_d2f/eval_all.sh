cd /home/chenkai/data/LoPA/scale_dream_d2f
bash /home/chenkai/data/LoPA/scale_dream_d2f/eval_dream_block.sh
bash /home/chenkai/data/LoPA/scale_dream_d2f/eval_dream_lopa.sh

cd /home/chenkai/data/LoPA/baselines/dllm-cache
bash /home/chenkai/data/LoPA/baselines/dllm-cache/eval_dream_instruct.sh


cd /home/chenkai/data/LoPA/baselines/dParallel-master/Dream/eval_instruct
bash /home/chenkai/data/LoPA/baselines/dParallel-master/Dream/eval_instruct/eval_dream_dp_my.sh

# cd /home/chenkai/data/LoPA/baselines/klass_my/src/
# bash /home/chenkai/data/LoPA/baselines/klass_my/src/eval_dream_my.sh






