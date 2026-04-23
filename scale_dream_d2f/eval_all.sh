cd /mnt/rl/xinyi/LoPA/scale_dream_d2f
bash /mnt/rl/xinyi/LoPA/scale_dream_d2f/eval_dream_block.sh
bash /mnt/rl/xinyi/LoPA/scale_dream_d2f/eval_dream_lopa.sh

cd /mnt/rl/xinyi/LoPA/baselines/dllm-cache
bash /mnt/rl/xinyi/LoPA/baselines/dllm-cache/eval_dream_instruct.sh


cd /mnt/rl/xinyi/LoPA/baselines/dParallel-master/Dream/eval_instruct
bash /mnt/rl/xinyi/LoPA/baselines/dParallel-master/Dream/eval_instruct/eval_dream_dp_my.sh

cd /mnt/rl/xinyi/LoPA/baselines/klass_my/src/
bash /mnt/rl/xinyi/LoPA/baselines/klass_my/src/eval_dream_my.sh






