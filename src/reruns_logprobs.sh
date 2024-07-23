#!/bin/bash
#SBATCH --job-name=reruns  # job name
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=16           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=00:59:59              # total run time limit (HH:MM:SS)

python disc_logprobs.py --st_mdl=gpt-3.5-turbo-1106 \
    --gsm8k \
    
python disc_logprobs.py --st_mdl=gpt-4-1106-preview \
    --gsm8k \

python mcqa_gen_logprobs.py --st_mdl=gpt-3.5-turbo-1106 \
    --gsm8k \
    
python mcqa_gen_logprobs.py --st_mdl=gpt-4-1106-preview \
    --gsm8k \
