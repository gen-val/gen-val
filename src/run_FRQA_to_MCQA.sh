#!/bin/bash
#SBATCH --job-name=run_frqa_to_mcqa  # job name
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=16           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=00:59:59              # total run time limit (HH:MM:SS)

python freeformqa_to_mcqa.py --st_mdl=gpt-3.5-turbo-1106 \
    --num_hotpotqa 100 \
    --num_multispanqa 100 \
    --num_24game 100 \
    --num_gsm8k 100 \
    --num_mmlu 100 \
    --num_collie 100 \
    --num_csqa 100 \
    --num_hellaswag 100 \
    --num_race 100 \
    --num_piqa 100
    