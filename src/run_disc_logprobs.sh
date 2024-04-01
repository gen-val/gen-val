#!/bin/bash
#SBATCH --job-name=run_frqa_to_mcqa  # job name
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=16           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=00:59:59              # total run time limit (HH:MM:SS)

# python disc_logprobs.py --st_mdl=gpt-3.5-turbo-1106 \
python disc_logprobs.py --st_mdl=gpt-4-1106-preview \
    --hotpotqa \
    --multispanqa \
    --24game \
    --gsm8k \
    --mmlu \
    --collie \
    --csqa \
    --hellaswag \
    --race \
    --piqa

    
