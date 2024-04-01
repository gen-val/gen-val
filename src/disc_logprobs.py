import argparse
import os
import json
import numpy as np
from datetime import date
from freeformqa_to_mcqa import init_logger, init_gpt

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)

def _disc_logprobs(args, data, sys_prompt_suffix=None):
    """Run discriminator evaluations under logprobs setting over the single dataset in data. 
       A system prompt suffix can be added to the system prompt, default None."""
    for d in data:
        if 'opts' not in d: continue # occasionally a data example might not have any associated options due to pass-through data
        assert len(args['opts']) == len(d['opts']), "Number of options must match"
        q_concat = d['question']
        tf_scores = {'TRUE': [], 'FALSE': []}  # Scores for true/false across all options

        # Iterate through each MC option
        for i, _ in enumerate(args['opts']):
            q_with_opt = q_concat + f"\nIs the following option correct?\nOption: {d['opts'][i]}"
            msgs = [{"role": "system", "content": f"Answer `TRUE` or `FALSE`: Is the option provided the correct answer to the question? {sys_prompt_suffix}"}, 
                    {"role": "user", "content": q_with_opt}]
            logprobs, tokens = args["gpt"].gen_top20(msgs, tokens=1)
            top20gens = dict(zip(tokens, logprobs))
            
            # Collect scores specifically for TRUE and FALSE responses
            for tf in ['TRUE', 'FALSE']:
                if tf in top20gens:
                    tf_scores[tf].append(top20gens[tf])
                else:
                    tf_scores[tf].append(0)

        d.update({"optgens": tf_scores})
        # Compute softmax scores across TRUE and FALSE for each option
        t_scores, f_scores = tf_scores['TRUE'], tf_scores['FALSE']
        for i in range(len(t_scores)):
            tf_scores['TRUE'][i], tf_scores['FALSE'][i] = softmax(np.array([t_scores[i], f_scores[i]])).tolist()
        
        # Calculate discriminator scores
        gt_scores = [] # list of scores for correctly labelled TRUE/FALSE for each option
        for i in range(len(args['opts'])):
            correct_tf = 'TRUE' if i == d['idx_gt'] else 'FALSE'
            gt_scores.append(tf_scores[correct_tf][i])
        
        # Update question dict with discriminator scores
        d.update({
            "top20gens": top20gens,
            "optsoftmax": gt_scores,
            "score": np.mean(gt_scores)  # Mean score across all options
        })
        args["logger"].info(f"Question: {d['question']} \n Option logprobs: {tf_scores} \n Score: {np.mean(gt_scores):.2%}")

    return data


def disc_logprobs(args):
    """ run discriminator evaluations under logprobs setting over all datasets specified in args """
    args = init_gpt(init_logger(args))
    dir_data = args['dir_data']
    scores = []

    for fn in os.listdir(dir_data):
        if fn.endswith("_mcqa.json") and args[fn.split('_')[0].lower()]:
            fn_in = os.path.join(dir_data, fn)
            fn_out = os.path.join(args["dir_res"], f"{fn.split('_')[0]}_disc_logprobs_{args['st_mdl']}.json") # output file name
            with open(fn_in, 'r') as f:
                data = json.load(f)
            
            data = _disc_logprobs(args, data)
            score = np.mean([q['score'] for q in data])
            scores.append(score)
            
            # Update JSON file with additional keys
            with open(fn_out, 'w') as f:
                json.dump(data, f, indent=4)
            
            args["logger"].info(f"Processed {fn.split('_')[0]} with mean softmax score: {score:.2%}")

    # Print overall model score
    overall_score = np.mean(scores)
    print(f"Overall model score: {overall_score:.2%}")

def main():
    consts = {
        "st_task": "disc_logprobs",  # name of task
        "dir_log": "../logs", # where to output logs
        "dir_data": "../data", # where to output data
        "dir_res": "../results", # where to output results
        "st_today": date.today().strftime("%Y_%m_%d"), # today's date
        "opts": ["A", "B", "C", "D"] # multiple choice options
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--st_mdl", type=str, help='model string (e.g. "gpt-3.5-turbo-1106" or "gpt-4-1106-preview")')
    parser.add_argument("--hotpotqa", action='store_true', help='Run on hotpotQA dataset, default is not running')
    parser.add_argument("--multispanqa", action='store_true', help='Run on multispanqa dataset, default is not running')
    parser.add_argument("--24game", action='store_true', help='[Run on 24game dataset, default is not running')
    parser.add_argument("--gsm8k", action='store_true', help='Run on gsm8k dataset, default is not running')
    parser.add_argument("--mmlu", action='store_true', help='Run on mmlu dataset, default is not running')
    parser.add_argument("--collie", action='store_true', help='Run on collie dataset, default is not running')
    parser.add_argument("--csqa", action='store_true', help='Run on csqa dataset, default is not running')
    parser.add_argument("--hellaswag", action='store_true', help='Run on hellaswag dataset, default is not running')
    parser.add_argument("--race", action='store_true', help='Run on race dataset, default is not running')
    parser.add_argument("--piqa", action='store_true', help='Run on piqa dataset, default is not running')
    args = vars(parser.parse_args())
    disc_logprobs({**args, **consts})


if __name__ == "__main__":
    main()