import argparse
import os
import json
import numpy as np
from datetime import date
from freeformqa_to_mcqa import init_logger, init_gpt

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)

def _gen_logprobs(args, data, sys_prompt_suffix=None):
    """ run generator evaluations under logprobs setting over the single dataset in data. A system prompt suffix can be added to the system prompt, default None """
    for d in data:
        if 'opts' not in d: continue # occasionally a data example might not have any associated options due to pass-through data
        assert len(args['opts']) == len(d['opts']), f"Number of options doesn't match for {args['opts']} and {d['opts']}"   
        q_concat = d['question']
        for i, opt in enumerate(args['opts']):
            q_concat += f"\n{opt}: {d['opts'][i]}"
        # Generate top 20 most likely answers
        msgs = [{"role": "system", "content": f"Generate a single token corresponding to the letter of the correct option for the following multiple choice question. {sys_prompt_suffix}"}, 
               {"role": "user", "content": q_concat}]
        logprobs, tokens = args["gpt"].gen_top20(msgs, tokens=1)
        top20gens = dict(zip(tokens, logprobs))       
        optgens = {tk: lp for tk, lp in top20gens.items() if tk in args['opts']} # Match with multiple choice options

        for opt in args['opts']: # Fill in missing options
            if opt not in optgens:
                optgens[opt] = 0
        
        # Compute softmax scores for the correct answer
        opt_softmaxes = softmax(np.array([optgens[opt] for opt in args['opts']])).tolist()
        gt_score = opt_softmaxes[d['idx_gt']] # softmax score for correct option
        
        # Update question dict
        d.update({
            "top20gens": top20gens,
            "optgens": optgens,
            "score": gt_score,
            "optsoftmax": opt_softmaxes
        })
        args["logger"].info(f"Question: {d['question']} \n Option logprobs: {optgens} \n Score: {gt_score:.2%}")
    return data

def gen_logprobs(args):
    """ run generator evaluations under logprobs setting over all datasets specified in args """
    args = init_gpt(init_logger(args))
    dir_data = args['dir_data']
    scores = []

    for fn in os.listdir(dir_data):
        if fn.endswith("_mcqa.json") and args[fn.split('_')[0].lower()]:
            fn_in = os.path.join(dir_data, fn)
            fn_out = os.path.join(args["dir_res"], f"{fn.split('_')[0]}_gen_logprobs_{args['st_mdl']}.json") # output file name
            with open(fn_in, 'r') as f:
                data = json.load(f)
            
            data = _gen_logprobs(args, data)
            print(fn.split('_')[0])
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
        "st_task": "gen_logprobs",  # name of task
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
    gen_logprobs({**args, **consts})


if __name__ == "__main__":
    main()