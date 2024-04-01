import argparse
import dill
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import sys
from datetime import date
from dotenv import load_dotenv
from typing import Any, Dict, List
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.utils import GPT
from utils.game24_solver import solve

def init_logger(args: dict):
    """ init logger """
    ## create file name and create its directories if they don't exist
    st_fn = f"{args['st_today']}_{args['st_task']}_{args['st_mdl']}"
    os.makedirs(args["dir_log"], exist_ok=True)

    ## Create file handler and set formatter
    file_handler = logging.FileHandler(f"{args['dir_log']}/{st_fn}.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    ## Create a logger instance, config logging, and add the file handler
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.addHandler(file_handler)
    
    args["logger"] = logger
    return args

def init_gpt(args: dict):
    """ init GPT and return new set of args with GPT object """
    # load api key from .env and create GPT object
    load_dotenv() 
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gpt = GPT(openai_api_key, logger=args["logger"], model=args["st_mdl"])
    args["gpt"] = gpt
    return args

def gen_options(args, data, sys_prompt_suffix = "", num_questions=10):
    # convert dataset from free-choice QA(FRQA) to multiple choice QA (MCQA). Data is list[dict{"answer": str, "question": str, "context": str}].
    # Returns list of dicts with added columns "opts" and "idx_gold": MCQA options and ground truth index(0-3), respectively
    indices = random.sample(range(len(data)), num_questions)
    mcqa_data = []
    for idx in indices:
        item = data[idx]
        gold, question, context = str(item['answer']), item['question'], []
        if 'context' in item: context = item['context']
        question_with_context = ""
        for i in range(len(context)):
            context_str = ''.join(context[i][1])
            question_with_context += f"Paragraph {i+1}: {context[i][0]} {context_str} "
        question_with_context += f"\nQuestion: {question}"
        opts = args["gpt"].get_mc_options(question_with_context, gold, sys_prompt_suffix=sys_prompt_suffix)
        opts = list(opts.values())
        while len(opts) < 3:
            opts_sup = args["gpt"].get_mc_options(question_with_context, gold, sys_prompt_suffix=sys_prompt_suffix) # supplementary options, occasionally we get less than 3 options here
            opts = opts.extend(list(opts_sup.values())[:3-len(opts)]) # append supplementary options to opts
        mcqa_data.append(item)
        idx_gt = np.random.randint(0, 4) # choose a random option to place correct answer under (among a, b, c, d)
        mcqa_data[-1]['idx'], mcqa_data[-1]['idx_gt'] = idx, idx_gt
        mcqa_data[-1]['opts'] = opts[:idx_gt] + [gold] + opts[idx_gt:]
        args["logger"].info(f"Generated options: {mcqa_data[-1]['opts']}")
    return mcqa_data

def hotpotqa(args: dict):
    """ Get data from hotpotqa dataset and convert to standardized MCQA """
    fn_data = '../hotpotqa/data/hotpotqa_dev_distractor.json' # filename
    with open(fn_data, 'r') as f: data = json.load(f) # keys in each row: ['_id', 'answer', 'question', 'supporting_facts', 'context', 'type', 'level']

    data = gen_options(args, data, num_questions=min(len(data), args["num_hotpotqa"]))
    with open("../data/hotpotqa_mcqa.json", 'w') as f: json.dump(data, f, indent=4)
    logging.info("hotpotqa_mcqa.json saved")

def multispanqa(args: dict):
    """ Get data from multispanqa dataset and convert to standardized MCQA """
    fn_data = '../multispanqa/data/train.json' # filename
    with open(fn_data, 'r') as f: data = json.load(f)['data'] # keys in each row: ['_id', 'answer', 'question', 'supporting_facts', 'context', 'type', 'level']
        
    # dataset-specific adjustments (we eventually want a list of dicts of format [{"id": "xx", "question": "xxx", "context": "xxx"}, ...]
    for d in data:
        num_span, ctxt = d['num_span'], d['context']
        span = ""; ans = [] # init
        # dataset is in IOB tagging
        for iob_idx, iob in enumerate(d['label']):
            if iob == 'B': span = ctxt[iob_idx]; ans.append(span)
            elif iob == 'I': span += ' ' + ctxt[iob_idx]; ans[-1] = span
            elif iob == 'O' and len(ans) >= num_span: break
        d['answer'], d['context'], d['question'] = ans, [["context", [' '.join(ctxt)]]], ' '.join(d['question'])

    data = gen_options(args, data, sys_prompt_suffix="Each answer should be the string form of a list of values, and all strings should come from the contextual paragraphs", num_questions=min(len(data), args["num_multispanqa"]))
    with open("../data/multispanqa_mcqa.json", 'w') as f: json.dump(data, f, indent=4)
    logging.info("multispanqa_mcqa.json saved")

def game24(args: dict):
    """ Get data from 24 Game dataset and convert to standardized MCQA """
    fn_data = '../24game/data/24game.csv' # filename
    data = pd.read_csv(fn_data).to_dict(orient='records')
    for d in data: d['question'] = d['Puzzles']
    for d in data: 
        digits = d['Puzzles'].split()
        sol = solve(digits)
        try:
            if round(eval(sol), 1) == 24:
                d['answer'] = sol
            else:
                d['answer'] = None
        except:
            d['answer'] = None
            print("question:", d['Puzzles'], "\n answer:", sol)
        for item in data:
            nums = item["Puzzles"].split()  # Split the question string into numbers
            item["question"] = f"The following is an instance of the 24 game. Move around the numbers and use arithmetic operations (+, -, /, *) to form the number 24 using the numbers ({nums[0]}, {nums[1]}, {nums[2]}, {nums[3]}). You must use each number exactly once."

    data = gen_options(args, data, num_questions=min(len(data), args["num_24game"]))
    with open("../data/24game_mcqa.json", 'w') as f: json.dump(data, f, indent=4)
    logging.info("24game_mcqa.json saved")

def gsm8k(args: dict):
    """ Get data from GSM8K dataset and convert to standardized MCQA """
    fn_data = '../gsm8k/data/test.jsonl' # filename
    data = []
    with open(fn_data, 'r') as files:
        for f in files: data.append(json.loads(f))
    for d in data:
        d['answer'] = d['answer'].split('\n#### ')[-1]

    data = gen_options(args, data, num_questions=min(len(data), args["num_gsm8k"]))
    with open("../data/gsm8k_mcqa.json", 'w') as f: json.dump(data, f, indent=4)
    logging.info("gsm8k_mcqa.json saved")

def mmlu(args: dict):
    """ Get data from MMLU dataset and convert to standardized MCQA """
    dir_data = '../mmlu/data/test' # data dir
    data = []
    subjects = [fn[:-9] for fn in os.listdir(dir_data) if fn.endswith('_test.csv')]
    for subj in subjects:
        subj_data = pd.read_csv(f"{dir_data}/{subj}_test.csv", header=None).to_numpy()
        subj_out = []
        for i, e in enumerate(subj_data):
            idx_gt = ord(e[5]) - ord('A')
            opts = list(e[1:5])
            subj_out.append({'subj': subj, 'idx': i, 'question': e[0], 'answer': opts[idx_gt], 'idx_gt': idx_gt, 'opts': opts}) 
        data.extend(subj_out)

    data = random.sample(data, min(len(data), args["num_mmlu"]))
    with open("../data/mmlu_mcqa.json", 'w') as f: json.dump(data, f, indent=4)
    logging.info("mmlu_mcqa.json saved")

def collie(args: dict):
    """ Get data from Collie dataset and convert to standardized MCQA """
    fn_data = '../collie/data/all_data.dill' # filename
    with open(fn_data, "rb") as f:
        data: Dict[str, List[Dict[str, Any]]] = dill.load(f)
    # process data
    DN_TASK_MAP = {
        'c01': 'word01',
        'c02': 'word02',
        'c03': 'word03',
        'c04': 'sent01',
        'c05': 'sent02',
        'c06a': 'sent03',
        'c07': 'sent04',
        'c08': 'para01',
        'c09': 'para02',
        'c10': 'para03',
        'c11': 'para04',
        'c12': 'para05',
        'c14': 'pass01'
    }
    def rename_dict_keys(original_dict, key_mapping):
        """
            Rename the keys of a dictionary based on a key_mapping dictionary. E.g.
            odict: {'text_c0': ['test'], 'text_c1': ['test']}
            kmap: {'c0': 'word_1'} 
            rename_dict_keys(odict, kmap): {'word_1': ['test'], 'test_c1': ['test']}

            original_dict: dictionary whose keys should be mapped
            key_mapping: mapping from old (including substrings of) keys to new ones
        """
        new_dict = {}
        for k, v in original_dict.items():
            # set new key = old key for now; we don't know if an actual new key exists
            nk = k
            for old_key, new_key in key_mapping.items():
                if old_key in k:
                    nk = new_key
            if nk not in new_dict:
                new_dict[nk] = v.copy()
            else:
                new_dict[nk].extend(v.copy())
        return new_dict
    data_names = list(data.keys())
    data_names = sorted(data_names, key=lambda x: (x.split('_')[1], x.split('_')[0]))
    # data containing only keys (dataset of origin name), and relevant constraint/
    # prompt info (no text corpus)
    d_no_text = {}
    for dn in data_names:
        # d_no_text[dn] = [{'c': ex['constraint'], 'question': ex['prompt'], \
        #                 't': ex['targets'], 'answer': ex['example']} for ex in data[dn]]
        d_no_text[dn] = [{'question': ex['prompt'], 'answer': ex['example']} for ex in data[dn]]
    data = rename_dict_keys(d_no_text, DN_TASK_MAP)
    data_names = list(data.keys())
    
    # generate options
    out_data = []
    num_q_per_cat = [len(data[dn]) for dn in data_names] # number of questions per category
    num_q_over = np.sum(num_q_per_cat) - args["num_collie"] # number of questions we're over the question count by
    while num_q_over > 0:
        num_q_per_cat[np.argmax(num_q_per_cat)] -= 1
        num_q_over -= 1
    for dn, num in zip(data_names, num_q_per_cat):
        subj_data = gen_options(args, data[dn], num_questions=num)
        for e in subj_data: 
            e['subj'] = dn
        subj_data = [d for d in subj_data if 'opts' in d] # only keep data with options
        out_data.extend(subj_data)
    with open("../data/collie_mcqa.json", 'w') as f: json.dump(out_data, f, indent=4)
    logging.info("collie_mcqa.json saved")

def csqa(args: dict):
    """ Get data from CSQA dataset and convert to standardized MCQA """
    dir_data = '../csqa/data'
    data = []
    for dir_cat in os.listdir(dir_data): # category dir
        path_cat = os.path.join(dir_data, dir_cat)
        if os.path.isdir(path_cat):
            for fn in os.listdir(path_cat):
                if fn.endswith('.json'):
                    fp = os.path.join(path_cat, fn)
                    with open(fp, 'r') as f:
                        json_data = json.load(f)
                    if len(json_data) >= 2:
                        data.append([json_data[0], json_data[1]])

    formatted_data = []
    for idx, task in enumerate(data):
        if task[0]['speaker'] == 'USER' and task[1]['speaker'] == 'SYSTEM':
            q = task[0]['utterance']
            a = task[1]['utterance']
            formatted_data.append({'_id': idx, 'question': q, 'answer': a})
            
    data = formatted_data

    data = gen_options(args, data, num_questions=min(len(data), args["num_csqa"]))
    with open("../data/csqa_mcqa.json", 'w') as f: json.dump(data, f, indent=4)
    logging.info("csqa_mcqa.json saved")

def hellaswag(args: dict):
    """ Get data from Hellaswag dataset and convert to standardized MCQA """
    fn_data = '../hellaswag/data/hellaswag_val.jsonl' # filename
    data = []
    with open(fn_data, 'r') as files:
        for f in files:
            data.append(json.loads(f))
    data = [{'question': d['ctx'], 'opts': d['endings'], 'answer': d['endings'][int(d['label'])], 'idx_gt': d['label']} for d in data]

    # data = gen_options(args, data, num_questions=min(len(data), args["num_hellaswag"]))
    data = random.sample(data, min(len(data), args["num_hellaswag"]))
    with open("../data/hellaswag_mcqa.json", 'w') as f: json.dump(data, f, indent=4)
    logging.info("hellaswag_mcqa.json saved")

def race(args: dict):
    """ Get data from RACE dataset and convert to standardized MCQA """
    dir_data = '../race/data/high' # data directory
    data = []
    for fn in os.listdir(dir_data):
        if fn.endswith('.txt'):
            fp = os.path.join(dir_data, fn)
            with open(fp, 'r') as f:
                task = [json.loads(l) for l in f]
                assert len(task) == 1
                data.append(task[0])
    data = [{'question': d['questions'][0], 'context': d['article'], 'opts': d['options'][0], 'idx_gt': ord(d['answers'][0])-ord('A')} for d in data]

    # data = gen_options(args, data, num_questions=min(len(data), args["num_race"]))
    data = random.sample(data, min(len(data), args["num_race"]))
    with open("../data/race_mcqa.json", 'w') as f: json.dump(data, f, indent=4)
    logging.info("race_mcqa.json saved")

def piqa(args: dict):
    """ Get data from PIQA dataset and convert to standardized MCQA """
    dir_data = '../piqa/data' # data directory
    data = []
    with open(f"{dir_data}/valid.jsonl", 'r') as files:
        for f in files:
            data.append(json.loads(f))
    with open(f"{dir_data}/valid-labels.lst", 'r') as f:
        ans = [a.strip() for a in f.readlines()]
    data = [{'question': d['goal'], 'options': [d['sol1'], d['sol2']], 'answer': [d['sol1'], d['sol2']][int(a)]} for d, a in zip(data, ans)]

    data = gen_options(args, data, num_questions=min(len(data), args["num_piqa"]))
    with open("../data/piqa_mcqa.json", 'w') as f: json.dump(data, f, indent=4)
    logging.info("piqa_mcqa.json saved")

def freeformqa_to_mcqa(args: dict):
    """
        Given a free-form question and answer, generate a multiple choice question with 3 incorrect options.
    """
    assert isinstance(args, dict), "param args must be of type dict" 
    args = init_gpt(init_logger(args))

    if args["num_hotpotqa"] > 0: hotpotqa(args)
    if args["num_multispanqa"] > 0: multispanqa(args)
    if args["num_24game"] > 0: game24(args)
    if args["num_gsm8k"] > 0: gsm8k(args)
    if args["num_mmlu"] > 0: mmlu(args)
    if args["num_collie"] > 0: collie(args)
    if args["num_csqa"] > 0: csqa(args)
    if args["num_hellaswag"] > 0: hellaswag(args)
    if args["num_race"] > 0: race(args)
    if args["num_piqa"] > 0: piqa(args)



def main():
    consts = {
        "st_task": "freeformqa_to_mcqa",  # name of task
        "dir_log": "../logs", # where to output logs
        "st_today": date.today().strftime("%Y_%m_%d") # today's date
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--st_mdl", type=str, help='model string (e.g. "gpt-3.5-turbo-1106" or "gpt-4-1106-preview")')
    parser.add_argument("--num_hotpotqa", type=int, default=0, help='Number of randomly sampled tasks from hotpotQA dataset to convert from FRQA to MCQA, default 0')
    parser.add_argument("--num_multispanqa", type=int, default=0, help='Number of randomly sampled tasks from multispanQA dataset to convert from FRQA to MCQA, default 0')
    parser.add_argument("--num_24game", type=int, default=0, help='Number of randomly sampled tasks from 24 Game dataset to convert from FRQA to MCQA, default 0')
    parser.add_argument("--num_gsm8k", type=int, default=0, help='Number of randomly sampled tasks from GSM8K dataset to convert from FRQA to MCQA, default 0')
    parser.add_argument("--num_mmlu", type=int, default=0, help='Number of randomly sampled tasks from MMLU dataset to convert from FRQA to MCQA, default 0')
    parser.add_argument("--num_collie", type=int, default=0, help='Number of randomly sampled tasks from COLLIE dataset to convert from FRQA to MCQA, default 0')
    parser.add_argument("--num_csqa", type=int, default=0, help='Number of randomly sampled tasks from CSQA dataset to convert from FRQA to MCQA, default 0')
    parser.add_argument("--num_hellaswag", type=int, default=0, help='Number of randomly sampled tasks from Hellawag dataset to convert from FRQA to MCQA, default 0')
    parser.add_argument("--num_race", type=int, default=0, help='Number of randomly sampled tasks from RACE dataset to convert from FRQA to MCQA, default 0')
    parser.add_argument("--num_piqa", type=int, default=0, help='Number of randomly sampled tasks from PIQA dataset to convert from FRQA to MCQA, default 0')
    

    args = vars(parser.parse_args())
    freeformqa_to_mcqa({**args, **consts})


if __name__ == "__main__":
    main()