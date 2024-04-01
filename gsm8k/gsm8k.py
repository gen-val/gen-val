import argparse
import json
import logging
import os
import pandas as pd
import random
import re
import sys
from datetime import date
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.utils import GPT

# Global vars
gpt = None
logger = None
data = None
st_fn = None

def load_dataset(task):
    """
        Load dataset for string task
    """
    # Load into a dictionary
    data = []
    if task == "gsm8k":
        with open('data/test.jsonl', 'r') as files:
            for f in files:
                data.append(json.loads(f)) # keys in each row: ['_id', 'answer', 'question', 'supporting_facts', 'context', 'type', 'level']
    return data

def compare_strings(str1, str2):
    """
    This function compares two strings after converting them to lowercase and stripping leading and trailing spaces.
    """
    return str1.lower().strip() == str(str2).lower().strip()

def compare_ints(str1, str2):
    """
        Tries to check if numbers represented by str1 and str2 are equal, where str2 is a refernce answer known to be an int
    """
    try:
        return eval(str1.replace("```", "").lower().strip()) == eval(str2)
    except:
        return False

def check_bool(str1, bool2):
    """
        Checks if string str1 equal to bool bool2
    """
    try:
        return bool(str1) == bool2
    except:
        return False


def parse_bool(str):
    """
        Parse bool from discriminator prediction, starting from the end. E.g. if str was "...FALSE.." parse_bool(str)
        would return False
    """
    idx_true = str.rfind("TRUE") # idx of TRUE
    idx_false = str.rfind("FALSE") # idx of FALSE

    # Check which appears first
    if idx_true > idx_false:
        return True
    elif idx_true < idx_false:
        return False
    else:
        return None



def construct_message(agents, question, idx):
    """
        Based on Multiagent Debate from https://github.com/composable-models/llm_multiagent_debate
    """
    # Use introspection when there are no other agents.
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

    prefix_string = "These are the recent/updated opinions from other agents: "

    for ag in agents:
        agent_response = ag[idx]["content"]
        prefix_string += f"\n\n One agent response: ```{agent_response}```"

    prefix_string = prefix_string + f"\n\n Use these opinions carefully as additional advice, can you provide an updated answer to {question}? Make sure to state your answer at the end of the response."
    return {"role": "user", "content": prefix_string}

def mad_gen(task, num_agents=2, rounds=3):
    """
        Given a task task, generate an answer using multiagent debate using num_agents agents in rounds rounds    
    """
    logger.info("~~~Multiagent debate starts~~~")
    agent_ctxs = [[{"role": "user", "content": f"{task} Make sure to state your answer at the end of the response."}] for _ in range(num_agents)]
    for round in range(rounds):
        for i, agent_ctx in enumerate(agent_ctxs):
            if round != 0:
                agent_ctxs_other = agent_ctxs[:i] + agent_ctxs[i+1:]
                message = construct_message(agent_ctxs_other, task, 2*round - 1)
                agent_ctx.append(message)
            completion = gpt.gen(msgs=agent_ctx, tokens=750).content
            assistant_message = {"role": "assistant", "content": completion}
            agent_ctx.append(assistant_message)
            logger.info(assistant_message)

    return agent_ctxs

def most_frequent(lst):
    freqs = {}
    for e in lst:
        if e in freqs:
            freqs[e] += 1
        else:
            freqs[e] = 1
    
    most_frequent = max(freqs, key=freqs.get)
    return most_frequent, freqs[most_frequent]

def extract_answer(ctxs):
    agent_answers = []
    for ctx in ctxs:
        pred = ctx[-1]['content']
        # Use regex to match last number (either int or decimal)
        matches = re.findall(r'\b\d*\.?\d+\b', pred)
        if matches:
            agent_answers.append(float(matches[-1]))
    return agent_answers, most_frequent(agent_answers)

def run_mad(num_agents=2, rounds=3):
    """
        rounds is number of turns before final answer
    """
    mad_gens = {} # dictionary of generations
    predictions = [] # list of predictions, reset after each set of runs
    indices = [] # question numbers
    answers = [] # list of gold answers
    evals = [] # whether prediction matches answer
    questions = [] # list of questions
    all_preds = [] # list of (list of predictions from all agents from each trial)


    # Evaluate
    num_questions = 200 # number of questions to run on
    indices = random.sample(range(len(data)), num_questions)
    for idx in indices:
        item = data[idx]
        gold, question = item['answer'], item['question']
        gold = gold.split('\n#### ')[-1] # extract final numerical answer from gold
        logger.info(question)
        generation = mad_gen(question, num_agents=num_agents, rounds=rounds)
        agent_preds, (pred, freq) = extract_answer(generation) # agent predictions list, final prediction, number of agents supporting final prediction
        logger.info(f"number of agents with selected answer: {freq}")
        mad_gens[idx] = generation
        
        predictions.append(pred)
        answers.append(gold)
        evals.append(compare_ints(predictions[-1], str(gold)))
        questions.append(question)
        all_preds.append(agent_preds)
        print(predictions[-1])

    # Print the predictions and answers
    logger.info(predictions)
    # logger.info(answers)
    percent_correct = (sum(evals) / len(evals)) * 100

    print(f"Percentage of True values: {percent_correct}%")

    # create a dataframe with the data
    df = pd.DataFrame({'Index': indices, 'Question': questions, 'Agent Predictions': all_preds, 'Prediction': predictions, 'Answer': answers, 'Eval': evals})
    # df = pd.DataFrame({'Index': indices, 'Question': questions, 'Prediction': predictions, 'Eval': evals})

    # save the datafrme to csv
    df.to_csv(f'results/{st_fn}_mad.csv', index=False)


def main():
    # run gsm8k on GPT-3.5: python gsm8k.py gpt-3.5-turbo-1106 gsm8k
    # run gsm8k on GPT-4: python gsm8k.py gpt-4-1106-preview gsm8k mad
    global gpt, logger, data, st_fn 

    parser = argparse.ArgumentParser()
    parser.add_argument('mdl', type=str, help='gpt-3.5-turbo-1106 or gpt-4-1106-preview')
    parser.add_argument('task', type=str, help='Task name, e.g. "gsm8k". Must have corresponding folder for data, results, etc.')
    parser.add_argument('mode', type=str, help='Mode, e.g. "mad" to run Multiagent debate. Options are "gen", "disc", "mad"')
    parser.add_argument('--sfx', type=str, default='', help='Suffix for run, added to end of output log and csv file names')
    args = parser.parse_args()

    # logging
    ## today's date
    st_today = date.today().strftime("%Y_%m_%d")

    ## file name:
    st_fn = f"{st_today}_{args.task}_n200_{args.mdl}{args.sfx}"

    ## Configure the logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    ## Create a file handler and set the formatter
    file_handler = logging.FileHandler(f"results/{st_fn}.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    ## Create a logger instance and add the file handler
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)

    # load api key from .env and create GPT object
    load_dotenv()  # Load environment variables from .env file
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gpt = GPT(openai_api_key, logger=logger, model=args.mdl)
    
    data = load_dataset(args.task)
    if args.mode == "mad":
        run_mad()

if __name__ == "__main__":
    main()

