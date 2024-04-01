import logging
import os
from openai import OpenAI, APIError

# Class for generating Chat Completions from GPT
class GPT:
    def __init__(self, api_key, logger, model="gpt-3.5-turbo-1106"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cost = 0  # To keep track of the cost
        if self.model == "gpt-3.5-turbo":
            self.ppk = 0.0010 * 1e-3  # $ price per token

        self.logger = logger

    def gen(self, msgs, functions=None, function_call=None, tokens=100):
        args = {
            "model": self.model,
            "messages": msgs,
            "tools": functions,
            "tool_choice": function_call,
            "max_tokens": tokens
        }
        if functions is None:
            del args["functions"]
            del args["function_call"]
        try:
            response = self.client.chat.completions.create(**args)
            # Update cost
            return response.choices[0].message
        except APIError as e:
            self.logger.error(f"Error: {e}")
            return None
        
    def gen_top20(self, msgs, functions=None, function_call=None, tokens=1):
        """
            Generate top 20 next-token completions by log probs in the form of a 2-tuple
            of lists (log probs, generations)
        """
        args = {
            "model": self.model,
            "messages": msgs,
            "tools": functions,
            "tool_choice": function_call,
            "max_tokens": tokens,
            "logprobs": True,
            "top_logprobs": 20,
        }
        if functions is None:
            del args["tools"]
            del args["tool_choice"]
        try:
            response = self.client.chat.completions.create(**args)
            # Update cost
            log_prob_obj = response.choices[0].logprobs.content[0].top_logprobs
            logprobs = [e.logprob for e in log_prob_obj]
            tokens = [e.token for e in log_prob_obj]
            return logprobs, tokens
        except APIError as e:
            self.logger.error(f"Error: {e}")
            return None, None
    
    def get_mc_options(self, question, answer, sys_prompt_suffix=None):
        """
            Given a multiple choice question, question, and ground truth answer, answer, (as strings) return GPT-generated multiple choice
            options in form dict{"a": str, "b": str, "c": str, "d":str} where answer is randomly assigned to one of the 4 options. An 
            optional string sys_prompt_suffix can be provided to append to the system prompt.
        """
        fns = [{"type":"function",
                "function": {
                "name": "return_list",
                "description": "Return the list of options",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "opt1": {"type": "string", "description": "Incorrect option 1"},
                        "opt2": {"type": "string", "description": "Incorrect option 2"},
                        "opt3": {"type": "string", "description": "Incorrect option 3"},
                    },
                    "required": ["opt1", "opt2", "opt3"]
                }
            }
        }]
        msgs = [{"role": "system", "content": f"I want to create a multiple choice question from a free-form question and answer. Generate 3 incorrect options given the correct answer. {sys_prompt_suffix}"}, 
               {"role": "user", "content": f"{question}\nAnswer: {answer}"}]
        args = {
            "model": self.model,
            "messages": msgs,
            "tools": fns,
            "tool_choice": {"type": "function", "function": {"name": "return_list"}},
        }
        try:
            res = self.client.chat.completions
            # Update cost
            opts = eval(res.create(**args).choices[0].message.tool_calls[0].function.arguments)
            return opts[:3]
        except APIError as e:
            self.logger.error(f"Error: {e}")
            return None
        
def config_logger(dir, st_fn):
    """
        directory dir and file name (as string) st_fn
    """
    ## Configure the logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    ## Create a file handler and set the formatter
    file_handler = logging.FileHandler(os.path.join(dir, "results", f"{st_fn}.log"))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    ## Create a logger instance and add the file handler
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)
    return logger