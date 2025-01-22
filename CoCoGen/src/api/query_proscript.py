"""
Given a prompt file and path to a task file with the following fields:

1. input_prompt_code: the code used to prompt codex
2. reference_code: expected completed code
3. reference_graph: expected graph

Runs inference over codex for each input_prompt_code, and adds the following fields to the output file:

4. generated_code: generated code
5. generated_graph: generated graph

The file can contain other metadata, but the fields above are required.
"""
from datetime import datetime
import shutil
import time
import openai
import pandas as pd
from tqdm import tqdm
import logging
import os
import random
from llama import Llama
from converters.get_converter import ConverterFactory
from converters.graph_code_converter import GraphPythonConverter
from src.prompting.constants import END, SEPARATOR
from src.api.openai_api_wrapper import openai_chat_api

logging.basicConfig(level=logging.INFO)


def construct_single_instruction(few_shot_prompt):
    '''
        Uses the few shot prompt to construct a single instruction
    '''
    conversation = []
    for index, message in enumerate(few_shot_prompt):
        
        # add system prompt and an empty message
        if message["role"] == "system" and index == 0:
            conversation.append(message)
            conversation.append({"role": "user", "content": ""})

        # add user prompt to the empty message
        elif message["role"] == "user":
            conversation[-1]['content'] += 'Input: {}\n'.format(message['content'].strip())
        else:
            conversation[-1]['content'] += 'Output: {}\n\n'.format(message['content'].strip())

    conversation[-1]['content'] += 'Output:'

    return conversation


def run(task_file_path: str,
        num_tasks: int,
        output_file_path: str,
        prompt_path: str,
        keep_writing_output: bool,
        engine: str,
        max_tokens:int, 
        max_requests_per_min: int,
        temperature: float,
        random_state: int,
        num_train: int,
        llama: bool,
        llama_ckpt_dir: str,
        llama_tokenizer_path: str,
        llama_seq_len: int):
    

    # # opening the json as string io
    # with open(task_file_path, "r") as f:
    #     task_file = f.read()
    # task_file = io.StringIO(task_file)

    tasks = pd.read_json(task_file_path, orient='records', lines=True)
    converter = ConverterFactory.get_converter(args.job_type)
    if num_tasks != -1:
        tasks = tasks.head(num_tasks)

    fixed_prompt_text = read_prompt(prompt_path)

    results = []

    cache = load_cache(output_file_path)

    num_requests = 0
    time_begin = time.time()

    # initialize the llama generator if needed
    if llama:
        generator = Llama.build(
            ckpt_dir=llama_ckpt_dir,
            tokenizer_path=llama_tokenizer_path,
            max_seq_len=llama_seq_len,
            max_batch_size=1,
        )

    for task_idx, task in tqdm(tasks.iterrows(), total=len(tasks)):

        is_success = False
        for cut_prompt_examples in [None, 1, 2, 3, 4, 5, 6]:

            
            num_requests += 1

            task_results = run_task(task=task, fixed_prompt_text=fixed_prompt_text,
                                    cache=cache, converter=converter, cut_prompt_examples=cut_prompt_examples, task_idx=task_idx,
                                    engine=engine, max_tokens=max_tokens, temperature=temperature, random_state=random_state, num_train=num_train,llama_generator=generator if llama else None)
            results.append(task_results)
            is_success = True
            try:
                break
            except openai.error.InvalidRequestError as e:
                
                logging.info(
                    f"InvalidRequestError: {e}, trying with shorter prompt (cut_prompt_examples={cut_prompt_examples + 1 if cut_prompt_examples is not None else 1})")
                # sleep for a bit to further avoid rate limit exceeded exceptions
                time.sleep(5)
                continue
            except Exception as e:  # something else went wrong
                print(e)
                logging.info(f"Task {task_idx} failed: {e}")
                break

        if is_success and keep_writing_output:
            if not llama:
                pd.DataFrame(results).to_json(
                    output_file_path, orient='records', lines=True)
            elif int(os.environ.get("LOCAL_RANK", 0)) == 0:
                pd.DataFrame(results).to_json(
                    output_file_path, orient='records', lines=True)

    print(
        f"Ran {len(results)} out of {len(tasks)} tasks ({len(results) / len(tasks):.2%})")
    pd.DataFrame(results).to_json( 
        output_file_path, orient='records', lines=True)


def run_task(task: dict, fixed_prompt_text: str, cache: dict, converter: GraphPythonConverter, task_idx: int, engine: str, max_tokens: int, cut_prompt_examples: int = None, temperature: float = 0.0, random_state: int = 0, num_train: int = 7, llama_generator=None) -> dict:
    """Runs the task, and returns the results.

    Args:
        task (dict): The task input
        fixed_prompt_text (str): Used for cases where the input prompt is fixed
        cache (dict): cache of previous results
        converter (GraphPythonConverter): A graph-python converter to parse results
        cut_prompt_examples (int, optional): If provided, the first `cut_prompt_examples` examples are 
                                             deleted. Prevents 4096 errors. Defaults to None.

    Returns:
        dict: A dictionary with the results.
    """
    start_time = time.time()
    
    prompt_text = fixed_prompt_text if fixed_prompt_text is not None else task['prompt']

    if cut_prompt_examples is not None:
        prompt_text_parts = prompt_text.split(END)
        prompt_text = END.join(prompt_text_parts[cut_prompt_examples:])

    # processing the prompt_text
    prompt_text_parts = prompt_text.split(END)

    # randomly sample few examples from the prompt
    random.seed(random_state)
    random.shuffle(prompt_text_parts)
    prompt_text_parts = prompt_text_parts[:num_train]

    prompt_io_text_parts = [t.split(SEPARATOR) for t in prompt_text_parts]
    few_shot_prompt = [{'role': 'system', 'content': 'You are an AI language model created by OpenAI. You will provide information and answer questions to the best of your knowledge and abilities.'}]
    for prompt_io_text_part in prompt_io_text_parts:
        if len(prompt_io_text_part) == 2:
            few_shot_prompt.append({'role': 'user', 'content': prompt_io_text_part[0]})
            few_shot_prompt.append({'role': 'assistant', 'content': prompt_io_text_part[0] + prompt_io_text_part[1]})

    # creating the prompt
    few_shot_prompt.append({'role': 'user', 'content': task['input_prompt_code']})

    # getting the response
    if llama_generator is None:
        response = openai_chat_api(few_shot_prompt, temperature=temperature, engine=engine, max_tokens=max_tokens)

        if response is None:
            completed_code = task['input_prompt_code']
        else:
            completed_code = response.choices[0].message.content

    else:
        results = llama_generator.text_completion(
            [construct_single_instruction(few_shot_prompt)[-1]['content']],
            max_gen_len=max_tokens,
            temperature=temperature,
            top_p=0.95,
        )[0]
        completed_code = results['generation'].split(END)[0]

    graph = converter.python_to_graph(completed_code)

    task_results = {k: v for (k, v) in task.items()}
    task_results["codex_response"] = completed_code
    task_results["generated_code"] = completed_code

    task_results["generated_graph"] = graph
    task_results["elapsed_time"] = time.time() - start_time

    return task_results


def read_prompt(prompt_path):
    if prompt_path is None:
        return None
    with open(prompt_path, "r") as f:
        prompt = f.read()
    return prompt


def load_cache(output_file_path: str):
    """We don't want to query codex repeatedly for the same input. If an output file exists, this
    function creates a "cache" of the results.
    The cache is implemented as a hashmap keyed by `input_prompt_code`, and maps to the 
    entire output entry

    Args:
        output_file_path (str): _description_
    """
    if not os.path.exists(output_file_path):
        return {}
    else:
        # make a backup of the file already there
        shutil.copyfile(output_file_path, output_file_path + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.copy(output_file_path, output_file_path + ".bak")
        cache_data = pd.read_json(
            output_file_path, orient='records', lines=True)
        cache = {row['input_prompt_code']: row.to_dict()
                 for _, row in cache_data.iterrows()}
        return cache


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file_path", type=str, default='data/proscript_script_generation/dev.jsonl')
    parser.add_argument("--num_tasks", type=int, default=-1)
    parser.add_argument("--output_file_path", type=str, default='data/proscript_script_generation/dev_test.jsonl')
    parser.add_argument("--prompt_path", type=str,
                        required=False, default='data/proscript_script_generation/prompt.txt')
    parser.add_argument("--job_type", type=str,
                        choices=ConverterFactory.supported_converters, default='proscript-literal')
    parser.add_argument("--keep_writing_output",
                        action="store_true", default=True)
    parser.add_argument("--engine", type=str, default='gpt-35-turbo')
    parser.add_argument("--max_requests_per_min", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=15)
    parser.add_argument('--llama', action='store_true', default=False)
    parser.add_argument('--llama_ckpt_dir', type=str, default='/data/shared_model/code_llama/CodeLlama-13b-Python')
    parser.add_argument('--llama_tokenizer_path', type=str, default='/data/shared_model/code_llama/CodeLlama-13b-Python/tokenizer.model')
    parser.add_argument('--llama_seq_len', type=int, default=7000)
    args = parser.parse_args()

    run(task_file_path=args.task_file_path, num_tasks=args.num_tasks, output_file_path=args.output_file_path, prompt_path=args.prompt_path, keep_writing_output=args.keep_writing_output, engine=args.engine, max_requests_per_min=args.max_requests_per_min, max_tokens=args.max_tokens, temperature=args.temperature, random_state=args.random_state, num_train=args.train_size, llama=args.llama, llama_ckpt_dir=args.llama_ckpt_dir, llama_tokenizer_path=args.llama_tokenizer_path, llama_seq_len=args.llama_seq_len)


