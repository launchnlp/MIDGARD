import os
import random
import json
import time
import openai
import copy
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List, Dict
from data_processor import sample_essay_dataset, sample_abstract_dataset, sample_cdcp_dataset
from prompt_scheme import code_prompt_nx, data_to_io, few_shot_chat_prompt

'''
    openai credentials
'''

# create a global client
client = openai.Client()

def openai_chat_api(
    messages: List[Dict[str, str]],
    engine: str = 'gpt-4',
    temperature: float = 0.0,
    max_tokens: int = 2000,
    top_p: float = 0.95,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    stop: List[str] = None,
    num_retries: int = 5
):
    '''
        Calls open ai chat api
    '''

    for _ in range(num_retries):
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
            return response
        except Exception as e:
            print(e)
            print('Retrying call to openai chat api')
            time.sleep(5)

    return None

def infer_structure(
    few_shot_prompt: List[Dict[str, str]],
    test_input: List[str],
    temperature: float = 0.0,
):
    '''
        Runs openai chat api for test_data
    '''

    response_list = []
    for input_str in tqdm(test_input):
        input_message = few_shot_prompt + [{'role': 'user', 'content': input_str}]
        tic = time.time()
        response = openai_chat_api(input_message, temperature=temperature)
        toc = time.time()
        if response is not None:
            try:
                response_list.append({
                    'content': response.choices[0].message.content,
                    'usage': response.usage,
                    'time': toc - tic
                })
            except Exception as e:
                print('Error in response:', e)
                response_list.append({
                    'content': '',
                    'usage': response.usage,
                    'time': toc - tic
                })

        else:
            response_list.append({
                'content': None,
                'usage': None,
                'time': None
            })
    return response_list

def sample_from_few_shot_prompt(
    few_shot_prompt: List[Dict[str, str]],
    sample_length: int = 15
):
    '''
        Sampling from the few shot prompts as it may have a lot of examples
    '''

    # sampling indices
    num_datapoints = len(few_shot_prompt) // 2
    effective_sample_length = min(sample_length, num_datapoints)

    if effective_sample_length == num_datapoints:
        return copy.deepcopy(few_shot_prompt)

    sampled_indices = random.sample(range(num_datapoints), effective_sample_length)

    # constructing the sampled few shot prompt
    new_few_shot_prompt = []
    new_few_shot_prompt.append(copy.deepcopy(few_shot_prompt[0]))
    for i in sampled_indices:
        new_few_shot_prompt.append(copy.deepcopy(few_shot_prompt[2 * i + 1]))
        new_few_shot_prompt.append(copy.deepcopy(few_shot_prompt[2 * i + 2]))

    return new_few_shot_prompt

def executor(
    train_size=3,
    test_size=50,
    random_state=42,
    data='essay',
    data_dir='ArgumentAnnotatedEssays-2.0/brat-project-final/',
    train_test_split_csv='ArgumentAnnotatedEssays-2.0/train-test-split.csv',
    system_prompt_file='system_prompt.txt',
    output_dir='simple_prompt/',
    prompt='simple_prompt',
    engine='gpt-35-turbo',
    force_component_identification=False,
    remove_identity=False,
    remove_edge_prediction=False,
    sorting='reading_order',
    add_cot=False,
    sample_multiple_outputs=False,
    num_samples=50,
    sample_temperature=0.9,
    sampling_shot=3,
    no_openai=False,
):
    '''
        Main executor for the project
    '''
    
    # creating a directory to store the output
    os.makedirs(output_dir, exist_ok=True)

    # reading system prompt file
    with open(system_prompt_file, 'r') as f:
        system_prompt = f.read()

    # creating a config file to store config details
    config = {
        'train_size': train_size,
        'test_size': test_size,
        'random_state': random_state,
        'data_dir': data_dir,
        'train_test_split_csv': train_test_split_csv,
        'system_prompt': system_prompt,
        'prompt': prompt,
        'engine': engine,
        'force_component_identification': force_component_identification,
        'remove_identity': remove_identity,
        'remove_edge_prediction': remove_edge_prediction,
        'data': data
    }

    # sampling the essay dataset
    if data == 'essay':
        print('Sampling essay dataset')
        train_data, test_data = sample_essay_dataset(
            train_test_split_csv=train_test_split_csv,
            data_dir=data_dir,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
            sorting=sorting
        )
    elif data == 'abstract':
        print('Sampling abstract dataset')
        train_data, test_data = sample_abstract_dataset(
            data_dir=data_dir,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
            sorting=sorting
        )
    elif data == 'cdcp':
        print('Sampling cdcp dataset')
        train_data, test_data = sample_cdcp_dataset(
            data_dir=data_dir,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state
        )

    # setting the correct test_size
    config['test_size'] = len(test_data)

    if prompt == 'simple_prompt':
        raise NotImplementedError('Simple prompt not implemented')
        '''
        prompt_fn = lambda x: simple_prompt(x, node_identity = not remove_identity)
        component_splitter = None
        '''
    elif prompt == 'code_prompt':
        prompt_fn = lambda x: code_prompt_nx(x, node_identity = not remove_identity, edge_prediction = not remove_edge_prediction, add_cot=add_cot)
        component_splitter = '# edge declarations' if force_component_identification else None
    else:
        raise NotImplementedError('Prompt {} not implemented'.format(prompt))
    train_input, train_output = data_to_io(train_data, prompt_fn, component_splitter)
    test_input, test_output = data_to_io(test_data, prompt_fn, component_splitter)

    # creating few shot prompt and saving it to config
    few_shot_prompt = few_shot_chat_prompt(train_input, train_output, system_prompt)
    config['few_shot_prompt'] = few_shot_prompt
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # running openai chat api
    if not sample_multiple_outputs:

        # running openai chat api
        if not no_openai:
            response_list = infer_structure(few_shot_prompt, test_input)

            # saving the response list
            for index, (test_i, test_o, response) in enumerate(zip(test_input, test_output, response_list)):
                result_dict = {
                    'input': test_i,
                    'output': test_o,
                    'inference': response['content']
                }
                with open(os.path.join(output_dir, 'result_{}.json'.format(index)), 'w') as f:
                    json.dump(result_dict, f, indent=4)
        
        # not running openai chat api
        else:
            for index, (test_i, test_o) in enumerate(zip(test_input, test_output)):
                result_dict = {
                    'input': test_i,
                    'output': test_o,
                    'full_prompt': few_shot_prompt + [{'role': 'user', 'content': test_i}],
                    'inference': '',
                    'usage': ''
                }
                with open(os.path.join(output_dir, 'result_{}.json'.format(index)), 'w') as f:
                    json.dump(result_dict, f, indent=4)
            
    else:
        # sampling multiple outputs
        for sample_index in range(num_samples):

            # creating a new directory
            sample_output_dir = os.path.join(output_dir, 'sample_{}'.format(sample_index))
            os.makedirs(sample_output_dir, exist_ok=True)

            # saving the config
            with open(os.path.join(sample_output_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)

            # sampling the outputs
            current_few_shot_prompt = sample_from_few_shot_prompt(few_shot_prompt, sample_length=sampling_shot)

            # running openai chat api
            if not no_openai:
                response_list = infer_structure(current_few_shot_prompt, test_input, temperature=sample_temperature)

                # saving the response list
                for index, (test_i, test_o, response) in enumerate(zip(test_input, test_output, response_list)):
                    result_dict = {
                        'input': test_i,
                        'output': test_o,
                        'inference': response['content']
                    }
                    with open(os.path.join(sample_output_dir, 'result_{}.json'.format(index)), 'w') as f:
                        json.dump(result_dict, f, indent=4)
            
            # not running openai chat api
            else:
                for index, (test_i, test_o) in enumerate(zip(test_input, test_output)):
                    result_dict = {
                        'input': test_i,
                        'output': test_o,
                        'full_prompt': current_few_shot_prompt + [{'role': 'user', 'content': test_i}],
                        'inference': '',
                        'usage': ''
                    }
                    with open(os.path.join(sample_output_dir, 'result_{}.json'.format(index)), 'w') as f:
                        json.dump(result_dict, f, indent=4)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_size', type=int, default=3)
    parser.add_argument('--test_size', type=int, default=50)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='data/ArgumentAnnotatedEssays-2.0/brat-project-final/')
    parser.add_argument('--train_test_split_csv', type=str, default='data/ArgumentAnnotatedEssays-2.0/train-test-split.csv')
    parser.add_argument('--system_prompt_file', type=str, default='default_prompt.txt')
    parser.add_argument('--output_dir', type=str, default='data/ArgumentAnnotatedEssays-2.0/outputs/code_prompt_3_50_42/')
    parser.add_argument('--prompt', type=str, default='code_prompt')
    parser.add_argument('--engine', type=str, default='gpt-35-turbo')
    parser.add_argument('--force_component_identification', action='store_true')
    parser.add_argument('--remove_identity', action='store_true')
    parser.add_argument('--remove_edge_prediction', action='store_true')
    parser.add_argument('--data', type=str, default='essay')
    parser.add_argument('--sorting', type=str, default='reading_order')
    parser.add_argument('--add_cot', action='store_true')
    parser.add_argument('--sample_multiple_outputs', action='store_true')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--sample_temperature', type=float, default=0.9)
    parser.add_argument('--sampling_shot', type=int, default=3)
    parser.add_argument('--no_openai', action='store_true')
    args = parser.parse_args()

    # running the executor
    executor(
        train_size=args.train_size,
        test_size=args.test_size,
        random_state=args.random_state,
        data=args.data,
        data_dir=args.data_dir,
        train_test_split_csv=args.train_test_split_csv,
        system_prompt_file=args.system_prompt_file,
        output_dir=args.output_dir,
        prompt=args.prompt,
        engine=args.engine,
        force_component_identification=args.force_component_identification,
        remove_identity=args.remove_identity,
        remove_edge_prediction=args.remove_edge_prediction,
        sorting=args.sorting,
        add_cot=args.add_cot,
        sample_multiple_outputs=args.sample_multiple_outputs,
        num_samples=args.num_samples,
        sample_temperature=args.sample_temperature,
        sampling_shot=args.sampling_shot,
        no_openai=args.no_openai,
    )
