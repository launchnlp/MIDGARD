import time
from typing import Dict, Any, List
from openai import OpenAI

from src.prompting.constants import END, END_LINE

'''
    OpenAI client
'''
openai_client = OpenAI()


# code: "code-davinci-001"

class OpenaiAPIWrapper:
    @staticmethod
    def call(prompt: str, max_tokens: int, engine: str) -> dict:
        response = openai_client.chat.completions.create(
            model=engine,
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=[END, END_LINE],
            # logprobs=3,
            best_of=1
        )
        return response

    @staticmethod
    def parse_response(response) -> Dict[str, Any]:
        text = response["choices"][0]["text"]
        return text


'''
    Adding Inder's Code for calling openai executions
'''
def openai_chat_api(
    messages: List[Dict[str, str]],
    engine: str = 'gpt-35-turbo',
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
            response = openai_client.chat.completions.create(
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
