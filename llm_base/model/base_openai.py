import json
import time
from typing import Dict, List

from llm_base.config.config import DynamicLLMConfig
from llm_base.base.base_data import MetaData
from llm_base.base.base_llm import BaseLLM
# from dingo.model.modelres import ModelRes
from llm_base.base.base_res import ResponseScoreReason
from llm_base.base.base_prompt import BasePrompt
# from dingo.model.response.response_class import ResponseScoreReason
# from dingo.utils import log
# from dingo.utils.exception import ConvertJsonError, ExceedMaxTokens
from pydantic import ValidationError


class BaseOpenAI(BaseLLM):
    prompt: BasePrompt = BasePrompt
    client = None
    dynamic_config = DynamicLLMConfig()

    @classmethod
    def set_prompt(cls, prompt: BasePrompt):
        cls.prompt = prompt

    @classmethod
    def create_client(cls):
        from openai import OpenAI

        if not cls.dynamic_config.key:
            raise ValueError("key cannot be empty in llm config.")
        elif not cls.dynamic_config.api_url:
            raise ValueError("api_url cannot be empty in llm config.")
        else:
            cls.client = OpenAI(api_key=cls.dynamic_config.key, base_url=cls.dynamic_config.api_url)

    @classmethod
    def build_messages(cls, input_data: MetaData) -> List:
        prompt = input_data.prompt if input_data.prompt is not None else cls.prompt.content
        messages = [
            {"role": "system", "content": cls.prompt.system_prompt},
            {"role": "user",
             "content": prompt + input_data.content + cls.prompt.output_format}
        ]
        return messages

    @classmethod
    def send_messages(cls, messages: List):
        if cls.dynamic_config.model:
            model_name = cls.dynamic_config.model
        else:
            model_name = cls.client.models.list().data[0].id

        params = cls.dynamic_config.parameters
        cls.validate_config(params)

        completions = cls.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=params.get('temperature', 0.3) if params else 0.3,
            top_p=params.get('top_p', 1) if params else 1,
            max_tokens=params.get('max_tokens', 4000) if params else 4000,
            presence_penalty=params.get('presence_penalty', 0) if params else 0,
            frequency_penalty=params.get('frequency_penalty', 0) if params else 0
        )

        if completions.choices[0].finish_reason == "length":
            print(f"Exceed max tokens: {params.get('max_tokens', 4000) if params else 4000}")

        return str(completions.choices[0].message.content)

    @classmethod
    def validate_numeric_range(cls, value, min_val, max_val, param_name):
        if not isinstance(value, (int, float)):
            raise ValueError(f"{param_name} must be a number")
        if not (min_val <= value <= max_val):
            raise ValueError(f"{param_name} must between {min_val} and {max_val}")

    @classmethod
    def validate_integer_positive(cls, value, param_name):
        if not isinstance(value, int):
            raise ValueError(f"{param_name} must be an integer")
        if value <= 0:
            raise ValueError(f"{param_name} must be greater than 0")

    @classmethod
    def validate_config(cls, parameters: Dict):
        if parameters is None:
            return

        # validate temperature
        if "temperature" in parameters:
            cls.validate_numeric_range(parameters["temperature"], 0, 2, "temperature")

        # validate top_p
        if "top_p" in parameters:
            cls.validate_numeric_range(parameters["top_p"], 0, 1, "top_p")

        # validate max_tokens
        if "max_tokens" in parameters:
            cls.validate_integer_positive(parameters["max_tokens"], "max_tokens")

        # validate presence_penalty
        if "presence_penalty" in parameters:
            cls.validate_numeric_range(parameters["presence_penalty"], -2.0, 2.0, "presence_penalty")

        # validate frequency_penalty
        if "frequency_penalty" in parameters:
            cls.validate_numeric_range(parameters["frequency_penalty"], -2.0, 2.0, "frequency_penalty")

    @classmethod
    def process_response(cls, response: str) -> ResponseScoreReason:
        print(f"response: {response}")
        response_model = ResponseScoreReason()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        try:
            response_model.state = True
            response_json = json.loads(response)
            response_model.response_json = response_json
        except json.JSONDecodeError:
            print(f'Convert to JSON format failed: {response}')

        

        # error_status
        # if response_model.score == 1:
        #     result.reason = [response_model.reason]
        # else:
        #     result.error_status = True
        #     result.type = cls.prompt.metric_type
        #     result.name = cls.prompt.__name__
        #     result.reason = [response_model.reason]

        return response_model

    @classmethod
    def eval(cls, input_data: MetaData) -> ResponseScoreReason:
        if cls.client is None:
            cls.create_client()

        messages = cls.build_messages(input_data)

        attempts = 3
        turn = 1
        # except_msg = ''
        # except_name = Exception.__class__.__name__
        response_model = ResponseScoreReason()
        while attempts >= turn:
            # try:
            response = cls.send_messages(messages)
            
            response_model = cls.process_response(response)
            if response_model.state:
                print(f"success response for  {input_data.content}")
                return response_model
            print(f"At most {attempts} times, failed {turn} time. fail response for {input_data.content}")
            turn += 1
        
        return response_model
        #     except (ValidationError, ExceedMaxTokens, ConvertJsonError) as e:
        #         except_msg = str(e)
        #         except_name = e.__class__.__name__
        #         break
        #     except Exception as e:
        #         attempts += 1
        #         time.sleep(1)
        #         except_msg = str(e)
        #         except_name = e.__class__.__name__

        # return ModelRes(
        #     error_status=True,
        #     type='QUALITY_BAD',
        #     name=except_name,
        #     reason=[except_msg]
        # )
