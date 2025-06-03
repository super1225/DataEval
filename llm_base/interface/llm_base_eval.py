import json
from typing import Dict, Optional
from llm_base.base.base_prompt import BasePrompt
from llm_base.model.base_openai import BaseOpenAI
from llm_base.config.config import DynamicLLMConfig
from llm_base.base.base_data import MetaData
from llm_base.base.base_res import ResponseScoreReason
from quality_filter.iterator.base import JsonIterator

class LLMBaseEval(JsonIterator):
    def __init__(self, user_prompt: str = None, system_prompt: str = None, output_format: str = None, api_key: str = None, api_url: str = None, model: str = None, parameters: dict = None, local_prompt: BasePrompt = None):
        '''
        初始化llmbase评估器
        '''
        # 若是api_key,api_url,model为空，则抛出一个异常
        if api_key is None or api_url is None or model is None:
            raise ValueError("api_key, api_url and model cannot be empty.")
        if local_prompt is not None:
            self.prompt = local_prompt
        else:
            self.prompt = BasePrompt()
            if user_prompt is None:
                self.prompt.content = user_prompt
            if system_prompt is None:
                self.prompt.system_prompt = system_prompt
            if output_format is None:
                self.prompt.output_format = output_format
        
        BaseOpenAI.dynamic_config = DynamicLLMConfig(
            key="sk-e8686abb857d4f91ac9e154a567e2baa",
            api_url="https://api.deepseek.com",
            model="deepseek-chat",
            parameters=parameters
        )
        self.llm = BaseOpenAI



    def __process__(self, input_data: dict) -> Optional[ResponseScoreReason]:
        '''
        处理输入数据
        '''
        response = ResponseScoreReason(state=False, response_json=None)
        if type(input_data) is str:
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {input_data}")
                return response
        elif input_data is None:
            return None
        meta_data = MetaData(**input_data)
        response = self.llm.eval(meta_data)
        return response
