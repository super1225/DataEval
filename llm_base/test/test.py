from llm_base.config.config import DynamicLLMConfig
from llm_base.base.base_data import MetaData
from llm_base.model.base_openai import BaseOpenAI


# 现在你可以导入本地包中的模块了
def llm():
    data = MetaData(
        data_id='123',
        # prompt="hello, introduce the world", 无效，请使用 BasePrompt
        content="Hello! The world is a vast and diverse place, full of wonders, cultures, and incredible natural beauty."
    )

    BaseOpenAI.dynamic_config = DynamicLLMConfig(
        key="sk-e8686abb857d4f91ac9e154a567e2baa",
        api_url="https://api.deepseek.com",
        model="deepseek-chat",
    )

    res = BaseOpenAI.eval(data)
    # print(res.response_json)

llm()