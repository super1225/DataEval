from typing import Protocol

from llm_base.base.base_data import MetaData
from llm_base.base.base_res import ResponseScoreReason
from llm_base.base.base_prompt import BasePrompt


class BaseLLM(Protocol):
    @classmethod
    def set_prompt(cls, prompt: BasePrompt):
        ...

    @classmethod
    def eval(cls, input_data: MetaData) -> ResponseScoreReason:
        ...
