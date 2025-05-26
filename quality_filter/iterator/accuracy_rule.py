
import enum
import json
import dataclasses
from typing import List, Dict

import ahocorasick

from .rule import BaseRule, DynamicRuleConfig, ModelRes
from quality_filter.util.files import input_data_to_str


def get_unsafe_words(file_path_list: List[str]) -> List[str]:
    """ https://github.com/DataEval/dingo/blob/main/dingo/model/rule/utils/util.py#L65 """
    
    unsafe_words_list = []
    for file_path in file_path_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                j = json.loads(line)
                word = str(j['word'])
                unsafe_words_list.append(word)
    return unsafe_words_list


class DetectModelType(str, enum.Enum):
    RULE_BASED = "RULE_BASED"
    MODEL_DETECT_DETOXIFY = "MODEL_DETECT_DETOXIFY"


class DetoxifyProcess(BaseRule):
    """check toxic"""
    def __init__(self, rules_dict: Dict[str | int, str] | List[str], detect_model: DetectModelType | str):
        super().__init__()
        self.process_name = "detoxify"
        self.rules_dict = rules_dict
        self.detect_model = detect_model if isinstance(detect_model, DetectModelType) else DetectModelType(detect_model)


    def __process__(self, input_data) -> ModelRes:
        input_data = input_data_to_str(input_data)
        
        if self.detect_model in [DetectModelType.RULE_BASED]:
            if self.rules_dict is None:
                raise ValueError("rules_dict must be provided for RULE_BASED model.")
            elif isinstance(self.rules_dict, dict):
                rules_dict = self.rules_dict
                key_list = list(rules_dict.values())
            else:
                key_list = self.rules_dict
                
            # 检测
            automaton = ahocorasick.Automaton()
            for index, key in enumerate(key_list):
                automaton.add_word(key, (index, key))
            automaton.make_automaton()
            matches = [(end_index - len(value[1]) + 1, value[1]) for end_index, value in automaton.iter(input_data)]
            
            # 返回
            res = ModelRes()
            if matches:
                res.error_status = True
                res.type = DetectModelType.RULE_BASED.value
                res.name = self.__class__.__name__,
                res.value = len(matches),
                res.reason = [value for index, value in matches]
            else:
                res.error_status = False
                res.type = DetectModelType.RULE_BASED.value
                res.name = self.__class__.__name__,
                res.value = 0.0,
                res.reason = []
            return res
            
        elif self.detect_model in [DetectModelType.MODEL_DETECT_DETOXIFY]:
            # 分类器模型检测示例
            return ModelRes(
                error_status=False,
                type=DetectModelType.MODEL_DETECT_DETOXIFY.value,
                name=self.__class__.__name__,
                value=0.0,
                reason=[]
            )