from .rule import BaseRule,ModelRes
from typing import Dict

class CheckStructureCompleteness(BaseRule):
    def __init__(self):
        super().__init__()
    def __process__(self,data_list, sample_ratio=5):
        if not data_list or not all(isinstance(item, dict) for item in data_list):
            raise ValueError("数据必须是一个包含字典的列表")
        sample_size=int(len(data_list)*sample_ratio)
        # 采样获取模板字段集合
        sample_data = data_list[:sample_size]
        template_fields = set()
        for item in sample_data:
            template_fields.update(item.keys())

        # 判定完整性
        results = []
        for idx, item in enumerate(data_list):
            item_fields = set(item.keys())
            is_complete = item_fields == template_fields
            results.append(is_complete)
        structure_completeness_ratio=sum(results)/len(data_list)
        structure_completeness_ratio_ = ModelRes()
        structure_completeness_ratio_.value = structure_completeness_ratio
        return structure_completeness_ratio_
