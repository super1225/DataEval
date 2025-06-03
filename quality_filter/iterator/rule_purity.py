from ftfy import is_bad
from .rule import BaseRule,ModelRes
from typing import Dict

class CheckNoisyData(BaseRule):
    def __init__(self):
        super().__init__()

    def __process__(self, input_data) -> ModelRes:
        # check_null_values(data: Union[Dict, List], path: List[str]) -> Dict[Any, Any]:
        """
        检查空值：空字符串或None视为空值。
        返回：空值行数，总行数，空值率
        """
        if not input_data:
            noisy_data_ratio_ = ModelRes()
            noisy_data_ratio_.value = 0
            return noisy_data_ratio_

        total = len(input_data)
        noisy_data_count = sum(1 for s in input_data if is_bad(s))
        noisy_data_ratio = noisy_data_count / total
        noisy_data_ratio_ = ModelRes()
        noisy_data_ratio_.value = round(noisy_data_ratio, 4)
        return noisy_data_ratio_



def garbled_ratio(strings):
    """
    计算字符串列表中含有乱码（由编码错误导致的）字符串的比例。

    参数:
        strings (list of str): 字符串列表

    返回:
        float: 乱码比例，范围为 0.0 ~ 1.0
    """


    return garbled_count / total