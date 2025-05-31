from .rule import BaseRule,ModelRes
from typing import Dict

class CheckNullValues(BaseRule):
    def __init__(self):
        super().__init__()

    def __process__(self, input_data) -> Dict[str, ModelRes]:
        # check_null_values(data: Union[Dict, List], path: List[str]) -> Dict[Any, Any]:
        """
        检查空值：空字符串或None视为空值。
        返回：空值行数，总行数，空值率
        """
        data = input_data
        total = len(data)
        null_count = sum(1 for x in data if x is None or x.strip() == "")
        null_ratio = null_count / total if total > 0 else 0.0
        # return null_count, total, round(null_ratio, 4)
        null_count_ = ModelRes()
        null_count_.value = null_count
        total_ = ModelRes()
        total_.value = total
        null_ratio_ = ModelRes()
        null_ratio_.value = round(null_ratio, 4)
        return {"null_count": null_count_, "total": total_, "null_ratio": null_ratio_}


