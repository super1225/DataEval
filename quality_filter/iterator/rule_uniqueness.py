from .rule import BaseRule,ModelRes
from typing import Dict
from collections import defaultdict

class CheckUniqueValues(BaseRule):
    def __init__(self):
        super().__init__()

    def __process__(self, input_data) -> Dict[str, ModelRes]:
        # def check_unique_values(data: Union[Dict, List], ) -> Dict[Any, Any]:
        """
        检查唯一值，包含NULL（None/空字符串）作为唯一值计入。
        返回：唯一值行数，总行数，唯一值率
        """
        data = input_data
        seen = set()
        unique_count = 0
        for item in data:
            if item not in seen:
                seen.add(item)
                unique_count += 1
        total = len(data)
        unique_ratio = unique_count / total if total > 0 else 0.0
        # return unique_count, total, round(unique_ratio, 4)
        unique_count_ = ModelRes()
        unique_count_.value = unique_count
        total_ = ModelRes()
        total_.value = total
        unique_ratio_ = ModelRes()
        unique_ratio_.value = round(unique_ratio, 4)
        return {
            "unique_count": unique_count_,
            "total": total_,
            "unique_ratio": unique_ratio_
        }


class CheckDuplicateValues(BaseRule):
    def __init__(self):
        super().__init__()

    def __process__(self, input_data) -> Dict[str, ModelRes]:
        # def check_duplicate_values(data: Union[Dict, List], path: List[str]) -> Dict[Any, Any]:
        """
        检查重复值行数（不含NULL），重复值率。
        返回：重复值行数，总行数，重复值率
        """
        data = input_data
        value_counts = defaultdict(int)
        for item in data:
            if item is not None and item.strip() != "":
                value_counts[item] += 1

        duplicate_rows = sum(v for v in value_counts.values() if v > 1)
        total = len(data)
        #print(duplicate_rows)
        duplicate_ratio = duplicate_rows / total if total > 0 else 0.0
        # return duplicate_rows, total, round(duplicate_ratio, 4)
        duplicate_count_ = ModelRes()
        duplicate_count_.value = duplicate_rows
        total_ = ModelRes()
        total_.value = total
        duplicate_ratio_ = ModelRes()
        duplicate_ratio_.value = round(duplicate_ratio, 4)
        return {
            "duplicate_count": duplicate_count_,
            "total": total_,
            "duplicate_ratio": duplicate_ratio_,
        }

from collections import defaultdict

class UniquenessManager(BaseRule):
    """UniquenessManager: Select and apply uniqueness or repetition check."""

    CHECK_MAP = {
        'uniqueness': CheckUniqueValues,
        'duplicate': CheckDuplicateValues
    }

    def __init__(self, check_type: str):
        super().__init__()
        if check_type not in self.CHECK_MAP:
            raise ValueError(f"Unsupported check type: {check_type}. Use 'uniqueness' or 'repetition'.")
        self.check_instance = self.CHECK_MAP[check_type]()

    def __process__(self, input_data) -> Dict[str, ModelRes]:
        """
        Perform the selected uniqueness or repetition check.
        """
        return self.check_instance.__process__(input_data)