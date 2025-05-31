from .rule import BaseRule,ModelRes
import re
from typing import Dict

class ValidateFormat(BaseRule):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def __process__(self, data) -> Dict[str, ModelRes]:
        """
        通用格式校验函数，根据传入正则表达式校验数据格式有效性
        忽略 None 和空字符串。
        返回：无效行数，有效行数，总行数，无效比率
        """
        regex = re.compile(self.pattern)
        valid_count = 0
        invalid_count = 0
        filtered_data = [x for x in data if x is not None and x.strip() != ""]

        for item in filtered_data:
            if regex.fullmatch(item.strip()):
                valid_count += 1
            else:
                invalid_count += 1

        total = len(filtered_data)
        invalid_ratio = invalid_count / total if total > 0 else 0.0

        invalid_count_ = ModelRes()
        invalid_count_.value = invalid_count
        valid_count_ = ModelRes()
        valid_count_.value = valid_count
        total_ = ModelRes()
        total_.value = total
        invalid_ratio_ = ModelRes()
        invalid_ratio_.value = round(invalid_ratio, 4)

        return {
            "invalid_count": invalid_count_,
            "valid_count": valid_count_,
            "total": total_,
            "invalid_ratio": invalid_ratio_
        }


class ValidateEmail(ValidateFormat):
    def __init__(self):
        self.pattern = r'^[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(.[a-zA-Z0-9_-]+)+$'
        super().__init__(self.pattern)


class ValidateIDCard(ValidateFormat):
    def __init__(self):
        self.pattern = (r'(^[1-9]\d{5}(18|19|20)\d{2}'
        r'((0[1-9])|(10|11|12))'
        r'(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]$)'
        r'|'
        r'(^[1-9]\d{5}\d{2}'
        r'((0[1-9])|(10|11|12))'
        r'(([0-2][1-9])|10|20|30|31)\d{3}$)')
        super().__init__(self.pattern)


class ValidateIPAddress(ValidateFormat):
    def __init__(self):
        self.pattern = (r'^(?:(?:1[0-9][0-9].)|(?:2[0-4][0-9].)|(?:25[0-5].)|'
                        r'(?:[1-9][0-9].)|(?:[0-9].)){3}'
                        r'(?:(?:1[0-9][0-9])|(?:2[0-4][0-9])|(?:25[0-5])|'
                        r'(?:[1-9][0-9])|(?:[0-9]))$')
        super().__init__(self.pattern)


class ValidatePhone(ValidateFormat):
    def __init__(self):
        self.pattern = r'^1[3-9]\d{9}$'
        super().__init__(self.pattern)


class ValidatePostcode(ValidateFormat):
    def __init__(self):
        self.pattern = r'^[1-9]\d{5}$'
        super().__init__(self.pattern)


class ValidateDate(ValidateFormat):
    def __init__(self):
        self.pattern = r'^[1-9]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])$'
        super().__init__(self.pattern)


class FormatValidatorManager(BaseRule):
    """FormatValidatorManager: Select and apply a specific format validator based on rule name."""

    VALIDATOR_MAP = {
        'email': ValidateEmail,
        'idcard': ValidateIDCard,
        'ip': ValidateIPAddress,
        'phone': ValidatePhone,
        'postcode': ValidatePostcode,
        'date': ValidateDate
    }

    def __init__(self, validator_name: str):
        super().__init__()
        if validator_name not in self.VALIDATOR_MAP:
            raise ValueError(f"Unsupported validator name: {validator_name}")
        self.validator_instance = self.VALIDATOR_MAP[validator_name]()

    def __process__(self, data_list) -> Dict[str, ModelRes]:
        """
        Validate a list of strings using the selected validator.
        """
        return self.validator_instance.__process__(data_list)