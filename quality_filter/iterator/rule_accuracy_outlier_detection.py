from .rule import BaseRule,ModelRes
from typing import Dict

class CheckOutlier(BaseRule):
    def __init__(self):
        super().__init__()

    def __process__(self, values) -> Dict[str, ModelRes]:
        # 总行数
        n = len(values)
        # if n == 0:
        #     return 0, 0, 0.0

        # 1. 计算均值（mean）
        total = 0.0
        for x in values:
            total += x
        mean = total / n

        # 2. 计算方差 var = (1/n) * sum((x - mean)^2)，然后开平方得到标准差
        sq_sum = 0.0
        for x in values:
            diff = x - mean
            sq_sum += diff * diff
        variance = sq_sum / n
        std = variance ** 0.5

        # 3. 计算阈值：3 倍标准差
        threshold = 3 * std

        # 4. 统计离群值：满足 |x - mean| > threshold
        outlier_count = 0
        for x in values:
            if abs(x - mean) > threshold:
                outlier_count += 1

        # 5. 计算离群值比例
        outlier_ratio = outlier_count / n
        outlier_count_ = ModelRes()
        outlier_count_.value = outlier_count
        total_count_= ModelRes()
        total_count_.value = n
        outlier_ratio_ = ModelRes()
        outlier_ratio_.value = round(outlier_ratio, 4)
        return {"outlier_count": outlier_count_, "total": total_count_, "outlier_ratio": outlier_ratio_}

class CheckValueRange(BaseRule):
    def __init__(self,valid_range):
        super().__init__()
        self.valid_range=valid_range
    def __process__(self, data) -> Dict[str, ModelRes]:

        invalid_count = 0
        total_count = len(data)

        for value in data:
            if isinstance(self.valid_range, (set, list)):
                if value not in self.valid_range:
                    invalid_count += 1
            elif isinstance(self.valid_range, tuple):
                if value<self.valid_range[0] or value>self.valid_range[1]:
                    invalid_count += 1
            elif callable(self.valid_range):  # 支持自定义函数判断
                if not self.valid_range(value):
                    invalid_count += 1

        valid_count = total_count - invalid_count
        invalid_rate = invalid_count / total_count if total_count else 0

        invalid_count_ = ModelRes()
        invalid_count_.value = invalid_count
        valid_count_ = ModelRes()
        valid_count_.value = valid_count
        total_count_ = ModelRes()
        total_count_.value = total_count
        invalid_ratio_ = ModelRes()
        invalid_ratio_.value = round(invalid_rate, 4)

        return {
            "invalid_count": invalid_count_,
            "valid_count": valid_count_,
            "total_count": total_count_,
            "invalid_ratio": invalid_ratio_
        }


