from llm_base.base.base_res import ResponseScoreReason
from quality_filter.iterator.base import JsonIterator
from typing import Optional
class ScoreCollection(JsonIterator):
    """
    A class to collect scores from multiple sources.
    """

    def __process__(self, input_data: ResponseScoreReason) -> Optional[float]:
        """
        Process the input data and collect scores.
        """
        if input_data is None:
            return None
        score = None
        if input_data.state:
            score = input_data.response_json.get("score")  
        return score
