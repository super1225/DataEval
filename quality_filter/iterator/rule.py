import json
from typing import Dict, List, Optional
from pydantic import BaseModel
import re
from collections import defaultdict
import json
from collections import Counter
from typing import Callable, List, Set, Tuple
import string
import zhon.hanzi
import unicodedata

TRANSLATION_TABLE_PUNCTUATION_EN = str.maketrans('', '', string.punctuation)
TRANSLATION_TABLE_PUNCTUATION_ZH = str.maketrans('', '', zhon.hanzi.punctuation)


class DynamicRuleConfig(BaseModel):
    threshold: Optional[float] = None
    pattern: Optional[str] = None
    key_list: Optional[List[str]] = None
    refer_path: Optional[List[str]] = None


class ModelRes(BaseModel):
    error_status: bool = False
    type: str = 'QUALITY_GOOD'
    name: str = 'Data'
    value: Optional[float] = None
    reason: List[str] = []


class BaseRule:
    # metric_type: str  # This will be set by the decorator
    # group: List[str]  # This will be set by the decorator
    # dynamic_config:  DynamicRuleConfig
    def __init__(self):
        pass

    def __process__(self, input_data) -> ModelRes:
        raise NotImplementedError()

    def on_start(self):
        """处理数据前，主要用于一些数据处理的准备工作，不应该用于具体数据处理"""
        pass

    def on_complete(self):
        """结束处理。主要用于数据处理结束后的清理工作，不应该用于具体数据处理"""
        pass

    @property
    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return f"{self.name}"


class Character(BaseRule):
    """check whether content has special characters. """

    def __init__(self):
        super().__init__()
        self.dynamic_config = DynamicRuleConfig(
            key_list=[
                r"u200e",
                # r"(\\\\;){3,}|(\{\}){3,}|(&nbsp;){3,}",
                r"&#247;|\? :",
                r"[�□]|\{\/U\}",
                r"U\+26[0-F][0-D]|U\+273[3-4]|U\+1F[3-6][0-4][0-F]|U\+1F6[8-F][0-F]",
                r"<\|.*?\|>"
            ]
        )

    def __process__(self, input_data) -> ModelRes:
        res = ModelRes()
        content = input_data[0]['data']
        if len(content) == 0:
            return res

        matches = []
        num = 0
        for p in self.dynamic_config.key_list:
            m = re.findall(p, content)
            num += len(m)
            matches = matches + m
        if num / len(content) >= 0.001:
            res.error_status = True
            res.value = num / len(content)
            # res.type = cls.metric_type
            res.name = self.__class__.__name__
            res.reason = list(set(matches))
        return res


class TextSlice:
    """A slice of text from a document."""

    def __init__(self, text: str, start: int, end: int):
        self.text = text
        self.start = start
        self.end = end


def normalize(
        text: str,
        remove_punct: bool = True,
        lowercase: bool = True,
        nfd_unicode: bool = True,
        white_space: bool = True
) -> str:
    """Normalize the text by lowercasing and removing punctuation."""
    # remove punctuation
    if remove_punct:
        text = text.translate(TRANSLATION_TABLE_PUNCTUATION_EN)
        text = text.translate(TRANSLATION_TABLE_PUNCTUATION_ZH)
    # lowercase
    if lowercase:
        text = text.lower()
    if white_space:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
    # NFD unicode normalization
    if nfd_unicode:
        text = unicodedata.normalize('NFD', text)
    return text


def split_paragraphs(
        text: str, normalizer: Callable[[str], str], remove_empty: bool = True
) -> Tuple[TextSlice]:
    """
    Split a string into paragraphs. A paragraph is defined as a sequence of zero or more characters, followed
    by a newline character, or a sequence of one or more characters, followed by the end of the string.
    """
    text_slices = tuple(
        TextSlice(normalizer(text[match.start():match.end()]), match.start(), match.end())
        for match in re.finditer(r"([^\n]*\n|[^\n]+$)", text)
    )

    if remove_empty is True:
        text_slices = tuple(
            text_slice for text_slice in text_slices if text_slice.text.strip()
        )

    return text_slices

# if __name__ == '__main__':
#     data = csv_to_json('data.csv', 'data.json')
#     with open('data.json') as f:
#         input_data = json.load(f)[0]["data"]
#     res = RuleSpecialCharacter().eval(input_data)
#     res1 = RuleLineEndWithTerminal().eval(input_data)
#     score = comprehensive_score({'SpecialCharacter': res.value, 'ending': res1.value},
#                                 {'SpecialCharacter': 0.6, 'ending': 0.4}, strategy='weighted_sum')
#     print(score)
