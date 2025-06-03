from .rule import BaseRule,DynamicRuleConfig,TextSlice,ModelRes,split_paragraphs,normalize
from typing import Tuple
import re

class EndWithTerminal(BaseRule):
    """check whether the ratio of line ends with terminal punctuation mark > 0.6 """

    def __init__(self):
        super().__init__()
        self.dynamic_config = DynamicRuleConfig(threshold=0.6, key_list=[".", "!", "?", "”", "\""])

    def __process__(self, input_data) -> ModelRes:
        res = ModelRes()
        raw_content = input_data[0]
        raw_lines: Tuple[TextSlice] = split_paragraphs(
            text=raw_content, normalizer=lambda x: x, remove_empty=True
        )
        num_lines = len(raw_lines)
        if num_lines == 0:
            return res

        terminal_marks = [line.text.rstrip()[-1] for line in raw_lines if
                          line.text and line.text.rstrip()[-1] not in self.dynamic_config.key_list]
        num_occurrences = sum([line.text.rstrip().endswith(tuple(self.dynamic_config.key_list)) for line in raw_lines])
        ratio = num_occurrences / num_lines
        res.value = ratio
        if ratio < self.dynamic_config.threshold:
            res.error_status = True
            # res.type = cls.metric_type
            res.name = self.__class__.__name__
            res.reason = list(set(terminal_marks))
        return res


class EndWithEllipsis(BaseRule):
    """check whether the ratio of line ends with ellipsis < 0.3 """

    def __init__(self):
        super().__init__()
        self.dynamic_config = DynamicRuleConfig(threshold=0.3, key_list=["...", "…"])

    def __process__(self, input_data) -> ModelRes:
        res = ModelRes()
        raw_content = input_data[0]
        raw_lines: Tuple[TextSlice] = split_paragraphs(
            text=raw_content, normalizer=lambda x: x, remove_empty=True
        )
        num_lines = len(raw_lines)
        if num_lines == 0:
            return res

        num_occurrences = sum(
            [line.text.rstrip().endswith(tuple(self.dynamic_config.key_list)) for line in raw_lines])
        ratio = num_occurrences / num_lines
        res.value = ratio
        if ratio > self.dynamic_config.threshold:
            res.error_status = True
            # res.type = cls.metric_type
            res.name = self.__class__.__name__
            res.reason = ["The ratio of lines end with ellipsis is: " + str(ratio)]
        return res


class SentenceNumber(BaseRule):
    """check whether the number of sentence in [3, 7500] """

    def __init__(self):
        super().__init__()
        self.dynamic_config = DynamicRuleConfig(key_list=['3', '7500'])
        self.SENT_PATTERN = re.compile(r'\b[^.!?\n]+[.!?]*', flags=re.UNICODE)

    def __process__(self, input_data) -> ModelRes:
        res = ModelRes()
        raw_content = input_data[0]
        num_sentence = len(self.SENT_PATTERN.findall(raw_content))
        res.value = num_sentence
        if num_sentence < int(self.dynamic_config.key_list[0]) or num_sentence > int(self.dynamic_config.key_list[1]):
            res.error_status = True
            # res.type = cls.metric_type
            res.name = self.__class__.__name__
            res.reason = ["The number of sentence is: " + str(num_sentence)]
        return res

class WordNumber(BaseRule):
    """check whether the number of word in [20, 100000] """

    def __init__(self):
        super.__init__()
        self.dynamic_config = DynamicRuleConfig(key_list=['20', '100000'])

    def __process__(self, input_data) -> ModelRes:
        res = ModelRes()
        normalized_content = normalize(input_data[0])
        normalized_words = tuple(normalized_content.split())
        num_normalized_words = len(normalized_words)
        res.value = num_normalized_words
        if num_normalized_words >= int(self.dynamic_config.key_list[0]) and num_normalized_words < int(self.dynamic_config.key_list[1]):
            pass
        else:
            res.error_status = True
            # res.type = cls.metric_type
            res.name = self.__class__.__name__
            res.reason = ["The number of word is: " + str(num_normalized_words)]
        return res


class RuleFieldCompletenessManager(BaseRule):
    """RuleManager: Selects and applies the appropriate rule based on provided rule name."""

    RULE_MAP = {
        'EndWithTerminal': EndWithTerminal,
        'EndWithEllipsis': EndWithEllipsis,
        'SentenceNumber': SentenceNumber,
        'WordNumber': WordNumber
    }

    def __init__(self, rule_name: str):
        super().__init__()
        if rule_name not in self.RULE_MAP:
            raise ValueError(f"Unsupported rule name: {rule_name}")
        self.rule_instance = self.RULE_MAP[rule_name]()  # instantiate the corresponding rule class

    def __process__(self, input_data) -> ModelRes:
        """Process input_data using the selected rule."""
        print(self.rule_instance.__process__(input_data))
        print('*'*100)
        return self.rule_instance.__process__(input_data)