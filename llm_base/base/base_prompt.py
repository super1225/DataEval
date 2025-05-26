from typing import List


class BasePrompt:
    # metric_type: str  # This will be set by the decorator
    # group: List[str]  # This will be set by the decorator
    output_format: str = """

若分数范围没有指定则默认score的范围为1-10，reason表示给出这个分数的原因，reason使用中文解释。
输出格式如下。 不要给出多余的解释和内容。
{
    "score": xxx,
    "reason": xxx,
}
"""
    content: str = "你是一个有用的助手"

    system_prompt: str = ""


