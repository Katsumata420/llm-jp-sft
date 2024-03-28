from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplate:
    response_prefix: str
    instruction_prefix: Optional[str] = None


TEMPLATES = {
    "alpaca": PromptTemplate(instruction_prefix="\n\n### 指示:\n", response_prefix="\n\n### 応答:\n"),
    "chat": PromptTemplate(instruction_prefix="USER:", response_prefix="\nASSISTANT:"),
    "none": PromptTemplate(response_prefix="fake token"),  # none の場合、Instruction Tuning として学習するのではなく、通常の CLM として学習する。そのため、response prefix も本来は不要だが、データ形式の都合上、ダミーのトークンを入れている。
}
