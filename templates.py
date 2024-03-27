from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplate:
    response_prefix: str
    instruction_prefix: Optional[str] = None


TEMPLATES = {
    "alpaca": PromptTemplate(instruction_prefix="\n\n### 指示:\n", response_prefix="\n\n### 応答:\n"),
    "chat": PromptTemplate(instruction_prefix="USER:", response_prefix="\nASSISTANT:"),
    "none": PromptTemplate(response_prefix="\n"),  # 真のプロンプトなし（空文字）は SFT の学習上、難しいというか入出力を切りたいので、改行だけ入れる
}
