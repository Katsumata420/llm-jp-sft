"""AttributePredictor で作成したデータをもとに SFT データを作成するスクリプト"""
import argparse
import json
from typing import Optional

from .build_regression_dataset import SYSTEM_MESSAGE, USER_PREFIX, ASSISTANT_PREFIX


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()


def load_data(input_file: str) -> list[dict]:
    data = []
    with open(input_file) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_sft_data(input_data: list[dict]) -> list[dict]:
    """SFT 向けのデータを作成する

    ASSISTANT 側では常にラベルが付与されており、そのラベルに基づいた Instruction を ASSISTANT_PREFIX のさらに前に追加する
    例としては次の通り（ラベルが quality:4, toxicity:0, humor:2 の場合）:
    ```json
    {"text": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{prompt}\nquality:4,toxicity:0,humor:2\n\n### 応答:\n{response}"}
    ```
    また、マルチターンの場合も同様に ASSISTANT_PREFIX の前に Instruction を追加する
    """
    sft_data = []
    for sample in input_data:
        sft_input = SYSTEM_MESSAGE
        for turn_idx, turn in enumerate(sample):
            role: str = turn["role"]
            content: str = turn["content"]
            label: Optional[dict] = turn["label"]

            if turn_idx % 2 == 0:
                assert role == "user"
                sft_input += USER_PREFIX + content
            else:
                assert role == "assistant"
                assert label is not None, "ASSISTANT のラベルが存在しません"
                label_text = ",".join([f"{k}:{v}" for k, v in label.items()])
                sft_input += f"\n{label_text}"
                sft_input += ASSISTANT_PREFIX + content
                # 各ターンごとに SFT データを作成
                sft_data.append({"text": sft_input})
    return sft_data


def main():
    args = get_args()

    input_data = load_data(args.input_file)

    sft_data = build_sft_data(input_data)

    with open(args.output_file, "w") as f:
        for data in sft_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
