"""SteerLM の Attribute Prediction Model 学習用のデータセットを作成するスクリプト

以下のような jsonl が出力フォーマット
    label の要素は STEERLM_LABELS に定義されているもの
{"text": tokenizer.apply_chat_template(input), "label": list[int]}
また、プロンプトの適用に tokenizer.apply_chat_template は用いておらず、PROMPT_TEMPLATE に定義されているものを用いている
マルチターンデータについては、各ターンごとに sample を作成する（SteerLM も同様の処理を行っている）
"""
import argparse
import json
from typing import Optional

from .common import STEERLM_LABELS


# llm-jp-sft フォーマット
SYSTEM_MESSAGE = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
USER_PREFIX = "\n\n### 指示:\n"
ASSISTANT_PREFIX = "\n\n### 応答:\n"
PROMPT_TEMPLATE_WO_SYSTEM = USER_PREFIX + "{input_text}" + ASSISTANT_PREFIX + "{answer}"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="steerlm_data.jsonl")
    parser.add_argument(
        "--ignore_label_value",
        default=-100,
        type=int,
        help="ラベルが存在しない場合の値（回帰モデル学習の際にはこの値で mask する）",
    )
    return parser.parse_args()


def load_data(input_file: str) -> list:
    samples = []
    with open(input_file) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def build_regression_dataset(samples: list, ignore_label_value: int) -> list:
    def build_llm_input(data: list) -> str:
        """マルチターンデータを LLM のプロンプト形式に変換する

        Returns:
            str: SYSTEM_MESSAGE  + \
                PROMPT_TEMPLATE_WO_SYSTEM.format(input_text=turn[0], answer=turn[1]) + \
                PROMPT_TEMPLATE_WO_SYSTEM.format(input_text=turn[2], answer=turn[3]) + ...
        """
        llm_input = SYSTEM_MESSAGE
        for idx in range(0, len(data), 2):
            llm_input += PROMPT_TEMPLATE_WO_SYSTEM.format(
                input_text=data[idx]["content"], answer=data[idx + 1]["content"]
            )
        return llm_input

    def build_regression_label(label: dict) -> list[int]:
        """ラベルを回帰モデル用のラベルに変換する

        Returns:
            list[int]: STEERLM_LABELS に対応するラベル
        """
        regression_label = []
        for attr in STEERLM_LABELS:
            if attr in label:
                regression_label.append(label[attr])
            else:
                regression_label.append(ignore_label_value)
        return regression_label

    regression_dataset = []
    for sample in samples:
        temp_data = []
        for turn_idx, turn in enumerate(sample):
            # turn_idx % 2 = 0 がユーザのターン
            # turn_idx % 2 = 1 がシステムのターン
            role: str = turn["role"]
            text: str = turn["content"]
            label: Optional[dict[str, int]] = turn["label"]

            if turn_idx % 2 == 0:
                assert role == "user"
                temp_data.append(turn)
            elif turn_idx % 2 == 1:
                assert role == "assistant"
                temp_data.append(turn)
                if label is None:
                    # label が存在しない場合は skip
                    continue
                llm_input = build_llm_input(temp_data)
                regression_dataset.append(
                    {"text": llm_input, "label": build_regression_label(label)}
                )

    return regression_dataset


def main():
    args = get_args()

    input_data = load_data(args.input_file)
    samples = build_regression_dataset(input_data, args.ignore_label_value)

    with open(args.output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
