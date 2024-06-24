"""AttributePredictor で作成したデータをもとに SFT データを作成するスクリプト"""
import argparse
import json
from collections import defaultdict, namedtuple
from typing import Optional

from .build_regression_dataset import SYSTEM_MESSAGE, USER_PREFIX, ASSISTANT_PREFIX


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true", help="冗長な出力、特に各ラベルごとの統計情報を表示する")
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


def stats_labels(data: list[dict]) -> dict:
    """ラベルごとの統計情報を表示する

    Returns:
        dict: ラベルごとの統計情報（key: label, value: {label_value: namedtuple(count, ratio)}）
    """
    stats = namedtuple("Stats", ["count", "ratio"])

    label_stats = defaultdict(dict)
    for sample in data:
        for turn in sample:
            if turn["role"] == "assistant":
                for label, value in turn["label"].items():
                    if label not in label_stats:
                        label_stats[label] = defaultdict(int)
                    label_stats[label][value] += 1

    for label, value_counts in label_stats.items():
        total = sum(value_counts.values())
        label_stats[label] = {value: stats(count, count / total) for value, count in value_counts.items()}

    for label, stats in sorted(label_stats.items(), key=lambda x: x[0]):
        print(f"Label: {label}")
        for value, stat in sorted(stats.items(), key=lambda x: x[0]):
            print(f"  {value}: {stat.count} ({stat.ratio:.2%})")
    return label_stats


def main():
    args = get_args()

    input_data = load_data(args.input_file)
    if args.verbose:
        stats_labels(input_data)

    sft_data = build_sft_data(input_data)

    with open(args.output_file, "w") as f:
        for data in sft_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
