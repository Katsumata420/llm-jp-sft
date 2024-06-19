"""SteerLM の Attribute Prediction Model を学習するためのデータを作成するスクリプト

出力されるデータは次のような jsonl
[
    {"role": "user", "content": "text", "label": {"key": value, ...}},
    ...
]
"""
import argparse
import json
from typing import Optional

from .common import OASST_LABELS

# https://github.com/NVIDIA/NeMo-Aligner/blob/2ac4365246fddce3289b4c61e31918fe616b9e82/examples/nlp/data/steerlm/common.py#L28
likeart_scale = 5


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oasst_train_data", type=str, required=True, help="message_id が付与された状態の llm-jp-tuning-data 出力（train）")
    parser.add_argument("--oasst_dev_data", type=str, required=True, help="message_id が付与された状態の llm-jp-tuning-data 出力（dev）")
    parser.add_argument("--id2label_file", type=str, required=True, help="message_id から label への変換辞書")
    parser.add_argument("--output_file", type=str, default="steerlm_data.jsonl")
    return parser.parse_args()


def load_id2label(file_path: str) -> dict:
    with open(file_path) as f:
        return json.load(f)


def build_sample(data: list[dict], id2label: dict) -> list[dict]:
    """SteerLM の Attribute Prediction Model 用のデータを作成する

    Args:
        data (list[dict]): oasst のデータ
        次のような dict が格納されている（必ず assistant で終わっている）
            {
                "ID": turn 全体でのID,
                "messages": [
                    {"role": "system", "content": ...},
                    {"role": "user", "content": ..., "message_id": ...},
                    {"role": "assistant", "content": ..., "message_id": ...},
                ]
            }
        id2label (dict): message_id から label への変換辞書

    Returns:
        list[dict]: SteerLM の Attribute Prediction Model 用のデータ
    """
    def format_label(raw_label: Optional[dict]) -> Optional[dict]:
        """oasst のラベルを steerlm のラベルに変換する

        Returns:
            Optional[dict]: key: value の単純な形式
            想定したラベルが一つも含まれていない場合は None を返す
            また raw_label が None の場合も None を返す
        """
        if raw_label is None:
            return None
        raw_names = raw_label["name"]
        raw_values = raw_label["value"]
        raw_key_value = {name: value for name, value in zip(raw_names, raw_values)}
        formatted_label = {}
        # filter only OASST_LABELS
        # round lickert scale to 5
        for label in OASST_LABELS:
            if label in raw_key_value:
                value = raw_key_value[label]
                if value is not None:
                    formatted_label[label] = round(value * (likeart_scale - 1))
        if len(formatted_label) == 0:
            return None
        return formatted_label

    samples_with_label = []
    for sample in data:
        messages_with_label = []
        for message in sample["messages"]:
            if message["role"] in ["user", "assistant"]:
                message_id = message["message_id"]
                if message_id not in id2label:
                    continue
                label = id2label[message_id]
                messages_with_label.append({
                    "role": message["role"],
                    "content": message["content"],
                    "label": format_label(label),
                })
        if len(messages_with_label) == len(sample["messages"]) - 1:  # exclude system message
            samples_with_label.append(messages_with_label)
    return samples_with_label


def main():
    args = get_args()
    id2label = load_id2label(args.id2label_file)

    train_data = load_id2label(args.oasst_train_data)
    dev_data = load_id2label(args.oasst_dev_data)
    oasst_data = train_data + dev_data

    samples_with_labels = build_sample(oasst_data, id2label)

    with open(args.output_file, "w") as f:
        for sample in samples_with_labels:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
