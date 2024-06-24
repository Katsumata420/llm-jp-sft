"""AnswerCarefully データを turn 形式で作成するスクリプト

出力は以下の形式:
[
    [
        {"role": "user", "content": "ユーザの発話", "label": None},
        {"role": "assistant", "content": "システムの発話", "label": None},
    ],
    ...
]
"""
import argparse
import json


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()


def load_ac_data(file_path: str) -> list[dict]:
    with open(file_path) as f:
        data = json.load(f)
    return data


def build_turn_data(ac_data: list[dict]) -> list[list[dict]]:
    turn_data = []
    for sample in ac_data:
        user_text = sample["text"]
        assistant_text = sample["output"]
        turn_data.append([
            {"role": "user", "content": user_text, "label": None},
            {"role": "assistant", "content": assistant_text, "label": None},
        ])
    return turn_data


def main():
    args = get_args()

    ac_data = load_ac_data(args.input_file)

    output_data = build_turn_data(ac_data)

    with open(args.output_file, "w") as f:
        for data in output_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
