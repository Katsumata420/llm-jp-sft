"""OASST のデータから有害ラベルの付与されたユーザー入力を取得するスクリプト

Examples:
  有害性が0-4の5段階中3以上の入力、または hate_speech、not_appropriate、sextual_content、pii が
  付与されたサンプルを抽出する場合
  >> python scripts/view_toxicity_inputs.py \
    --input_file /path/to/oasst.jsonl \  # 有害性のラベルが付与されているデータ
    --use_toxicity_label hate_speech not_appropriate sextual_content pii \
    --toxicity_threshold 3 \  # 有害性が0-4の5段階中、3以上の応答
    --output_file /path/to/oasst_toxicity_inputs_all-binary_toxicity3.jsonl
"""
import argparse
import json


# In the OASST's paper, see the hate_speech, not_appropriate, sextual_content as toxicity labels.
# We add the PII as a toxicity label.
OASST_TOXICITY_BINARY_LABELS = {
    "hate_speech",
    "not_appropriate",
    "sextual_content",
    "pii",
}
OASST_TOXICITY_FIVE_SCALE_LABEL = "toxicity"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument(
        "--use_toxicity_label",
        nargs="*",
        choices=OASST_TOXICITY_BINARY_LABELS,
        default=[],
    )
    parser.add_argument("--toxicity_threshold", type=int, default=6)
    parser.add_argument("--output_file", default="extracted_user_input.jsonl")
    return parser.parse_args()


def load_samples(file_path: str) -> list[list[dict]]:
    samples = []
    with open(file_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def extract_user_input(
    samples: list[list[dict]],
    use_toxicity_label: list[str],
    toxicity_threshold: int,
    is_first_turn: bool,
) -> list[dict]:
    def check_label(label: dict) -> bool:
        assert len(label["name"]) == len(label["value"])
        label_dict = {key: value for key, value in zip(label["name"], label["value"])}
        # check binary
        for toxicy_label in use_toxicity_label:
            binary_score = label_dict.get(toxicy_label)
            if binary_score is not None and binary_score == 1.0:
                return True

        # check five scale
        if OASST_TOXICITY_FIVE_SCALE_LABEL in label_dict:
            toxicity_score = label_dict[OASST_TOXICITY_FIVE_SCALE_LABEL]
            five_scale_score = round(toxicity_score * 4)
            if five_scale_score >= toxicity_threshold:
                return True

        return False

    extracted_user_input = []
    for sample in samples:
        for idx, turn in enumerate(sample):
            if is_first_turn:
                if idx > 0:
                    break

            if turn["role"] == "user":
                user_input = turn["content"]
                label = turn["label"]
                is_toxicity = check_label(label)
                if is_toxicity:
                    extracted_user_input.append(
                        {"user_input": user_input, "label": label}
                    )
            else:
                continue

    return extracted_user_input


def main():
    args = get_args()
    print(args)

    samples = load_samples(args.input_file)

    extracted_user_input = extract_user_input(
        samples, args.use_toxicity_label, args.toxicity_threshold, is_first_turn=True
    )

    print(f"Extracted {len(extracted_user_input)} user inputs")
    with open(args.output_file, "w") as f:
        for user_input in extracted_user_input:
            f.write(json.dumps(user_input, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
