"""SteerLM の Attribute Prediction Model 用のデータを作成する

出力されるデータは次のような jsonl
[
    {"role": "user", "content": "text", "label": {"key": value, ...}},
    ...
]
"""
import argparse
import json

from datasets import load_dataset

from .common import HELPSTEER_LABELS

# https://github.com/NVIDIA/NeMo-Aligner/blob/2ac4365246fddce3289b4c61e31918fe616b9e82/examples/nlp/data/steerlm/common.py#L30
data_id = "kunishou/HelpSteer-35k-ja"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="helpsteer_data.jsonl")
    return parser.parse_args()


def build_samples(dataset) -> list[dict]:
    samples = []
    for data in dataset:
        prompt = data["prompt_ja"]
        response = data["response_ja"]

        helpfulness = data["helpfulness"]
        correctness = data["correctness"]
        coherence = data["coherence"]
        complexity = data["complexity"]
        verbosity = data["verbosity"]
        turns = [
            {"role": "user", "content": prompt, "label": None},
            {"role": "assistant", "content": response, "label": {
                "helpfulness": helpfulness,
                "correctness": correctness,
                "coherence": coherence,
                "complexity": complexity,
                "verbosity": verbosity
            }}
        ]
        samples.append(turns)
    return samples


def main():
    args = get_args()

    dataset = load_dataset(data_id, split="train")

    samples = build_samples(dataset)

    with open(args.output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
