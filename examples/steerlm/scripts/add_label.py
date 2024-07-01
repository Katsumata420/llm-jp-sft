import json
import argparse

from steerlm_hf.preprocess.common import STEERLM_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json")
    parser.add_argument("--output_json")
    parser.add_argument("--toxicity_label", required=True, type=int)

    args = parser.parse_args()

    label = {}
    four_label = ["quality", "helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    zero_label = ["humor", "creativity"]
    for l in STEERLM_LABELS:
        if l == "toxicity":
            label[l] = args.toxicity_label
        elif l in four_label:
            label[l] = 4
        elif l in zero_label:
            label[l] = 0
        else:
            raise ValueError(f"Unknown label: {l}")

    with open(args.input_json, "r") as f:
        data = json.load(f)

    for d in data:
        d["label"] = label

    with open(args.output_json, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
