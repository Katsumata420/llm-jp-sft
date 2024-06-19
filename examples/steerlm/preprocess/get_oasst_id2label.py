import argparse
import json

from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="OpenAssistant/oasst1")
    parser.add_argument("--output_file", type=str, default="oasst1_id2label.json")
    parser.add_argument("--language", type=str, default="en")
    return parser.parse_args()


def main():
    args = get_args()

    dataset = load_dataset(args.dataset)
    id2label = {}
    for sample in dataset["train"]:
        lang = sample["lang"]
        if lang != args.language:
            continue
        id2label[sample["message_id"]] = sample["labels"]

    with open(args.output_file, "w") as f:
        json.dump(id2label, f, indent=2)


if __name__ == "__main__":
    main()
