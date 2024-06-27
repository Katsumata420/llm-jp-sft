"""適当な設定でLLMモデルを推論するスクリプト
要件:
     - 入出力は wandb の Table で保存
       - 入力モデル名
       - _id
       - 入力テキスト
       - 出力テキスト
    - 別途指定した id に対応する入出力も保存
    - 別途統計情報を取得する
      単語の定義自体は sudashi の mode-c に準拠
      - 単語の頻度
        - 単語と頻度のペア
        - top-30 単語
"""
import argparse
import json
from collections import defaultdict
from typing import Optional

import torch
import wandb
from peft import PeftModel, PeftConfig
from sudachipy import Dictionary, SplitMode
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from steer_lm_hf.preprocess.common import STEERLM_LABELS


WANDB_ENTITY_NAME=
PROJECT_NAME=
FONT="/path/to/font/NotoSansJP/static/NotoSansJP-Regular.ttf"
EXCLUDE_POS=["助詞", "助動詞", "補助記号"]
TOP=30

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLMモデルの推論を行うスクリプト")
    parser.add_argument(
        "model_path", type=str, help="推論対象のモデルのパス"
    )
    parser.add_argument(
        "input_path", type=str, help="推論対象の入力データのパス"
    )
    parser.add_argument(
        "output_path", type=str, help="推論結果の出力先のパス"
    )
    parser.add_argument("analysis_id", type=str, help="解析対象の ID が格納されたファイルパス")
    parser.add_argument("--is_lora", action="store_true", help="LoRAのデータかどうか")
    # generation config
    parser.add_argument("--do_sample", action="store_true", help="サンプリングするかどうか")
    parser.add_argument("--max_length", type=int, default=2048, help="生成するトークンの最大長")
    parser.add_argument("--temperature", type=float, default=0.7, help="サンプリング時の温度")
    parser.add_argument("--top_p", type=float, default=0.95, help="トークンの選択確率の上限")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="トークンの繰り返しペナルティ")
    # output config
    parser.add_argument("--model_name_aka", type=str, default="", help="モデルの名前")
    parser.add_argument("--prompt_type", type=str, choices=["alpaca", "chat", "inst", "none"], default="alpaca")
    # wandb config
    parser.add_argument("--is_wandb", action="store_true", help="wandb に保存するかどうか")
    parser.add_argument("--wandb_config_json", type=str, help="wandb に使用する config ファイル", default=None)
    return parser.parse_args()


def get_prompt_format(prompt_type: str) -> str:
    prompt_format: str
    if prompt_type == "alpaca":
        prompt_format = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{input_text}\n\n### 応答:\n"
    elif prompt_type == "alpaca-steerlm":
        prompt_format = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{input_text}\n{label}\n\n### 応答:\n"
    elif prompt_type == "chat":
        prompt_format = "USER:{input_text}\nASSISTANT:"
    elif prompt_type == "inst":
        # https://huggingface.co/docs/transformers/v4.39.1/en/chat_templating
        prompt_format = "[INST]{input_text}[/INST]"
    elif prompt_type == "none":
        prompt_format = "{input_text}"
    else:
        raise NotImplementedError()
    return prompt_format


def get_word_frequency(output_llm_outputs: list[dict[str, str]]) -> dict[str, int]:
    tokenizer = Dictionary().create()
    freq = defaultdict(int)
    for output in output_llm_outputs:
        for word in tokenizer.tokenize(output["output"], mode=SplitMode.C):
            pos = word.part_of_speech()[0]
            surface = word.surface()
            if pos in EXCLUDE_POS:
                continue
            freq[surface] += 1
    return freq


def get_tokenized_texts(output_llm_outputs: list[dict[str, str]]) -> list[str]:
    tokenizer = Dictionary().create()
    texts = []
    for output in output_llm_outputs:
        for word in tokenizer.tokenize(output["output"], mode=SplitMode.C):
            pos = word.part_of_speech()[0]
            surface = word.surface()
            if pos in EXCLUDE_POS:
                continue
            texts.append(surface)
    return texts


def save_wandb(
    output_llm_outputs: list[dict[str, str]],
    analysis_ids: list[str],
    wandb_config: Optional[dict],
) -> None:
    """wandb にデータを保存する
    保存する内容は次のとおり
    - Table に保存
      - 全件出力
      - ID に対応する出力
      - 統計情報
        - 単語の頻度
    """
    model_name = output_llm_outputs[0]["model"]
    # Table に保存
    with wandb.init(entity=WANDB_ENTITY_NAME, project=PROJECT_NAME, name=model_name, config=wandb_config) as run:
        # 全件出力
        columns = list(output_llm_outputs[0].keys())
        items = [[output[column] for column in columns] for output in output_llm_outputs]
        all_output_table = wandb.Table(data=items, columns=columns)

        # ID に対応する出力
        specific_id_items = [output for output in output_llm_outputs if output["ID"] in analysis_ids]
        specific_id_items = [[output[column] for column in columns] for output in specific_id_items]
        specific_id_output_table = wandb.Table(data=specific_id_items, columns=columns)

        # 統計情報
        ## 単語の頻度
        word_frequency = get_word_frequency(output_llm_outputs)
        word_freq_items = [[word, freq] for word, freq in word_frequency.items()]
        freq_table_columns = ["word", "frequency"]
        freq_table = wandb.Table(data=word_freq_items, columns=freq_table_columns)

        ## 上位K単語
        top_ten_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:TOP]
        top_ten_freq_columns = ["model", "words"]
        top_ten_freq_items = [[model_name, ", ".join([word for word, _ in top_ten_words])]]
        top_ten_freq_table = wandb.Table(data=top_ten_freq_items, columns=top_ten_freq_columns)

        run.log({
            "all_output": all_output_table,
            "specific_id_output": specific_id_output_table,
            "word_freq": freq_table,
            "top_ten_freq": top_ten_freq_table,
        })


def main():
    args = get_args()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.is_lora:
        config = PeftConfig.from_pretrained(args.model_path)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype="auto")
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto")

    if torch.cuda.is_available():
        model.to("cuda")

    with open(args.input_path, "r") as f:
        original_data = json.load(f)

    with open(args.analysis_id, "r") as f:
        _ids_data = json.load(f)
        analysis_ids = [d["ID"] for d in _ids_data]

    prompt_format = get_prompt_format(args.prompt_type)

    output_llm_outputs: list[dict[str, str]] = []
    with torch.no_grad():
        for sample in tqdm(original_data):
            if args.prompt_type == "alpaca-steerlm":
                assert "label" in sample, "ラベルが存在しません"
                assert list(sample["label"].keys()) == STEERLM_LABELS, "ラベルが不正です"
                label = ",".join([f"{k}:{v}" for k, v in sample["label"].items()])
                input_text = sample["text"]
                input_text = prompt_format.format(input_text=input_text, label=label)
            else:
                input_text = prompt_format.format(input_text=sample["text"])
            input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
            output = model.generate(
                input_ids=input_ids,
                do_sample=args.do_sample,
                max_new_tokens=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0]
            output = output[input_ids.size(1):]

            output_text = tokenizer.decode(output.tolist(), skip_special_tokens=True)
            output_llm_outputs.append({
                "ID": sample["ID"],
                "model": args.model_name_aka if args.model_name_aka != "" else args.model_path,
                "input-text": sample["text"],
                "input-prompt": input_text,
                "output": output_text,
                "reference": sample["output"],
            })

    with open(args.output_path, "w") as f:
        for sample in output_llm_outputs:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # wandb に保存
    if args.is_wandb:
        if args.wandb_config_json is not None:
            with open(args.wandb_config_json, "r") as f:
                wandb_config = json.load(f)
        else:
            wandb_config = None
        save_wandb(output_llm_outputs, analysis_ids, wandb_config)


if __name__ == "__main__":
    main()
