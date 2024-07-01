"""SteerLM で使用する Attribute を推定した結果出力、保存する

入力されるデータは次のような jsonl
[
    {"role": "user", "content": "text", "label": None},  # 場合によっては label が存在しない（e.g. AnswerCarefully）
    {"role": "assistant", "content": "text", "label": {"key": value, ...}},
    ...
]

出力されるデータは次のような jsonl
[
    {"role": "user", "content": "text", "label": None},
    {"role": "assistant", "content": "text", "label": {"key": value, ...}},
    ...
]

Examples:
    $ python -m steer_lm_hf.attribute_predict.run_inference \
        --input_file helpsteer_data.jsonl \
        --output_file helpsteer_data_with_attribute.jsonl \
        --model_name_or_id "/path/to/Attribute_predictor" \
        --batch_size 1 \
        --torch_dtype "bf16"
"""
import argparse
import json
from typing import Optional

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from .model import AttributePredictor, IGNORE_LABEL_VALUE
from ..preprocess.build_regression_dataset import SYSTEM_MESSAGE, USER_PREFIX, ASSISTANT_PREFIX
from ..preprocess.common import STEERLM_LABELS


class AttributePredictorInference:
    """Attribute Predictor の推論を行うクラス"""
    def __init__(
        self,
        model_id_or_name: str,
        lora_adapter: Optional[str],
        n_batch: int = 8,
        torch_dtype: str = "fp32",
        model_max_length: int = 2048,
        config_path: Optional[str] = None,
    ) -> None:
        if torch_dtype == "fp16":
            self.dtype = torch.float16
        elif torch_dtype == "bf16":
            self.dtype = torch.bfloat16
        elif torch_dtype == "fp32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

        if config_path is not None:
            config = AutoConfig.from_pretrained(config_path)
        else:
            config = None
        model = AttributePredictor.from_pretrained(model_id_or_name, torch_dtype=self.dtype, config=config)
        print(model)

        if lora_adapter is not None:
            self.model = PeftModel.from_pretrained(model, lora_adapter)
        else:
            self.model = model

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_name)
        self.max_length = model_max_length
        self.n_batch = n_batch

    def inference_batch(self, texts: list[str]) -> list[list[float]]:
        """テキストのリストを入力として Attribute のリストを出力する"""
        results = []
        for i in range(0, len(texts), self.n_batch):
            batch_texts = texts[i:i + self.n_batch]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length, return_token_type_ids=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            results.extend(logits.cpu().tolist())
        return results

    def inference_single(self, text: str) -> list[float]:
        """テキストを入力として Attribute のリストを出力する"""
        return self.inference_batch([text])[0]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="attribute_predict_result.jsonl")
    parser.add_argument("--model_name_or_id", type=str, required=True)
    parser.add_argument("--lora_adapter", type=str, default=None, help="LoRA の adapter を指定する")
    parser.add_argument("--batch_size", type=int, default=1, help="推論時のバッチサイズですが、今回は single で推論を行うため 1 にしています")
    parser.add_argument("--torch_dtype", type=str, default="fp32")
    parser.add_argument("--max_token_length", type=int, default=2048, help="入力テキストの最大トークン長")
    parser.add_argument("--config_path", type=str, default=None, help="モデルの config_path")
    return parser.parse_args()


def load_data(input_file: str) -> list:
    loaded_data = []
    with open(input_file) as f:
        for line in f:
            loaded_data.append(json.loads(line))
    return loaded_data


def run_inference(inference_model: AttributePredictorInference, loaded_data: list) -> list:
    """"すべてのデータに対して推論を行う

    注意事項として、推論結果は label が空の部分にのみ適用される
    また、role が assistant の部分のみ推論を行う
    その際の属性値は round する
    https://github.com/NVIDIA/NeMo-Aligner/blob/2ac4365246fddce3289b4c61e31918fe616b9e82/examples/nlp/data/steerlm/attribute_annotate.py#L139-L140
    """
    inference_results = []
    for sample in loaded_data:
        inference_result = []
        llm_input = SYSTEM_MESSAGE
        for turn_idx, turn in tqdm(enumerate(sample)):
            if turn_idx % 2 == 0:
                assert turn["role"] == "user"
            else:
                assert turn["role"] == "assistant"

            if turn["role"] == "user":
                inference_result.append({
                    "role": "user",
                    "content": turn["content"],
                    "label": None,  # 混乱を避けるため、user の方は必ず None にする
                })
                llm_input += USER_PREFIX + turn["content"]
            elif turn["role"] == "assistant":
                llm_input += ASSISTANT_PREFIX + turn["content"]
                predict_result = [min(4, max(0, float(v))) for v in inference_model.inference_single(llm_input)]  # 0 <= value <= 4
                assert len(predict_result) == len(STEERLM_LABELS)
                attribute_dict = {attr: round(value) for attr, value in zip(STEERLM_LABELS, predict_result)}

                if "label" not in turn or turn["label"] is None:
                    inference_result.append({
                        "role": "assistant",
                        "content": turn["content"],
                        "label": attribute_dict,
                    })
                else:
                    # label が存在する場合は、その部分はそのまま使う
                    # label の key が存在しない場合は、推論結果を使う
                    # label の key は存在するが、その値が None or IGNORE_LABEL_VALUE の場合も推論結果を使う
                    for attr in STEERLM_LABELS:
                        if attr not in turn["label"]:
                            turn["label"][attr] = attribute_dict[attr]
                        elif turn["label"][attr] is None or turn["label"][attr] == IGNORE_LABEL_VALUE:
                            turn["label"][attr] = attribute_dict[attr]
                    inference_result.append(turn)
        inference_results.append(inference_result)

    return inference_results


def main():
    args = get_args()

    inference_model = AttributePredictorInference(
        args.model_name_or_id, args.lora_adapter, args.batch_size, args.torch_dtype, args.max_token_length, args.config_path
    )
    loaded_data = load_data(args.input_file)
    results = run_inference(inference_model, loaded_data)

    with open(args.output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
