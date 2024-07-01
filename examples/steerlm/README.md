# SteerLM with huggingface

Huggingface ライブラリを利用した SteerLM の実装です。
主に Attribute Prediction Model をおいています。

基本的に [nvidia 公式の document](https://github.com/NVIDIA/NeMo-Aligner/blob/8d8ef38bc7b190d9a3d0b5face76147b9e0ae863/docs/user-guide/steerlm.rst) と同様の手順で実行できるようにしています。

## Requirement
install.sh を参考にしてください。

## Attribute Prediction Model の構築から SteerLM SFT データへの属性付与までの手順

以下の手順で Attribute Prediction Model を構築し、SteerLM SFT データに属性を付与します。

1. Preprocess: OASST/HelpSteer のデータを Attribute Prediction Model に適した形式に変換します
2. Train Attribute Prediction Model: Attribute Prediction Model を学習します
3. Prediction Attribute to SFT Data: 学習した Attribute Prediction Model を用いて SFT データに属性を付与します

### Preprocess
OASST/HelpSteer のデータを Attribute Prediction Model に適した形式に変換します

#### OASST Data

OASST については、事前に message_id 付きの日本語化 oasst1/2 のデータを取得していることが前提です。

その上で、以下の手順でデータを Attribute Prediction Model に適した形式に変換します。

```bash
# oasst1
$ python -m steerlm_hf.preprocess.get_oasst_id2label \
    --dataset OpenAssistant/oasst1 \
    --output_file /path/to/steerlm/data/attribute_prediction/temp/oasst1_id2label.json
$ python -m steerlm_hf.preprocess.build_steerlm_data \
    --oasst_train_data /path/to/tuning_data/train/oasst1_ja.json \
    --oasst_dev_data /path/to/tuning_data/dev/oasst1_ja.json \
    --id2label_file /path/to/steerlm/data/attribute_prediction/temp/oasst1_id2label.json \
    --output_file /path/to/steerlm/data/attribute_prediction/oasst1_data.jsonl
# oasst2
$ python -m steerlm_hf.preprocess.get_oasst_id2label \
    --dataset OpenAssistant/oasst2 \
    --output_file /path/to/steerlm/data/attribute_prediction/temp/oasst2_id2label.json
$ python -m steerlm_hf.preprocess.build_steerlm_data \
    --oasst_train_data /path/to/tuning_data/train/oasst2_ja.json \
    --oasst_dev_data /path/to/tuning_data/dev/oasst2_ja.json \
    --id2label_file /path/to/steerlm/data/attribute_prediction/temp/oasst2_id2label.json \
    --output_file /path/to/steerlm/data/attribute_prediction/oasst2_data.jsonl
```

#### HelpSteer Data

以下の手順でデータを Attribute Prediction Model に適した形式に変換します。

```bash
$ python -m steerlm_hf.preprocess.build_helpsteer_data \
    --output_file /path/to/steerlm/data/attribute_prediction/helpsteer_data.jsonl
```

#### 作成したデータを Attribute Prediction Model に適した形式に変換

下記のように、作成したデータを組み合わせた後、Attribute Prediction Model に適した形式に変換します。

```bash
$ cat /path/to/steerlm/data/attribute_prediction/oasst1_data.jsonl \
    /path/to/steerlm/data/attribute_prediction/oasst2_data.jsonl \
    /path/to/steerlm/data/attribute_prediction/helpsteer_data.jsonl \
    > /path/to/steerlm/data/attribute_prediction/attribute_prediction_data.jsonl
$ python -m steerlm_hf.preprocess.build_regression_dataset \
    --input_file /path/to/steerlm/data/attribute_prediction/attribute_prediction_data.jsonl \
    --output_file /path/to/steerlm/data/attribute_prediction/attribute_prediction_data_regression.jsonl
```

このようにして作成したファイル `attribute_prediction_data_regression.jsonl` を Attribute Prediction Model の学習に利用します。

### Train Attribute Prediction Model
Attribute Prediction Model を学習します

`run_train.sh` に簡単なサンプル（LoRA を実装済み）を記載しています。

必要に応じて適宜変更してください。
（`attn_implementation` を `flash_attention_2` に設定することで、Flash Attention を利用できます）

なお、受け付けるベースモデルのアーキテクチャは llama に限定されています。

また、ds_config.json についても適切に設定してください。

```bash
$ bash run_train.sh \
  /path/to/steerlm/data/attribute_prediction/attribute_prediction_data_regression.jsonl \
  /path/to/steerlm/models/AttributePredictionModel \
  llm-jp/llm-jp-13b-v2.0
```

#### Prediction Attribute to SFT Data
学習した Attribute Prediction Model を用いて SFT データに属性を付与します

ここでは例として、AnswerCarefully001 を想定します

まずは SFT データを次のような jsonl フォーマットに変換します

```json
[{"role": "user", "content": "text", "label": none}, {"role": "assistant", "content": "text", "label": {"key": value}}]
[{"role": "user", "content": "text", "label": none}, {"role": "assistant", "content": "text", "label": {"key": value}}]
...
```

AnswerCarefully001 の場合、次のスクリプトを用意しています

```bash
$ python -m steerlm_hf.preprocess.build_answercarefully \
    --input_file /path/to/AnswerCarefully001_Dev.jsonl \
    --output_file /path/to/steerlm/data/AnswerCarefully001_attribute_prediction_data.jsonl
```

次に、作成したデータに属性を付与します

`run_inference.sh` に簡単なサンプルを記載しています。

必要に応じて適宜変更してください。

また、推論に使用するモデルは、学習した Attribute Prediction Model を指定してください。

ただし、LoRA モデルの場合、別途 `LlamaForSequenceClassification` Config を用意するか、LoRA モデルのマージを実施する必要があります。

```bash
$ bash run_inference.sh \
    /path/to/steerlm/data/AnswerCarefully001_attribute_prediction_data.jsonl \
    /path/to/steerlm/data/sft/AnswerCarefully001_with_attribute.jsonl \
    /path/to/steerlm/models/AttributePredictionModel
```

oasst1/2 データについても同様に Attribute を付与することができます。

それぞれ Preprocess で作成したデータを利用してください。
oasst1 は `/path/to/steerlm/data/attribute_prediction/oasst1_data.jsonl` で作成したファイルを、oasst2 は `/path/to/steerlm/data/attribute_prediction/oasst2_data.jsonl` で作成したファイルを利用してください。

### Train SteerLM
作成した Attribute が付与された SFT データを用いて SteerLM を学習します

SFT の学習自体は本 llm-jp-sft の README.md に記載されている手順に従ってください。

ここでは llm-jp-sft の学習用フォーマットに変換する手順を記載します。

```bash
$ cat /path/to/steerlm/data/sft/AnswerCarefully001_with_attribute.jsonl \
    /path/to/steerlm//data/sft/oasst1_with_attribute.jsonl \
    /path/to/steerlm/data/sft/oasst2_with_attribute.jsonl \
    > /path/to/steerlm/data/sft/sft_data.jsonl
$ python -m steerlm_hf.preprocess.build_sft_data \
    --input_file /path/to/steerlm/data/sft/sft_data.jsonl \
    --output_file /path/to/steerlm/data/sft/llm-jp-sft_steerlm.jsonl \
    -v  # 各属性ごとの統計情報を表示
```

この作成したデータ `llm-jp-sft_steerlm.jsonl` を llm-jp-sft の学習に利用してください。

### Inference using SteerLM
学習した SteerLM を用いて推論を行います

`scripts/run_predict.py` に簡単なサンプルを記載しています。

ただし、このスクリプトは入力テキストに加えて、入力データに　SteerLM のラベルを付与する必要があります。

以下はこの SteerLM のラベルを付与するスクリプトの例です。

この例では、`AnswerCarefully001_Test.json` に Toxicity のラベルを0で付与しています。
（他の Attribute については、humor のみ0で、残りは4で付与しています）

```bash
$ python -m scripts.add_label \
    --input_json /path/to/AnswerCarefully001_Test.json \
    --output_json /path/to/AnswerCarefully001_Test_with_label.json \
    --toxicity_label 0
```

このようにして作成したファイル `AnswerCarefully001_Test_with_label.json` を利用して推論を行います。

```bash
$ python -m scripts.run_predict \
    /path/to/steerlm/sft/models/SteerLM \
    /path/to/AnswerCarefully001_Test_with_label.json \
    /path/to/steerlm/sft/results/AnswerCarefully001_Test_with_label.json \
    --is_lora \
    --do_sample \
    --prompt_type alpaca-steerlm
```


/path/to/steerlm/sft/results/AnswerCarefully001_Test_with_label.json に結果が出力されます。
