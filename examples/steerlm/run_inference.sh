
export HF_HOME=
BASE_MODEL=

INPUT_DATA=$1
OUTPUT_PATH=$2
PEFT_PATH=$3

python -m steer_lm_hf.attribute_predict.run_inference \
  --input_file $INPUT_DATA \
  --output_file $OUTPUT_PATH \
  --model_name_or_id $BASE_MODEL \
  --lora_adapter $PEFT_PATH \
  --batch_size 1 \
  --torch_dtype bf16
