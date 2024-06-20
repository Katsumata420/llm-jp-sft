
INPUT_DATA_PATH=$1
OUTPUT_DATA_PATH=$2

MODEL_NAME=$3

BATCH_SIZE=32
GRAD_ACCUM=1

export WANDB_ENTITY_NAME=
export WANDB_PROJECT_NAME=
export HF_HOME=

python -m attribute_predict.run_classification \
    --train_file $INPUT_DATA_PATH \
    --output_dir $OUTPUT_DATA_PATH \
    --model_name_or_path $MODEL_NAME \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --per_gpu_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_epochs 3.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 3 \
    --overwrite_output_dir \
    --seed 42 \
    --use_lora \
    --fp16 \
    --lora_target_model llama \
    --lora_r 8 \
    --lora_alpha 16 \
    --do_regression \
    --eval_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model eval_mse \
    --text_column_name text
