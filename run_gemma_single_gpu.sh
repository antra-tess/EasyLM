#!/bin/bash

set -eu

echo "Starting Gemma-3 training on a SINGLE GPU without quantization (bf16 precision)..."

# Create directories (with configurable paths)
OUTPUT_DIR=${OUTPUT_DIR:-"./gemma3_single_gpu_output"}
TEMPLATE_PATH=${TEMPLATE_PATH:-"templates/borg_chat_exp.yaml"}
DATA_PATH=${DATA_PATH:-"./data/conversations.jsonl"}

# Make sure directories exist
mkdir -p $OUTPUT_DIR

echo "Using:"
echo "- Output directory: $OUTPUT_DIR"
echo "- Template: $TEMPLATE_PATH"
echo "- Data: $DATA_PATH"

# Use a smaller model to fit on a single GPU
MODEL_NAME=${MODEL_NAME:-"google/gemma-3-2b-pt"}

# Run the training script on a single GPU
python gemma_sft_train.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_path $DATA_PATH \
    --template_path $TEMPLATE_PATH \
    --max_seq_length 1024 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_steps 75 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_steps 100 \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --overwrite_output_dir \
    --gradient_checkpointing True \
    --save_total_limit 3 \
    --max_grad_norm 1.0 \
    --load_in_8bit False \
    --load_in_4bit False

echo "Training complete!" 