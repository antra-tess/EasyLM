#!/bin/bash

set -eu

echo "Starting simple Gemma-3 SFT training test..."

# Create directories (with configurable paths)
OUTPUT_DIR=${OUTPUT_DIR:-"./gemma_sft_output"}
TEMPLATE_PATH=${TEMPLATE_PATH:-"templates/borg_chat_exp.yaml"}
DATA_PATH=${DATA_PATH:-"./data/conversations.jsonl"}

# Make sure directories exist
mkdir -p $OUTPUT_DIR

echo "Using:"
echo "- Output directory: $OUTPUT_DIR"
echo "- Template: $TEMPLATE_PATH"
echo "- Data: $DATA_PATH"

# Run a very simple version of training for testing purposes
# Single GPU, limited parameters, focused on debugging
python gemma_sft_train.py \
    --model_name_or_path "google/gemma-3-2b-pt" \
    --dataset_path $DATA_PATH \
    --template_path $TEMPLATE_PATH \
    --max_seq_length 512 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --save_steps 50 \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --overwrite_output_dir \
    --gradient_checkpointing True \
    --save_total_limit 1 \
    --max_grad_norm 1.0

echo "Test training complete!" 