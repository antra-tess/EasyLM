#!/bin/bash

set -eu

echo "Starting Gemma-3-27b SFT training on single GPU with unquantized LoRA (bf16)..."

# Check if WANDB_API_KEY is set
echo "WANDB_API_KEY is${WANDB_API_KEY:+ set}${WANDB_API_KEY:-" not set"}"
echo "HF_TOKEN is${HF_TOKEN:+ set}${HF_TOKEN:-" not set"}"

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

# Run the training script without DeepSpeed on a single GPU
python gemma_sft_train.py \
    --model_name_or_path "google/gemma-3-27b-pt" \
    --dataset_path $DATA_PATH \
    --template_path $TEMPLATE_PATH \
    --max_seq_length 1024 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_steps 75 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_steps 100 \
    --output_dir $OUTPUT_DIR \
    --report_to "wandb" \
    --run_name "gemma-3-27b-sft-single-gpu-bf16" \
    --bf16 True \
    --tf32 True \
    --overwrite_output_dir \
    --gradient_checkpointing True \
    --save_total_limit 3 \
    --max_grad_norm 1.0 \
    --ddp_find_unused_parameters False

echo "Training complete!" 