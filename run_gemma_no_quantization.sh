#!/bin/bash

set -eu

echo "Starting Gemma-3 multi-GPU training WITHOUT quantization (bf16 precision)..."

# Create directories (with configurable paths)
OUTPUT_DIR=${OUTPUT_DIR:-"./gemma3_full_precision_output"}
TEMPLATE_PATH=${TEMPLATE_PATH:-"templates/borg_chat_exp.yaml"}
DATA_PATH=${DATA_PATH:-"./data/conversations.jsonl"}

# Make sure directories exist
mkdir -p $OUTPUT_DIR

echo "Using:"
echo "- Output directory: $OUTPUT_DIR"
echo "- Template: $TEMPLATE_PATH"
echo "- Data: $DATA_PATH"

# Set torch distributed environment variables
export NCCL_DEBUG=INFO

# Define DeepSpeed config inline as a JSON string
DS_CONFIG='{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "bf16": {"enabled": "auto"},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}'

# Run the training script with DeepSpeed for multi-GPU
deepspeed --num_gpus=2 gemma_sft_train.py \
    --model_name_or_path "google/gemma-3-27b-pt" \
    --dataset_path $DATA_PATH \
    --template_path $TEMPLATE_PATH \
    --max_seq_length 1024 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
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
    --ddp_find_unused_parameters False \
    --deepspeed "$DS_CONFIG" \
    --load_in_8bit False \
    --load_in_4bit False

echo "Training complete!" 