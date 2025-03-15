#!/bin/bash

set -eu

echo "Starting Gemma-3-27b SFT training on 2x A100 80GB GPUs with unquantized LoRA (bf16)..."

# Check if WANDB_API_KEY is set
echo "WANDB_API_KEY is${WANDB_API_KEY:+ set}${WANDB_API_KEY:-" not set"}"
echo "HF_TOKEN is${HF_TOKEN:+ set}${HF_TOKEN:-" not set"}"

# Create directories
OUTPUT_DIR="./gemma_sft_output"
TEMPLATE_PATH="templates/borg_chat_exp.yaml"
DATA_PATH="conversations_all.jsonl"

# Make sure directories exist
mkdir -p $OUTPUT_DIR

# Create DeepSpeed config file with optimized settings for 2x A100 80GB with unquantized LoRA
cat > ds_config.json << EOF
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": 32,
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": false,
    "zero_allow_untested_optimizer": true,
    "zero_force_ds_cpu_optimizer": false
}
EOF

# Run the training script with DeepSpeed on 2 GPUs
python -m torch.distributed.run \
    --nproc_per_node=2 \
    gemma_sft_train.py \
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
    --deepspeed ds_config.json \
    --run_name "gemma-3-27b-sft-a100-bf16" \
    --bf16 True \
    --tf32 True \
    --overwrite_output_dir \
    --gradient_checkpointing True \
    --save_total_limit 3 \
    --max_grad_norm 1.0 \
    --ddp_find_unused_parameters False

echo "Training complete!" 