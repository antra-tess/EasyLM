#!/bin/bash

set -eu

echo "Starting Gemma-3 multi-GPU training OPTIMIZED FOR SPEED (bf16 precision)..."

# Create directories (with configurable paths)
OUTPUT_DIR=${OUTPUT_DIR:-"./gemma3_full_precision_output"}
TEMPLATE_PATH=${TEMPLATE_PATH:-"templates/borg_chat_exp.yaml"}
DATA_PATH=${DATA_PATH:-"./conversations_all.jsonl"}

# Make sure directories exist
mkdir -p $OUTPUT_DIR

# Print absolute paths to help with debugging
TEMPLATE_ABS_PATH=$(realpath "$TEMPLATE_PATH" 2>/dev/null || echo "File not found: $TEMPLATE_PATH")
DATA_ABS_PATH=$(realpath "$DATA_PATH" 2>/dev/null || echo "File not found: $DATA_PATH")

echo "Using:"
echo "- Output directory: $OUTPUT_DIR"
echo "- Template: $TEMPLATE_PATH (absolute: $TEMPLATE_ABS_PATH)"
echo "- Data: $DATA_PATH (absolute: $DATA_ABS_PATH)"

# Check if files exist
if [[ ! -f "$TEMPLATE_PATH" ]]; then
    echo "ERROR: Template file not found: $TEMPLATE_PATH"
    exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: Dataset file not found: $DATA_PATH"
    exit 1
fi

# Display first few lines of the dataset to verify its structure
echo "==== Dataset preview (first 5 lines) ===="
head -n 5 "$DATA_PATH"
echo "==== End of dataset preview ===="

# Display template file content to verify its structure
echo "==== Template file content ===="
cat "$TEMPLATE_PATH"
echo "==== End of template file content ===="

# Diagnostic information about CUDA setup
echo "==== CUDA Diagnostics ===="
echo "Checking for CUDA installation..."

# Try to find CUDA installation paths
CUDA_PATHS=(
    "/usr/local/cuda"
    "/usr/cuda"
    "/opt/cuda"
    "/usr/lib/cuda"
)

CUDA_FOUND=false
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "Found CUDA at: $path"
        if [ -f "$path/include/cuda_runtime.h" ]; then
            echo "✅ Found cuda_runtime.h at $path/include/cuda_runtime.h"
            export CUDA_HOME="$path"
            export CUDA_TOOLKIT_ROOT_DIR="$path"
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
            export CPATH="$CUDA_HOME/include:$CPATH"
            CUDA_FOUND=true
            break
        else
            echo "❌ cuda_runtime.h not found in $path/include"
        fi
    fi
done

if [ "$CUDA_FOUND" = false ]; then
    echo "No standard CUDA installation found. Using nvcc to determine CUDA location..."
    NVCC_PATH=$(which nvcc 2>/dev/null || echo "")
    if [ -n "$NVCC_PATH" ]; then
        CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
        echo "Found CUDA at: $CUDA_HOME (derived from nvcc path)"
        export CUDA_HOME="$CUDA_HOME"
        export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        export CPATH="$CUDA_HOME/include:$CPATH"
        if [ -f "$CUDA_HOME/include/cuda_runtime.h" ]; then
            echo "✅ Found cuda_runtime.h at $CUDA_HOME/include/cuda_runtime.h"
            CUDA_FOUND=true
        else
            echo "❌ cuda_runtime.h not found in $CUDA_HOME/include"
        fi
    else
        echo "❌ nvcc not found in PATH"
    fi
fi

if [ "$CUDA_FOUND" = true ]; then
    echo "Using CUDA_HOME: $CUDA_HOME"
    echo "Adding GPU architecture for A100 (sm_80)"
    export TORCH_CUDA_ARCH_LIST="8.0"
    # Remove any previous failed compilations
    rm -rf /home/athuser/.cache/torch_extensions/py312_cu124/fused_adam
else
    echo "⚠️  Could not find CUDA installation with cuda_runtime.h"
    echo "Switching to standard PyTorch optimizer to avoid compilation issues"
    
    # Use these environment variables to disable CUDA ops compilation
    export DS_BUILD_OPS=0 
    export DS_BUILD_FUSED_ADAM=0
    export DS_BUILD_FUSED_LAMB=0
    
    # Add parameter to use standard PyTorch optimizer
    ADDITIONAL_ARGS="--optim adamw_torch"
fi
echo "==== End CUDA Diagnostics ===="

# Set torch distributed environment variables
export NCCL_DEBUG=INFO

# Define DeepSpeed config with AdamW optimizer 
DS_CONFIG='{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "none"
    },
    "offload_param": {
      "device": "none"
    },
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto",
      "total_num_steps": "auto"
    }
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true
  }
}'

# Set default additional args if not defined
ADDITIONAL_ARGS=${ADDITIONAL_ARGS:-""}

# Run with the appropriate optimizer
deepspeed --num_gpus=2 gemma_sft_train.py \
    --model_name_or_path "google/gemma-3-27b-pt" \
    --dataset_path $DATA_PATH \
    --template_path $TEMPLATE_PATH \
    --max_seq_length 1024 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_steps 100 \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --tf32 True \
    --overwrite_output_dir \
    --gradient_checkpointing True \
    --save_total_limit 3 \
    --max_grad_norm 1.0 \
    --ddp_find_unused_parameters False \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --weight_decay 0.01 \
    --deepspeed "$DS_CONFIG" \
    --load_in_8bit False \
    --load_in_4bit False \
    $ADDITIONAL_ARGS

echo "Training complete!" 