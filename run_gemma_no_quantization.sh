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

# First check the Conda environment's CUDA installation since that's the most likely to be configured correctly
NVCC_PATH=$(which nvcc 2>/dev/null || echo "")
if [ -n "$NVCC_PATH" ]; then
    echo "Found nvcc at: $NVCC_PATH"
    
    # Get the Conda environment base path (typically 2 levels up from bin/nvcc)
    CONDA_ENV_PATH=$(dirname $(dirname "$NVCC_PATH"))
    echo "Conda environment path: $CONDA_ENV_PATH"
    
    # Check if this is a full CUDA installation with headers
    if [ -f "$CONDA_ENV_PATH/include/cuda_runtime.h" ]; then
        echo "✅ Found cuda_runtime.h in Conda environment at $CONDA_ENV_PATH/include/cuda_runtime.h"
        export CUDA_HOME="$CONDA_ENV_PATH"
        export CUDA_TOOLKIT_ROOT_DIR="$CONDA_ENV_PATH"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        export CPATH="$CUDA_HOME/include:$CPATH"
        CUDA_FOUND=true
    else
        echo "⚠️ Found nvcc in Conda environment but cuda_runtime.h is missing"
        echo "Checking if include directory exists..."
        if [ -d "$CONDA_ENV_PATH/include" ]; then
            echo "Include directory exists. Listing available headers:"
            ls -la "$CONDA_ENV_PATH/include" | grep -i cuda
        else
            echo "Include directory does not exist in Conda environment"
        fi
        
        # Try to find headers in standard Conda locations
        if [ -f "$CONDA_ENV_PATH/lib/cuda/include/cuda_runtime.h" ]; then
            echo "✅ Found cuda_runtime.h at alternate Conda location: $CONDA_ENV_PATH/lib/cuda/include"
            export CUDA_HOME="$CONDA_ENV_PATH/lib/cuda"
            export CUDA_TOOLKIT_ROOT_DIR="$CONDA_ENV_PATH/lib/cuda"
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
            export CPATH="$CUDA_HOME/include:$CPATH"
            CUDA_FOUND=true
        else
            CUDA_FOUND=false
        fi
    fi
else
    echo "❌ nvcc not found in PATH"
    CUDA_FOUND=false
fi

# If we didn't find CUDA in the Conda environment, check standard system locations
if [ "$CUDA_FOUND" != true ]; then
    # Try to find CUDA installation paths
    CUDA_PATHS=(
        "/usr/local/cuda"
        "/usr/cuda"
        "/opt/cuda"
        "/usr/lib/cuda"
    )

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
fi

if [ "$CUDA_FOUND" = true ]; then
    echo "Using CUDA_HOME: $CUDA_HOME"
    
    # Check if we have the CUDA version
    if [ -f "$CUDA_HOME/version.txt" ]; then
        CUDA_VERSION=$(cat "$CUDA_HOME/version.txt")
        echo "CUDA Version: $CUDA_VERSION"
    elif [ -n "$NVCC_PATH" ]; then
        CUDA_VERSION=$($NVCC_PATH --version | grep "release" | awk '{print $6}' | sed 's/,//')
        echo "CUDA Version from nvcc: $CUDA_VERSION"
    fi
    
    # Set architecture for A100 GPUs
    echo "Adding GPU architecture for A100 (sm_80)"
    export TORCH_CUDA_ARCH_LIST="8.0"
    
    # Remove any previous failed compilations
    TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"~/.cache/torch_extensions"}
    echo "Clearing previous compilation artifacts from $TORCH_EXTENSIONS_DIR"
    rm -rf ~/.cache/torch_extensions/*/fused_adam
    
    # Print detailed diagnostic info for compilation
    echo "CUDA environment variables:"
    echo "- CUDA_HOME=$CUDA_HOME"
    echo "- CUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR"
    echo "- CPATH=$CPATH"
    
    # Check if we have the essential files for compilation
    ESSENTIAL_FILES=(
        "$CUDA_HOME/include/cuda_runtime.h"
        "$CUDA_HOME/include/cuda.h"
        "$CUDA_HOME/include/device_launch_parameters.h"
    )
    
    ALL_FILES_FOUND=true
    for file in "${ESSENTIAL_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "✅ Found: $file"
        else
            echo "❌ Missing: $file"
            ALL_FILES_FOUND=false
        fi
    done
    
    if [ "$ALL_FILES_FOUND" = true ]; then
        echo "All essential CUDA header files found. Compilation should succeed."
    else
        echo "⚠️ Some essential CUDA header files are missing. Fallback to CPU optimizer is recommended."
        CUDA_FOUND=false
    fi
else
    echo "⚠️ Could not find CUDA installation with cuda_runtime.h"
fi

if [ "$CUDA_FOUND" != true ]; then
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