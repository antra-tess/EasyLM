#!/bin/bash

set -eu

echo "Starting Gemma-3 multi-GPU training OPTIMIZED FOR SPEED (bf16 precision)..."

# Initialize environment variables to prevent "unbound variable" errors
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-""}
export PATH=${PATH:-""}
export CPATH=${CPATH:-""}

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

# First try to find CUDA installation using the system paths
echo "Searching for CUDA headers in system locations..."
CUDA_SYSTEM_PATHS=(
    "/usr/local/cuda"
    "/usr/local/cuda-12.4"
    "/usr/cuda"
    "/usr/cuda-12.4"
    "/opt/cuda"
    "/opt/cuda-12.4"
)

CUDA_FOUND=false
for path in "${CUDA_SYSTEM_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "Found potential CUDA installation at: $path"
        if [ -f "$path/include/cuda_runtime.h" ]; then
            echo "✅ Found cuda_runtime.h at $path/include/cuda_runtime.h"
            CUDA_FOUND=true
            SYSTEM_CUDA_HOME="$path"
            break
        else
            echo "❌ cuda_runtime.h not found in $path/include"
        fi
    fi
done

# Now check the Conda environment's CUDA installation
NVCC_PATH=$(which nvcc 2>/dev/null || echo "")
if [ -n "$NVCC_PATH" ]; then
    echo "Found nvcc at: $NVCC_PATH"
    NVCC_VERSION=$($NVCC_PATH -V 2>&1 | grep "release" | awk '{print $5}' | awk -F, '{print $1}')
    echo "NVCC version: $NVCC_VERSION"
    
    # Get the Conda environment base path (typically 2 levels up from bin/nvcc)
    CONDA_ENV_PATH=$(dirname $(dirname "$NVCC_PATH"))
    echo "Conda environment path: $CONDA_ENV_PATH"
    
    # Check if this is a full CUDA installation with headers
    if [ -f "$CONDA_ENV_PATH/include/cuda_runtime.h" ]; then
        echo "✅ Found cuda_runtime.h in Conda environment at $CONDA_ENV_PATH/include/cuda_runtime.h"
        export CUDA_HOME="$CONDA_ENV_PATH"
        export CUDA_TOOLKIT_ROOT_DIR="$CONDA_ENV_PATH"
        CUDA_FOUND=true
    else
        echo "⚠️ Found nvcc in Conda environment but cuda_runtime.h is missing"
        
        # Try to find headers in standard Conda locations
        if [ -f "$CONDA_ENV_PATH/lib/cuda/include/cuda_runtime.h" ]; then
            echo "✅ Found cuda_runtime.h at alternate Conda location: $CONDA_ENV_PATH/lib/cuda/include"
            export CUDA_HOME="$CONDA_ENV_PATH/lib/cuda"
            export CUDA_TOOLKIT_ROOT_DIR="$CONDA_ENV_PATH/lib/cuda"
            CUDA_FOUND=true
        elif [ "$CUDA_FOUND" = true ] && [ -n "$SYSTEM_CUDA_HOME" ]; then
            # Use system CUDA headers but Conda's nvcc
            echo "⚠️ Using system CUDA headers with Conda's nvcc"
            export CUDA_HOME="$SYSTEM_CUDA_HOME"
            export CUDA_TOOLKIT_ROOT_DIR="$SYSTEM_CUDA_HOME"
            # Keep Conda's nvcc in the path
            export PATH="$CONDA_ENV_PATH/bin:$PATH"
        else
            # Try to find headers with find command
            echo "Searching for cuda_runtime.h across the system..."
            CUDA_RUNTIME_PATH=$(find / -name "cuda_runtime.h" -type f 2>/dev/null | head -n 1)
            if [ -n "$CUDA_RUNTIME_PATH" ]; then
                echo "✅ Found cuda_runtime.h at: $CUDA_RUNTIME_PATH"
                CUDA_INCLUDE_DIR=$(dirname "$CUDA_RUNTIME_PATH")
                CUDA_ROOT=$(dirname "$CUDA_INCLUDE_DIR")
                export CUDA_HOME="$CUDA_ROOT"
                export CUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT"
                CUDA_FOUND=true
            else
                CUDA_FOUND=false
            fi
        fi
    fi
else
    echo "❌ nvcc not found in PATH"
fi

if [ "$CUDA_FOUND" = true ]; then
    echo "Using CUDA_HOME: $CUDA_HOME"
    
    # Set up environment variables properly
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"
    export CPATH="$CUDA_HOME/include:$CPATH"
    
    # Set architecture for A100 GPUs
    echo "Adding GPU architecture for A100 (sm_80)"
    export TORCH_CUDA_ARCH_LIST="8.0"
    
    # Set compiler path if we're using system CUDA with Conda nvcc
    if [ -n "$NVCC_PATH" ]; then
        echo "Setting NVCC path to Conda's nvcc: $NVCC_PATH"
        export NVCC="$NVCC_PATH"
    fi
    
    # Remove any previous failed compilations
    echo "Clearing previous compilation artifacts..."
    find ~/.cache/torch_extensions -name "fused_adam" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Print detailed diagnostic info for compilation
    echo "CUDA environment variables:"
    echo "- CUDA_HOME=$CUDA_HOME"
    echo "- CUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR"
    echo "- PATH=$PATH"
    echo "- LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
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
        # Export TORCH_EXTENSIONS_DIR to a clean location to ensure fresh builds
        export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
        echo "Setting torch extensions dir to: $TORCH_EXTENSIONS_DIR"
        mkdir -p "$TORCH_EXTENSIONS_DIR"
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
    "reduce_bucket_size": 5e8,
    "prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
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
    "contiguous_memory_optimization": true,
    "number_checkpoints": 1,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  },
  "steps_per_print": 100,
  "wall_clock_breakdown": false,
  "tensorboard": {
    "enabled": true,
    "output_path": "./logs/tensorboard"
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
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
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