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

# Check for CUDA installations with priority on matching PyTorch's version
echo "Searching for CUDA installations..."

# First check if we're in a conda environment and prioritize that
CONDA_PREFIX=${CONDA_PREFIX:-""}
if [[ -n "$CONDA_PREFIX" ]]; then
    echo "Active conda environment detected at: $CONDA_PREFIX"
    
    # Check for common CUDA locations within conda environment
    CONDA_CUDA_PATHS=(
        "$CONDA_PREFIX"
        "$CONDA_PREFIX/lib/cuda"
        "$CONDA_PREFIX/pkgs/cuda"
        "$CONDA_PREFIX/pkgs/cuda-toolkit"
    )
    
    # Look for include/cuda_runtime.h in conda environment
    for path in "${CONDA_CUDA_PATHS[@]}"; do
        if [[ -f "$path/include/cuda_runtime.h" ]]; then
            echo "✅ Found cuda_runtime.h in conda environment at $path/include/cuda_runtime.h"
            CUDA_FOUND=true
            SYSTEM_CUDA_HOME="$path"
            break
        elif [[ -f "$path/include/cuda/cuda_runtime.h" ]]; then
            echo "✅ Found cuda_runtime.h in conda environment at $path/include/cuda/cuda_runtime.h"
            CUDA_FOUND=true
            SYSTEM_CUDA_HOME="$path"
            break
        fi
    done
    
    # If we found CUDA in conda environment, use it and skip other checks
    if [[ "$CUDA_FOUND" = true ]]; then
        echo "Using CUDA from conda environment"
    else
        # Look for library paths containing CUDA in conda environment
        echo "Searching for CUDA libraries in conda environment..."
        
        # Find all libcudart.so files in conda environment
        CONDA_CUDA_LIBS=$(find "$CONDA_PREFIX" -name "libcudart.so*" -o -name "libcudart.dylib" 2>/dev/null || echo "")
        
        if [[ -n "$CONDA_CUDA_LIBS" ]]; then
            # Take the first match
            FIRST_LIB=$(echo "$CONDA_CUDA_LIBS" | head -n 1)
            echo "Found CUDA library at: $FIRST_LIB"
            
            # Get the directory containing the library
            LIB_DIR=$(dirname "$FIRST_LIB")
            
            # Go up two levels to get a potential CUDA_HOME
            POTENTIAL_CUDA_HOME=$(dirname "$(dirname "$LIB_DIR")")
            
            if [[ -d "$POTENTIAL_CUDA_HOME/include" ]]; then
                echo "Found potential CUDA home at: $POTENTIAL_CUDA_HOME"
                
                # Check if cuda_runtime.h exists in this location
                if [[ -f "$POTENTIAL_CUDA_HOME/include/cuda_runtime.h" ]]; then
                    echo "✅ Found cuda_runtime.h at $POTENTIAL_CUDA_HOME/include/cuda_runtime.h"
                    CUDA_FOUND=true
                    SYSTEM_CUDA_HOME="$POTENTIAL_CUDA_HOME"
                elif [[ -f "$POTENTIAL_CUDA_HOME/include/cuda/cuda_runtime.h" ]]; then
                    echo "✅ Found cuda_runtime.h at $POTENTIAL_CUDA_HOME/include/cuda/cuda_runtime.h"
                    CUDA_FOUND=true
                    SYSTEM_CUDA_HOME="$POTENTIAL_CUDA_HOME"
                fi
            fi
        fi
    fi
fi

# If we didn't find CUDA in conda environment, check system paths
if [[ "$CUDA_FOUND" != true ]]; then
    # Get PyTorch's CUDA version
    TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'NA')")
    echo "PyTorch CUDA version: $TORCH_CUDA_VERSION"
    
    CUDA_SYSTEM_PATHS=(
        "/usr/local/cuda"
        "/usr/local/cuda-12.4"
        "/usr/local/cuda-12.1"
        "/usr/local/cuda-11.8"
        "/usr/cuda"
        "/usr/cuda-12.4"
        "/usr/cuda-12.1"
        "/usr/cuda-11.8"
        "/opt/cuda"
        "/opt/cuda-12.4"
        "/opt/cuda-12.1"
        "/opt/cuda-11.8"
    )

    # Check additional package manager locations (conda, pip, apt)
    PYTHON_PATH=$(which python)
    PYTHON_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

    ADDITIONAL_CUDA_PATHS=(
        "${PYTHON_SITE_PACKAGES}/torch/lib"
        "${CONDA_PREFIX}/lib/python*/site-packages/torch/lib"
        "${CONDA_PREFIX}/pkgs/*/lib"
    )

    # Combine all paths with PyTorch CUDA version paths first
    if [[ "$TORCH_CUDA_VERSION" != "NA" ]]; then
        ALL_CUDA_PATHS=(
            "/usr/local/cuda-${TORCH_CUDA_VERSION}"
            "/usr/cuda-${TORCH_CUDA_VERSION}"
            "/opt/cuda-${TORCH_CUDA_VERSION}"
            "${ADDITIONAL_CUDA_PATHS[@]}"
            "${CUDA_SYSTEM_PATHS[@]}"
        )
    else
        ALL_CUDA_PATHS=(
            "${CUDA_SYSTEM_PATHS[@]}"
            "${ADDITIONAL_CUDA_PATHS[@]}"
        )
    fi

    for path in "${ALL_CUDA_PATHS[@]}"; do
        # Handle glob patterns
        for expanded_path in $path; do
            if [ -d "$expanded_path" ]; then
                echo "Found potential CUDA installation at: $expanded_path"
                
                # Look for cuda_runtime.h in include or include/cuda dirs
                if [ -f "$expanded_path/include/cuda_runtime.h" ]; then
                    echo "✅ Found cuda_runtime.h at $expanded_path/include/cuda_runtime.h"
                    CUDA_FOUND=true
                    SYSTEM_CUDA_HOME="$expanded_path"
                    break 2
                elif [ -f "$expanded_path/include/cuda/cuda_runtime.h" ]; then
                    echo "✅ Found cuda_runtime.h at $expanded_path/include/cuda/cuda_runtime.h"
                    CUDA_FOUND=true
                    SYSTEM_CUDA_HOME="$expanded_path"
                    break 2
                fi
            fi
        done
    done
fi

# Additional fallback: check for NVCC in PATH
if [[ "$CUDA_FOUND" != true ]]; then
    NVCC_PATH=$(which nvcc 2>/dev/null || echo "")
    if [[ -n "$NVCC_PATH" ]]; then
        echo "Found nvcc at: $NVCC_PATH"
        NVCC_VERSION=$($NVCC_PATH -V 2>&1 | grep "release" | awk '{print $5}' | awk -F, '{print $1}')
        echo "NVCC version: $NVCC_VERSION"
        
        # Get the base directory (typically 2 levels up from bin/nvcc)
        NVCC_BASE=$(dirname "$(dirname "$NVCC_PATH")")
        echo "NVCC base directory: $NVCC_BASE"
        
        if [[ -f "$NVCC_BASE/include/cuda_runtime.h" ]]; then
            echo "✅ Found cuda_runtime.h at $NVCC_BASE/include/cuda_runtime.h"
            CUDA_FOUND=true
            SYSTEM_CUDA_HOME="$NVCC_BASE"
        fi
    fi
fi

# If we found PyTorch with CUDA but couldn't match it, disable custom CUDA ops
if [[ "$TORCH_CUDA_VERSION" != "NA" && "$CUDA_FOUND" = false ]]; then
    echo "⚠️ Could not find matching CUDA installation for PyTorch's CUDA $TORCH_CUDA_VERSION"
    echo "Will disable DeepSpeed CUDA extensions and use PyTorch optimizers"
    
    # Use these environment variables to disable CUDA ops compilation
    export DS_BUILD_OPS=0 
    export DS_BUILD_FUSED_ADAM=0
    export DS_BUILD_FUSED_LAMB=0
    export TORCH_EXTENSIONS_DIR="${PWD}/.torch_extensions_cpu_only"
    mkdir -p "$TORCH_EXTENSIONS_DIR"
    
    # Create DeepSpeed config without requiring CUDA extensions
    cat > ds_config.json << EOF
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "none"
        },
        "contiguous_gradients": true,
        "overlap_comm": true,
        "allgather_partitions": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "prefetch_bucket_size": 5e8,
        "sub_group_size": 1e9,
        "round_robin_gradients": true
    },
    "activation_checkpointing": {
        "partition_activations": true,
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
    },
    "memory_breakdown": true
}
EOF

    # Run using torch.distributed directly instead of deepspeed launcher
    echo "Using torch.distributed.run instead of deepspeed launcher to avoid CUDA extension compilation"
    python -m torch.distributed.run \
        --nproc_per_node=2 \
        gemma_sft_train.py \
        --model_name_or_path "google/gemma-3-27b-pt" \
        --dataset_path $DATA_PATH \
        --template_path $TEMPLATE_PATH \
        --max_seq_length 1024 \
        --lora_rank 48 \
        --lora_alpha 96 \
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
        --optim adamw_torch \
        --deepspeed ds_config.json \
        --load_in_8bit False \
        --load_in_4bit False
else
    # We found a working CUDA installation
    if [ "$CUDA_FOUND" = true ]; then
        echo "Using CUDA_HOME: $SYSTEM_CUDA_HOME"
        
        # Set up environment variables properly
        export CUDA_HOME="$SYSTEM_CUDA_HOME"
        export CUDA_TOOLKIT_ROOT_DIR="$SYSTEM_CUDA_HOME"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"
        export CPATH="$CUDA_HOME/include:$CPATH"
        
        # Set architecture for A100 GPUs
        echo "Adding GPU architecture for A100 (sm_80)"
        export TORCH_CUDA_ARCH_LIST="8.0"
        
        # Remove any previous failed compilations
        echo "Clearing previous compilation artifacts..."
        find ~/.cache/torch_extensions -name "fused_adam" -type d -exec rm -rf {} + 2>/dev/null || true
        
        # Set a clean extensions directory
        export TORCH_EXTENSIONS_DIR="${PWD}/.torch_extensions"
        mkdir -p "$TORCH_EXTENSIONS_DIR"
        
        # Create standard DeepSpeed config with FusedAdam
        cat > ds_config.json << EOF
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "none"
        },
        "contiguous_gradients": true,
        "overlap_comm": true,
        "allgather_partitions": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "prefetch_bucket_size": 5e8,
        "sub_group_size": 1e9,
        "round_robin_gradients": true
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
    },
    "memory_breakdown": true
}
EOF

        # Run using deepspeed
        deepspeed --num_gpus=2 gemma_sft_train.py \
            --model_name_or_path "google/gemma-3-27b-pt" \
            --dataset_path $DATA_PATH \
            --template_path $TEMPLATE_PATH \
            --max_seq_length 1024 \
            --lora_rank 48 \
            --lora_alpha 96 \
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
            --deepspeed ds_config.json \
            --load_in_8bit False \
            --load_in_4bit False
    else
        echo "⚠️ Could not find any CUDA installation with cuda_runtime.h"
        echo "Switching to torch.distributed with CPU optimizer"
        
        # Use these environment variables to disable CUDA ops compilation
        export DS_BUILD_OPS=0 
        export DS_BUILD_FUSED_ADAM=0
        export DS_BUILD_FUSED_LAMB=0
        
        # Create minimal DeepSpeed config
        cat > ds_config.json << EOF
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "none"
        },
        "contiguous_gradients": true,
        "overlap_comm": true,
        "allgather_partitions": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "prefetch_bucket_size": 5e8
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": true,
        "number_checkpoints": 1,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    },
    "steps_per_print": 100,
    "wall_clock_breakdown": false
}
EOF

        # Run using torch.distributed
        python -m torch.distributed.run \
            --nproc_per_node=2 \
            gemma_sft_train.py \
            --model_name_or_path "google/gemma-3-27b-pt" \
            --dataset_path $DATA_PATH \
            --template_path $TEMPLATE_PATH \
            --max_seq_length 1024 \
            --lora_rank 48 \
            --lora_alpha 96 \
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
            --optim adamw_torch \
            --deepspeed ds_config.json \
            --load_in_8bit False \
            --load_in_4bit False
    fi
fi

echo "Training complete!" 