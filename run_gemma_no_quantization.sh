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

# Initialize variables
CUDA_FOUND=false
# Get PyTorch's CUDA version upfront
TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'NA')")
echo "PyTorch CUDA version: $TORCH_CUDA_VERSION"

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
            
            # Extract CUDA version from the library path if possible
            if [[ "$FIRST_LIB" =~ libcudart\.so\.([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
                CUDA_MAJOR="${BASH_REMATCH[1]}"
                CUDA_MINOR="${BASH_REMATCH[2]}"
                echo "Detected CUDA version from library: $CUDA_MAJOR.$CUDA_MINOR"
            fi
            
            # Get the directory containing the library
            LIB_DIR=$(dirname "$FIRST_LIB")
            
            # Find possible include directories - look in several places
            POSSIBLE_INCLUDE_DIRS=(
                "$CONDA_PREFIX/include"
                "$CONDA_PREFIX/targets/x86_64-linux/include"
                "${LIB_DIR}/../include"
                "${LIB_DIR}/../../include"
                "$CONDA_PREFIX/lib/python*/site-packages/torch/include"
            )
            
            for include_dir in "${POSSIBLE_INCLUDE_DIRS[@]}"; do
                # Handle glob patterns
                for expanded_dir in $include_dir; do
                    if [[ -d "$expanded_dir" ]]; then
                        echo "Checking potential include directory: $expanded_dir"
                        if [[ -f "$expanded_dir/cuda_runtime.h" ]]; then
                            echo "✅ Found cuda_runtime.h at $expanded_dir/cuda_runtime.h"
                            CUDA_FOUND=true
                            SYSTEM_CUDA_HOME=$(dirname "$expanded_dir")
                            break 2
                        elif [[ -f "$expanded_dir/cuda/cuda_runtime.h" ]]; then
                            echo "✅ Found cuda_runtime.h at $expanded_dir/cuda/cuda_runtime.h"
                            CUDA_FOUND=true
                            SYSTEM_CUDA_HOME=$(dirname "$expanded_dir")
                            break 2
                        fi
                    fi
                done
            done
            
            # If we still haven't found cuda_runtime.h, try recursive find
            if [[ "$CUDA_FOUND" != true ]]; then
                echo "Searching for cuda_runtime.h in conda environment..."
                CUDA_HEADER=$(find "$CONDA_PREFIX" -name "cuda_runtime.h" -type f 2>/dev/null | head -n 1)
                if [[ -n "$CUDA_HEADER" ]]; then
                    echo "✅ Found cuda_runtime.h at $CUDA_HEADER"
                    CUDA_FOUND=true
                    CUDA_INCLUDE_DIR=$(dirname "$CUDA_HEADER")
                    SYSTEM_CUDA_HOME=$(dirname "$CUDA_INCLUDE_DIR")
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
    
    # Set default additional args if not defined
    ADDITIONAL_ARGS="--optim adamw_torch ${ADDITIONAL_ARGS:-""}"
    
    # Check Python and CUDA diagnostics
    echo "Python diagnostics:"
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
    
    # Create DeepSpeed config without requiring CUDA extensions
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
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9
    },
    "gradient_clipping": 1.0,
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 2,
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
        "contiguous_memory_optimization": true
    }
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
        --per_device_train_batch_size 12 \
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
        --deepspeed ds_config.json \
        --load_in_8bit False \
        --load_in_4bit False \
        --use_flash_attn True \
        $ADDITIONAL_ARGS
else
    # We found a working CUDA installation
    if [ "$CUDA_FOUND" = true ]; then
        echo "Using CUDA_HOME: $SYSTEM_CUDA_HOME"
        
        # Get PyTorch's CUDA version if not already set
        if [[ -z "$TORCH_CUDA_VERSION" ]]; then
            TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'NA')")
            echo "PyTorch CUDA version: $TORCH_CUDA_VERSION"
        fi
        
        # Get the CUDA version from our found CUDA installation if possible
        if [[ -z "$CUDA_MAJOR" || -z "$CUDA_MINOR" ]]; then
            # Try to find nvcc in multiple locations
            NVCC_PATHS=(
                "$SYSTEM_CUDA_HOME/bin/nvcc"
                "$(which nvcc 2>/dev/null)"
            )
            
            for NVCC_PATH in "${NVCC_PATHS[@]}"; do
                if [[ -x "$NVCC_PATH" ]]; then
                    echo "Found NVCC at: $NVCC_PATH"
                    NVCC_VERSION=$("$NVCC_PATH" -V 2>&1 | grep "release" | awk '{print $5}' | awk -F, '{print $1}')
                    echo "NVCC version: $NVCC_VERSION"
                    
                    # Parse major.minor
                    if [[ "$NVCC_VERSION" =~ ([0-9]+)\.([0-9]+) ]]; then
                        CUDA_MAJOR="${BASH_REMATCH[1]}"
                        CUDA_MINOR="${BASH_REMATCH[2]}"
                        echo "Parsed CUDA version: $CUDA_MAJOR.$CUDA_MINOR"
                        break
                    fi
                fi
            done
        fi
        
        # If still no CUDA version, try libcudart.so versioning
        if [[ -z "$CUDA_MAJOR" || -z "$CUDA_MINOR" ]]; then
            echo "Trying to determine CUDA version from libcudart.so..."
            CUDART_PATHS=$(find "$CONDA_PREFIX" -name "libcudart.so*" 2>/dev/null || echo "")
            if [[ -n "$CUDART_PATHS" ]]; then
                FIRST_LIB=$(echo "$CUDART_PATHS" | head -n 1)
                if [[ "$FIRST_LIB" =~ libcudart\.so\.([0-9]+)\.([0-9]+) ]]; then
                    CUDA_MAJOR="${BASH_REMATCH[1]}"
                    CUDA_MINOR="${BASH_REMATCH[2]}"
                    echo "CUDA version from library: $CUDA_MAJOR.$CUDA_MINOR"
                fi
            fi
        fi
        
        # Check if the found CUDA version matches PyTorch's CUDA version
        CUDA_VERSION_MISMATCH=false
        if [[ "$TORCH_CUDA_VERSION" != "NA" && -n "$CUDA_MAJOR" && -n "$CUDA_MINOR" ]]; then
            FOUND_CUDA_VERSION="$CUDA_MAJOR.$CUDA_MINOR"
            if [[ "$FOUND_CUDA_VERSION" != "$TORCH_CUDA_VERSION" ]]; then
                echo "⚠️ Warning: Found CUDA version ($FOUND_CUDA_VERSION) doesn't match PyTorch's CUDA version ($TORCH_CUDA_VERSION)"
                CUDA_VERSION_MISMATCH=true
            else
                echo "✅ Found CUDA version matches PyTorch CUDA version: $FOUND_CUDA_VERSION"
            fi
        fi
        
        # Set up environment variables properly
        export CUDA_HOME="$SYSTEM_CUDA_HOME"
        export CUDA_TOOLKIT_ROOT_DIR="$SYSTEM_CUDA_HOME"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"
        export CPATH="$CUDA_HOME/include:$CPATH"
        
        # Check if nvcc exists in the expected location, if not try to find it in PATH
        if [[ ! -f "$CUDA_HOME/bin/nvcc" ]]; then
            NVCC_PATH=$(which nvcc 2>/dev/null || echo "")
            if [[ -n "$NVCC_PATH" ]]; then
                echo "NVCC not found in CUDA_HOME/bin, but found in PATH: $NVCC_PATH"
                # Get the parent directory of nvcc's directory
                NVCC_BIN_DIR=$(dirname "$NVCC_PATH")
                NVCC_PARENT=$(dirname "$NVCC_BIN_DIR")
                echo "Adjusting CUDA_HOME to point to parent of nvcc: $NVCC_PARENT"
                
                # Directly change CUDA_HOME to the parent directory of nvcc
                export CUDA_HOME="$NVCC_PARENT"
                export CUDA_TOOLKIT_ROOT_DIR="$NVCC_PARENT"
                
                # Update PATH and LD_LIBRARY_PATH for the new CUDA_HOME
                export PATH="$CUDA_HOME/bin:$PATH"
                export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"
                
                # If the include directory has cuda_runtime.h, add it to CPATH
                if [[ -f "$CUDA_HOME/include/cuda_runtime.h" ]]; then
                    echo "Found cuda_runtime.h in new CUDA_HOME/include"
                    export CPATH="$CUDA_HOME/include:$CPATH"
                elif [[ -f "$SYSTEM_CUDA_HOME/include/cuda_runtime.h" ]]; then
                    # If we can't find cuda_runtime.h in the new CUDA_HOME, keep using the old one for includes
                    echo "Using original SYSTEM_CUDA_HOME for include files"
                    export CPATH="$SYSTEM_CUDA_HOME/include:$CPATH"
                fi
                
                echo "Updated CUDA_HOME to: $CUDA_HOME"
            fi
        fi
        
        # Add PyTorch's CUDA include paths to help DeepSpeed find the right headers
        TORCH_CUDA_INCLUDE=$(python -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent/'include')" 2>/dev/null || echo "")
        if [[ -n "$TORCH_CUDA_INCLUDE" && -d "$TORCH_CUDA_INCLUDE" ]]; then
            echo "Adding PyTorch's CUDA include path: $TORCH_CUDA_INCLUDE"
            export CPATH="$TORCH_CUDA_INCLUDE:$CPATH"
        fi
        
        # Find any cuda_runtime.h in PyTorch's installation
        TORCH_CUDA_RUNTIME=$(find $(python -c "import torch; print(torch.__path__[0])") -name "cuda_runtime.h" 2>/dev/null | head -n 1)
        if [[ -n "$TORCH_CUDA_RUNTIME" ]]; then
            TORCH_CUDA_INCLUDE_DIR=$(dirname "$TORCH_CUDA_RUNTIME")
            echo "Found cuda_runtime.h in PyTorch installation: $TORCH_CUDA_RUNTIME"
            echo "Adding to CPATH: $TORCH_CUDA_INCLUDE_DIR"
            export CPATH="$TORCH_CUDA_INCLUDE_DIR:$CPATH"
        fi
        
        # Set architecture for A100 GPUs
        echo "Adding GPU architecture for A100 (sm_80)"
        export TORCH_CUDA_ARCH_LIST="8.0"
        
        # Special handling for CUDA version mismatch
        if [[ "$CUDA_VERSION_MISMATCH" = true ]]; then
            echo "Attempting to work around CUDA version mismatch..."
            
            # Try to locate system CUDA that might match PyTorch's version
            if [[ -d "/usr/local/cuda-$TORCH_CUDA_VERSION" ]]; then
                echo "Found system CUDA matching PyTorch version at /usr/local/cuda-$TORCH_CUDA_VERSION"
                export CUDA_HOME="/usr/local/cuda-$TORCH_CUDA_VERSION"
                export CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-$TORCH_CUDA_VERSION"
                export PATH="/usr/local/cuda-$TORCH_CUDA_VERSION/bin:$PATH"
                export LD_LIBRARY_PATH="/usr/local/cuda-$TORCH_CUDA_VERSION/lib64:/usr/local/cuda-$TORCH_CUDA_VERSION/lib:$LD_LIBRARY_PATH"
                export CPATH="/usr/local/cuda-$TORCH_CUDA_VERSION/include:$CPATH"
            else
                # Force DeepSpeed to skip CUDA version check
                export DS_SKIP_CUDA_CHECK=1
                export NVCC_PREPEND_FLAGS="-ccbin=$SYSTEM_CUDA_HOME/bin"
                # Disable CUDA extensions entirely if version mismatch is too severe
                if [[ "$FOUND_CUDA_VERSION" == "11."* && "$TORCH_CUDA_VERSION" == "12."* ]]; then
                    echo "⚠️ Major version mismatch (CUDA 11.x vs 12.x). Using CPU optimizers."
                    export DS_BUILD_OPS=0 
                    export DS_BUILD_FUSED_ADAM=0
                    export DS_BUILD_FUSED_LAMB=0
                    # Force PyTorch's own implementation
                    ADDITIONAL_ARGS="--optim adamw_torch"
                fi
            fi
        fi
        
        # Remove any previous failed compilations
        echo "Clearing previous compilation artifacts..."
        find ~/.cache/torch_extensions -name "fused_adam" -type d -exec rm -rf {} + 2>/dev/null || true
        find ~/.cache/pip -name "deepspeed" -type d -exec rm -rf {} + 2>/dev/null || true
        
        # Set a clean extensions directory
        export TORCH_EXTENSIONS_DIR="${PWD}/.torch_extensions_$(date +%s)"
        mkdir -p "$TORCH_EXTENSIONS_DIR"
        
        # Print debug info
        echo "CUDA environment variables:"
        echo "CUDA_HOME=$CUDA_HOME"
        echo "CUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR"
        echo "CPATH=$CPATH"
        echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
        echo "DS_SKIP_CUDA_CHECK=$DS_SKIP_CUDA_CHECK"
        echo "TORCH_CUDA_VERSION=$TORCH_CUDA_VERSION"
        
        # Final check for necessary CUDA components
        echo "Running final CUDA component checks:"
        echo "- NVCC location: $(which nvcc 2>/dev/null || echo 'Not found')"
        if [ -x "$(which nvcc 2>/dev/null)" ]; then
            echo "- NVCC version: $(nvcc --version | grep "release" || echo 'Cannot determine')"
        fi
        echo "- CUDA includes: $(find $CPATH -name "cuda_runtime.h" 2>/dev/null | head -n1 || echo 'Not found')"
        echo "- CUDA libraries: $(find $LD_LIBRARY_PATH -name "libcudart.so*" 2>/dev/null | head -n1 || echo 'Not found')"

        # Set additional options for deepspeed
        export TORCH_DISTRIBUTED_DEBUG=DETAIL
        export CUDA_DEVICE_DEBUG=1

        # Configure DS_BUILD vars to avoid unnecessary compilation issues
        # If we can't find nvcc in a reliable location, disable building the ops
        if [[ ! -x "$(which nvcc 2>/dev/null)" ]]; then
            echo "⚠️ NVCC not found in PATH, disabling DeepSpeed CUDA ops compilation"
            export DS_BUILD_OPS=0
            export DS_BUILD_FUSED_ADAM=0
            export DS_BUILD_FUSED_LAMB=0
            ADDITIONAL_ARGS="--optim adamw_torch ${ADDITIONAL_ARGS:-""}"
        else
            # If NVCC is found, ensure we use it by explicitly setting NVCC_PREPEND_FLAGS
            NVCC_PATH=$(which nvcc)
            echo "✅ NVCC found at $NVCC_PATH, enabling DeepSpeed ops compilation"
            
            # Extract exact CUDA version from nvcc
            NVCC_VERSION_STR=$($NVCC_PATH --version | grep "release" | awk '{print $5}' | awk -F, '{print $1}')
            echo "NVCC reports CUDA version: $NVCC_VERSION_STR"
            
            # Set exact CUDA version from nvcc output to help with compilation
            export NVCC_CUDA_VERSION="$NVCC_VERSION_STR"
            export CUDA_VERSION="$NVCC_VERSION_STR"
            
            # Find a suitable C compiler (gcc)
            GCC_PATH=$(which gcc 2>/dev/null || echo "")
            if [[ -n "$GCC_PATH" ]]; then
                echo "Found GCC at: $GCC_PATH"
                GCC_VERSION=$($GCC_PATH --version | head -n1)
                echo "GCC version: $GCC_VERSION"
                
                # Set compiler paths for CUDA
                export CC="$GCC_PATH"
                export CXX="$(which g++ 2>/dev/null || echo "")"
                export CUDAHOSTCXX="$CXX"
                echo "Set CC=$CC"
                echo "Set CXX=$CXX"
                echo "Set CUDAHOSTCXX=$CUDAHOSTCXX"
            fi
            
            # Set NVCC options to ensure proper compilation
            export NVCC_PREPEND_FLAGS="-ccbin=$(dirname $NVCC_PATH)"
            export DS_BUILD_OPS=1
            
            echo "Setting CUDA environment variables for building:"
            echo "NVCC_CUDA_VERSION=$NVCC_CUDA_VERSION"
            echo "CUDA_VERSION=$CUDA_VERSION"
            echo "NVCC_PREPEND_FLAGS=$NVCC_PREPEND_FLAGS"
            
            # Create a symlink from nvcc to a location DeepSpeed will look for it
            if [[ ! -f "$CUDA_HOME/bin/nvcc" && "$NVCC_PATH" != "$CUDA_HOME/bin/nvcc" ]]; then
                echo "Creating directory structure in CUDA_HOME ($CUDA_HOME) if needed"
                mkdir -p "$CUDA_HOME/bin"
                echo "Creating symlink from $NVCC_PATH to $CUDA_HOME/bin/nvcc"
                ln -sf "$NVCC_PATH" "$CUDA_HOME/bin/nvcc" 2>/dev/null || echo "Failed to create symlink (permission issue)"
            fi
        fi
        
        # Create standard DeepSpeed config with FusedAdam
        cat > ds_config.json << EOF
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": false
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": false
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 5e9
    },
    "gradient_clipping": 1.0,
    "train_batch_size": 48,
    "train_micro_batch_size_per_gpu": 12,
    "gradient_accumulation_steps": 2,
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
        "contiguous_memory_optimization": true
    }
}
EOF

        # Set default additional args if not defined
        ADDITIONAL_ARGS=${ADDITIONAL_ARGS:-""}

        # Function to run torch.distributed.run as a fallback
        run_with_torch_distributed() {
            echo "Falling back to torch.distributed.run with CPU optimizer..."
            
            # Disable CUDA ops when falling back
            export DS_BUILD_OPS=0 
            export DS_BUILD_FUSED_ADAM=0
            export DS_BUILD_FUSED_LAMB=0
            
            # Set default additional args to use PyTorch optimizer
            FALLBACK_ARGS="--optim adamw_torch ${ADDITIONAL_ARGS}"
            
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
                --per_device_train_batch_size 12 \
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
                --deepspeed ds_config.json \
                --load_in_8bit False \
                --load_in_4bit False \
                --use_flash_attn True \
                $FALLBACK_ARGS
        }

        # Try running with deepspeed first
        echo "Attempting to run with deepspeed..."
        deepspeed --num_gpus=2 gemma_sft_train.py \
            --model_name_or_path "google/gemma-3-27b-pt" \
            --dataset_path $DATA_PATH \
            --template_path $TEMPLATE_PATH \
            --max_seq_length 1024 \
            --lora_rank 48 \
            --lora_alpha 96 \
            --lora_dropout 0.05 \
            --per_device_train_batch_size 12 \
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
            --deepspeed ds_config.json \
            --load_in_8bit False \
            --load_in_4bit False \
            --use_flash_attn True \
            $ADDITIONAL_ARGS

        # If deepspeed fails, try the fallback
        if [ $? -ne 0 ]; then
            echo "DeepSpeed failed with exit code $?, trying fallback method..."
            run_with_torch_distributed
        fi
    else
        echo "⚠️ Could not find any CUDA installation with cuda_runtime.h"
        echo "Switching to torch.distributed with CPU optimizer"
        
        # Use these environment variables to disable CUDA ops compilation
        export DS_BUILD_OPS=0 
        export DS_BUILD_FUSED_ADAM=0
        export DS_BUILD_FUSED_LAMB=0
        
        # Set default additional args if not defined
        ADDITIONAL_ARGS="--optim adamw_torch ${ADDITIONAL_ARGS:-""}"
        
        # Check Python and CUDA diagnostics
        echo "Python diagnostics:"
        python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
        
        # Create minimal DeepSpeed config
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
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9
    },
    "gradient_clipping": 1.0,
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 2,
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
        "contiguous_memory_optimization": true
    }
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
            --per_device_train_batch_size 12 \
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
            --deepspeed ds_config.json \
            --load_in_8bit False \
            --load_in_4bit False \
            --use_flash_attn True \
            $ADDITIONAL_ARGS
    fi
fi

echo "Training complete!" 