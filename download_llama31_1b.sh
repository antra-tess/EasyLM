#!/bin/bash

set -eu

echo "Starting Llama-3 1B model download..."

# Check if HF_TOKEN is set
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    exit 1
fi

MODEL_NAME="meta-llama/Llama-3-1b"
OUTPUT_DIR="/mnt/disk2/llama-3.1-1b"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download model files using huggingface-cli
huggingface-cli download --token $HF_TOKEN \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False \
    "$MODEL_NAME"

echo "Download complete to $OUTPUT_DIR"
