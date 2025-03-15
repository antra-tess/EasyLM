#!/bin/bash

set -eu

echo "Installing dependencies for Hugging Face training with Gemma..."

# Create a Python virtual environment (optional)
# python -m venv venv
# source venv/bin/activate

# Install transformers, datasets, and other HF libraries
pip install transformers==4.38.0
pip install datasets==2.16.1
pip install accelerate==0.25.0
pip install peft==0.7.1
pip install bitsandbytes==0.41.1  # Still needed for quantization options
pip install wandb
pip install PyYAML

# Install compatible DeepSpeed version for PyTorch 2.1.x
pip install deepspeed==0.9.5 --no-build-isolation

# Install additional dependencies for efficient training
pip install ninja  # For faster compilation
pip install sentencepiece  # For tokenization
pip install tensorboard  # For logging

# Install TRL for RLHF later (if needed)
pip install trl==0.7.6

echo "Dependencies installed successfully!" 