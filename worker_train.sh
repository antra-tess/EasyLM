#!/bin/bash

set -eu

echo "Starting training on worker..."

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

echo "WANDB_API_KEY is${WANDB_API_KEY:+ set}${WANDB_API_KEY:-" not set"}"
echo "HF_TOKEN is${HF_TOKEN:+ set}${HF_TOKEN:-" not set"}"

## Run training
cd ~/EasyLM && python -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,1' \
    --dtype='bf16' \
    --llama.base_model='llama31_8b' \
    --tokenizer="meta-llama/Meta-Llama-3.1-8B" \
    --load_checkpoint='params::/mnt/disk2/llama-3.1-8b' \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.template="""
sequence:
  - no_loss: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>"
  - no_loss: "{instruction} {input}<|eot_id|>"
  - no_loss: "<|start_header_id|>assistant<|end_header_id|>"
  - with_loss: "{output}<|eot_id|>"
""" \
    --train_dataset.huggingface_dataset.name="" \
    --train_dataset.huggingface_dataset.path='tatsu-lab/alpaca' \
    --train_dataset.huggingface_dataset.seq_length=2048 \
    --train_dataset.huggingface_dataset.batch_size=64 \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=100 \
    --optimizer.adamw_optimizer.lr_decay_steps=1000 \
    --total_steps=2000 \
    --log_freq=50 \
    --save_model_freq=0 \
    --logger.online=true \
    --logger.project='levanter-sft' \
    --logger.entity='antra-cyborgism' \
    --logger.output_dir='params::/mnt/disk2/easylm_chkp'
