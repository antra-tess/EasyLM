#!/bin/bash

set -eu

echo "Starting training on worker..."

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

echo "WANDB_API_KEY is${WANDB_API_KEY:+ set}${WANDB_API_KEY:-" not set"}"

# Run training
cd ~/EasyLM && python -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,1' \
    --llama.base_model='llama3_8b' \
    --load_checkpoint='params::gs://finetune70b/llama-3-8b/llama-3-8b' \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='[instruction+input],output' \
    --train_dataset.huggingface_dataset.name="" \
    --train_dataset.huggingface_dataset.path='tatsu-lab/alpaca' \
    --logger.online=true \
    --logger.project='levanter-sft' \
    --logger.entity='antra-tess'
