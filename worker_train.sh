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
    --train_dataset.type='json' \
    --train_dataset.text_processor.template="$(cat templates/borg_chat.yaml)" \
    --train_dataset.json_dataset.path="/mnt/disk2/simulect_conversations.json" \
    --train_dataset.json_dataset.seq_length=1024 \
    --train_dataset.json_dataset.batch_size=64 \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=100 \
    --optimizer.adamw_optimizer.lr_decay_steps=1000 \
    --total_steps=8000 \
    --log_freq=50 \
    --save_model_freq=500 \
    --logger.online=true \
    --logger.project='levanter-sft' \
    --logger.entity='antra-cyborgism' \
    --logger.output_dir='params::/mnt/disk2/easylm_chkp'
