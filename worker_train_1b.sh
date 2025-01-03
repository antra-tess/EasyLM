#!/bin/bash

set -eu

echo "Starting LoRA training on worker..."

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

echo "WANDB_API_KEY is${WANDB_API_KEY:+ set}${WANDB_API_KEY:-" not set"}"
echo "HF_TOKEN is${HF_TOKEN:+ set}${HF_TOKEN:-" not set"}"

## Run training
cd ~/EasyLM && python -m EasyLM.models.llama.llama_lora_train \
    --mesh_dim='1,-1,1' \
    --dtype='bf16' \
    --llama.base_model='llama32_1b' \
    --llama.lora_rank=16 \
    --llama.lora_alpha=32 \
    --llama.lora_dropout=0.1 \
    --llama.lora_attn=true \
    --llama.lora_mlp=true \
    --tokenizer="meta-llama/Llama-3.2-1B" \
    --load_checkpoint='base_params::/mnt/disk2/llama-3.2-1b.easylm' \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.template="$(cat templates/llama_chat.yaml)" \
    --train_dataset.huggingface_dataset.name="" \
    --train_dataset.huggingface_dataset.path='tatsu-lab/alpaca' \
    --train_dataset.huggingface_dataset.seq_length=512 \
    --train_dataset.huggingface_dataset.batch_size=64 \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.lr=2e-3 \
    --optimizer.adamw_optimizer.end_lr=5e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=100 \
    --optimizer.adamw_optimizer.lr_decay_steps=3000 \
    --total_steps=15000 \
    --log_freq=50 \
    --save_model_freq=500 \
    --logger.online=true \
    --logger.project='levanter-sft' \
    --logger.entity='antra-cyborgism' \
    --logger.output_dir='/mnt/disk2/easylm_chkp' \
    --checkpointing.save_min_step=0 \
    --checkpointing.save_loss_threshold=999.0
