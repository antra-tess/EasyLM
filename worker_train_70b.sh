#!/bin/bash

set -eu

echo "Starting 70B training on worker..."

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

echo "WANDB_API_KEY is${WANDB_API_KEY:+ set}${WANDB_API_KEY:-" not set"}"
echo "HF_TOKEN is${HF_TOKEN:+ set}${HF_TOKEN:-" not set"}"

## Run training
cd ~/EasyLM && python -m EasyLM.models.llama.llama_lora_train \
    --mesh_dim='1,-1,8' \
    --dtype='bf16' \
    --llama.base_model='llama31_70b' \
    --tokenizer="meta-llama/Meta-Llama-3.1-70B" \
    --load_checkpoint='base_params::gs://finetune70b/llama-3.1-70b' \
    --train_dataset.type='json' \
    --train_dataset.text_processor.template="$(cat templates/borg_chat.yaml)" \
    --train_dataset.json_dataset.path="/mnt/disk2/simulect_conversations.jsonl" \
    --train_dataset.json_dataset.seq_length=1024 \
    --train_dataset.json_dataset.batch_size=64 \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.lr=1e-4 \
    --llama.scan_attention=true \
    --llama.scan_query_chunk_size=1024 \
    --llama.scan_key_chunk_size=1024 \
    --llama.lora_rank=32 \
    --llama.lora_alpha=64 \
    --llama.lora_dropout=0.1 \
    --llama.lora_attn=true \
    --llama.lora_mlp=false \
    --optimizer.adamw_optimizer.end_lr=1e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=500 \
    --optimizer.adamw_optimizer.lr_decay_steps=5000 \
    --total_steps=4000 \
    --log_freq=50 \
    --save_model_freq=125 \
    --logger.online=true \
    --logger.project='levanter-sft' \
    --logger.entity='antra-cyborgism' \
    --logger.output_dir='gs://finetune70b/easylm_checkpoints_70b'