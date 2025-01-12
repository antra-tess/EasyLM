#!/bin/bash

set -eu

echo "Starting serving on worker..."

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

echo "WANDB_API_KEY is${WANDB_API_KEY:+ set}${WANDB_API_KEY:-" not set"}"
echo "HF_TOKEN is${HF_TOKEN:+ set}${HF_TOKEN:-" not set"}"

cd ~/EasyLM

# Run worker client
python -m EasyLM.worker_client \
    --mesh_dim='1,-1,1' \
    --dtype='bf16' \
    --llama.base_model='llama32_1b' \
    --param_dtype='bf16' \
    --tokenizer='meta-llama/Llama-3.2-1B' \
    --load_checkpoint='base_params_unsharded::/mnt/disk3/llama-3.2-1b.easylm' \
    --input_length=1024 \
    --seq_length=2048 \
    --do_sample=True \
    --top_k=150 \
    --top_p=0.99 \
    --lm_server.port=5009 \
    --load_lora='base_params_unsharded::/mnt/disk3/easylm_chkp/30bf244bf80c464d854f0c1addf55409/checkpoint_1000/streaming_params' \
    --lora_mode=True \
    --llama.lora_rank=16 \
    --llama.lora_alpha=32 \
    --llama.lora_dropout=0.1 \
    --llama.lora_attn=true \
    --llama.lora_mlp=false \
    --coordinator_url='http://51.81.181.136:5010'
