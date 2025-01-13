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
    --mesh_dim='1,-1,8' \
    --dtype='bf16' \
    --llama.base_model='llama31_70b' \
    --param_dtype='bf16' \
    --tokenizer='meta-llama/Llama-3.1-70B' \
    --load_checkpoint="base_params::/mnt/$INFER_DISK/llama-3.1-70b" \
    --input_length=768 \
    --seq_length=1536 \
    --do_sample=True \
    --top_k=150 \
    --top_p=0.99 \
    --lm_server.port=5009 \
    --load_lora="base_params::/mnt/$INFER_DISK/easylm_chkp/0f24b03e4bec41f49c9b172a7adf2eea/checkpoint_1000/streaming_params" \
    --lora_mode=True \
    --llama.lora_rank=32 \
    --llama.lora_alpha=64 \
    --llama.lora_dropout=0.1 \
    --llama.lora_attn=true \
    --llama.lora_mlp=false \
    --coordinator_url='http://51.81.181.136:5010'
