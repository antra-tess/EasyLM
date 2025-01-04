#!/bin/bash

set -eu

echo "Starting serving on worker..."

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

echo "WANDB_API_KEY is${WANDB_API_KEY:+ set}${WANDB_API_KEY:-" not set"}"
echo "HF_TOKEN is${HF_TOKEN:+ set}${HF_TOKEN:-" not set"}"

cd ~/EasyLM

# Run serving
python -m EasyLM.models.llama.llama_serve \
    --mesh_dim='1,-1,1' \
    --dtype='bf16' \
    --llama.base_model='llama32_1b' \
    --tokenizer="meta-llama/Llama-3.2-1B" \
    --load_checkpoint='base_params_unsharded::/mnt/disk2/llama-3.2-1b.easylm' \
    --input_length=1024 \
    --seq_length=2048 \
    --do_sample=True \
    --top_k=50 \
    --top_p=0.95 \
    --lm_server.port=5009 \
    --lm_server.batch_size=1 \
    --jax_distributed.initialize_jax_distributed=false \
    --jax_distributed.coordinator_address=''
