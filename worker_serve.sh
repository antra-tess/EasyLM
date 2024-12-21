#!/bin/bash

set -eu

echo "Starting serving on worker..."

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

echo "WANDB_API_KEY is${WANDB_API_KEY:+ set}${WANDB_API_KEY:-" not set"}"
echo "HF_TOKEN is${HF_TOKEN:+ set}${HF_TOKEN:-" not set"}"

cd ~/EasyLM

# Run serving
cd python -m EasyLM.models.llama.llama_serve \
    --mesh_dim='1,-1,1' \
    --dtype='bf16' \
    --llama.base_model='llama3_8b' \
    --tokenizer="meta-llama/Meta-Llama-3-8B" \
    --load_checkpoint='params::gs://finetune70b/easylm_checkpoints/streaming_params' \
    --input_length=1024 \
    --seq_length=2048 \
    --do_sample=True \
    --top_k=50 \
    --top_p=0.95 \
    --temperature=0.8 \
    --lm_server.port=5009 \
    --lm_server.batch_size=1
