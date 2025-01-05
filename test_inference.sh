#!/bin/bash

set -eu

echo "Starting test inference on worker..."

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

echo "HF_TOKEN is${HF_TOKEN:+ set}${HF_TOKEN:-" not set"}"

cd ~/EasyLM

# Run test inference
#python -m EasyLM.test_inference
#!/bin/bash

set -eu

echo "Starting test inference on worker..."

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

echo "HF_TOKEN is${HF_TOKEN:+ set}${HF_TOKEN:-" not set"}"

cd ~/EasyLM

# Run test inference
python -m EasyLM.test_inference \
    --mesh_dim='1,-1,1' \
    --dtype='bf16' \
    --llama.base_model='llama32_1b' \
    --tokenizer='meta-llama/Llama-3.2-1B' \
    --load_checkpoint='base_params_unsharded::/mnt/disk2/llama-3.2-1b.easylm' \
    --load_lora='params::/mnt/disk2/easylm_chkp/1febd23a8f154d748cbc59896981b9aa/checkpoint_15000/streaming_params' \
    --lora_mode=False \
    --input_length=1024 \
    --seq_length=2048 \
    --do_sample=True \
    --top_k=50 \
    --top_p=0.95 \
    --lm_server.port=5009

#    --llama.lora_rank=16 \
#    --llama.lora_alpha=32 \
#    --llama.lora_dropout=0.1 \
#    --llama.lora_attn=true \
#    --llama.lora_mlp=true \
