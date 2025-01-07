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
    --param_dtype='bf16' \
    --tokenizer='meta-llama/Llama-3.2-1B' \
    --load_checkpoint='base_params_unsharded::/mnt/disk2/trained/simulect8b.easylm' \
    --input_length=1024 \
    --seq_length=2048 \
    --do_sample=True \
    --top_k=50 \
    --top_p=0.95 \
    --lm_server.port=5009
