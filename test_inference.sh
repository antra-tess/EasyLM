#!/bin/bash

set -eu

echo "Starting test inference on worker..."

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

echo "HF_TOKEN is${HF_TOKEN:+ set}${HF_TOKEN:-" not set"}"

cd ~/EasyLM

# Run test inference
python -m EasyLM.test_inference
#!/bin/bash

set -eu

echo "Starting test inference on worker..."

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

echo "HF_TOKEN is${HF_TOKEN:+ set}${HF_TOKEN:-" not set"}"

cd ~/EasyLM

# Run test inference with parameters from environment
python -m EasyLM.test_inference \
    --mesh_dim="${MESH_DIM:-1,-1,1}" \
    --dtype="${DTYPE:-bf16}" \
    --llama.base_model="${MODEL_NAME:-llama32_1b}" \
    --tokenizer="${TOKENIZER:-meta-llama/Llama-3.2-1B}" \
    --load_checkpoint="base_params_unsharded::${MODEL_PATH:-/mnt/disk2/llama-3.2-1b.easylm}" \
    --input_length="${INPUT_LENGTH:-1024}" \
    --seq_length="${SEQ_LENGTH:-2048}" \
    --do_sample="${DO_SAMPLE:-True}" \
    --top_k="${TOP_K:-50}" \
    --top_p="${TOP_P:-0.95}"
