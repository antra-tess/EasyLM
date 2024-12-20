#!/bin/bash

set -eu

echo "Starting TPU pod training..."

# Deploy to all workers
echo "Deploying training command to all workers..."
gcloud compute tpus tpu-vm ssh finetune-70b --zone=us-central2-b --worker=all --command="cd ~/EasyLM && python -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,1' \
    --llama.base_model='llama3_8b' \
    --load_checkpoint='params::gs://finetune70b/llama-3-8b/llama-3-8b' \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='[instruction+input],output' \
    --train_dataset.huggingface_dataset.name="" \
    --train_dataset.huggingface_dataset.path='tatsu-lab/alpaca'"

echo "Training command deployed to all workers!"
