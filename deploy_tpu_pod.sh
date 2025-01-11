#!/bin/bash

# Exit on error and undefined variables
set -eu

echo "Starting TPU pod deployment..."

# Push local changes to antra's repo
echo "Pushing local changes to repository..."
git push antra

# Deploy to all workers
# Get HF token from environment or ask for it
if [ -z "${HF_TOKEN}" ]; then
    echo "HF_TOKEN not found in environment"
    read -p "Please enter your HuggingFace token: " token
    export HF_TOKEN="${token}"
fi

# Add setup flag with default to false
setup=${1:-false}

echo "Deploying to TPU workers..."
if [ "$setup" = "true" ]; then
    echo "Running with full environment setup..."
    gcloud compute tpus tpu-vm ssh finetune70b --zone=us-central2-b --worker=all --command="export HF_TOKEN='${HF_TOKEN}' && export WANDB_API_KEY='${WANDB_API_KEY}'  && cd ~ && rm -rf EasyLM && git clone https://github.com/antra-tess/EasyLM.git && cd EasyLM && chmod +x setup_tpu_pod.sh && ./setup_tpu_pod.sh"
else
    echo "Running without environment setup..."
    gcloud compute tpus tpu-vm ssh finetune70b --zone=us-central2-b --worker=all --command="export HF_TOKEN='${HF_TOKEN}' && export WANDB_API_KEY='${WANDB_API_KEY}' && cd ~ && (if [ -d 'EasyLM' ]; then cd EasyLM && git pull; else git clone https://github.com/antra-tess/EasyLM.git && cd EasyLM; fi) && export PYTHONPATH=~/EasyLM:\${PYTHONPATH}"
fi

echo "TPU pod deployment complete!"
