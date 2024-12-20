#!/bin/bash

# Exit on error and undefined variables
set -eu

echo "Starting TPU pod deployment..."

# Push local changes to antra's repo
echo "Pushing local changes to repository..."
git push antra

# Deploy to all workers
# Get HF token from environment or ask for it
if [ -z "${HUGGINGFACE_TOKEN}" ]; then
    echo "HUGGINGFACE_TOKEN not found in environment"
    read -p "Please enter your HuggingFace token: " token
    export HUGGINGFACE_TOKEN="${token}"
fi

echo "Deploying to TPU workers..."
gcloud compute tpus tpu-vm ssh finetune-70b --zone=us-central2-b --worker=all --command="export HUGGINGFACE_TOKEN='${HUGGINGFACE_TOKEN}' && cd /tmp && rm -rf EasyLM && git clone https://github.com/antra-tess/EasyLM.git && cd EasyLM && chmod +x setup_tpu_pod.sh && ./setup_tpu_pod.sh"

echo "TPU pod deployment complete!"
