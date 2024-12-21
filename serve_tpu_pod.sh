#!/bin/bash

# Exit on error and undefined variables
set -eu

echo "Starting TPU pod serving deployment..."

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

echo "Deploying to TPU workers..."
gcloud compute tpus tpu-vm ssh finetune-70b --zone=us-central2-b --worker=all --command="export HF_TOKEN='${HF_TOKEN}' && cd ~ && (if [ -d 'EasyLM' ]; then cd EasyLM && git pull; else git clone https://github.com/antra-tess/EasyLM.git && cd EasyLM; fi) && git update-index --chmod=+x worker_serve.sh && chmod +x worker_serve.sh && ./worker_serve.sh"

echo "TPU pod serving deployment complete!"
