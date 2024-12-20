#!/bin/bash

# Exit on error and undefined variables
set -eu

echo "Starting TPU pod deployment..."

# Push local changes to antra's repo
echo "Pushing local changes to repository..."
git push antra

# Deploy to all workers
echo "Deploying to TPU workers..."
gcloud compute tpus tpu-vm ssh finetune-70b --zone=us-central2-b --worker=all --command="cd /tmp && rm -rf EasyLM && git clone https://github.com/antra-tess/EasyLM.git && cd EasyLM && ./setup_tpu_pod.sh"

echo "TPU pod deployment complete!"
