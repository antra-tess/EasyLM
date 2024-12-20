#!/bin/bash

set -eu
git push antra

echo "Starting TPU pod training deployment..."

# Copy worker script to all workers
echo "Copying worker script to all workers..."
gcloud compute tpus tpu-vm scp worker_train.sh finetune-70b:~/worker_train.sh --zone=us-central2-b --worker=all

# Make script executable and run on all workers
echo "Starting training on all workers..."
gcloud compute tpus tpu-vm ssh finetune-70b --zone=us-central2-b --worker=all --command="chmod +x ~/worker_train.sh && export WANDB_API_KEY='${WANDB_API_KEY}' && ~/worker_train.sh"

echo "Training deployment complete!"
