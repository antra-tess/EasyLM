#!/bin/bash

set -eu

echo "Starting TPU pod training deployment..."

# Push latest changes
echo "Pushing changes to repository..."
git push antra

# Copy worker script and start training on all workers
echo "Starting training on all workers..."
gcloud compute tpus tpu-vm scp worker_train.sh finetune-70b:~/worker_train.sh --zone=us-central2-b --worker=all
gcloud compute tpus tpu-vm ssh finetune-70b --zone=us-central2-b --worker=all --command="cd ~/EasyLM && git pull && chmod +x ~/worker_train.sh && export WANDB_API_KEY='${WANDB_API_KEY}' && ~/worker_train.sh"

echo "Training deployment complete!"
