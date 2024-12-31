#!/bin/bash

set -eu

echo "Starting TPU pod training deployment..."

# Push latest changes
echo "Pushing changes to repository..."
git pull
git push antra

export SCRIPT=worker_train.sh

# Copy worker script and start training on all workers
echo "Starting training on all workers..."
gcloud compute tpus tpu-vm scp $SCRIPT finetune-70b:~/$SCRIPT --zone=us-central2-b --worker=all
gcloud compute tpus tpu-vm ssh finetune-70b --zone=us-central2-b --worker=all --command="cd ~/EasyLM && git fetch && git reset --hard HEAD && git checkout optim_fwd4 && git pull && chmod +x ~/$SCRIPT && export WANDB_API_KEY='${WANDB_API_KEY}'  && export HF_TOKEN='${HF_TOKEN}' && ~/$SCRIPT"

echo "Training deployment complete!"
