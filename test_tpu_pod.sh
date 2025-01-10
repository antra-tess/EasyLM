#!/bin/bash

# Exit on error and undefined variables
set -eu

echo "Starting TPU pod test deployment..."

# Push local changes to antra's repo
echo "Pushing local changes to repository..."
git pull
git push antra

export SCRIPT=test_inference_sim.sh


# Copy script and run on all workers
echo "Starting test on all workers..."
gcloud compute tpus tpu-vm scp $SCRIPT finetune-70b:~/$SCRIPT --zone=us-central2-b --worker=all
gcloud compute tpus tpu-vm ssh finetune-70b --zone=us-central2-b --worker=all --command="cd ~/EasyLM && git fetch && git reset --hard HEAD && git checkout main && git pull && chmod +x ~/$SCRIPT && export HF_TOKEN='${HF_TOKEN}' && ~/$SCRIPT"

echo "TPU pod test deployment complete!"