#!/bin/bash

# Exit on error and undefined variables
set -eu

echo "Starting TPU pod serving deployment..."

# Push local changes to antra's repo
echo "Pushing local changes to repository..."
git push antra

# Start coordinator server in background
echo "Starting coordinator server..."
python -m EasyLM.coordinator &
COORDINATOR_PID=$!

# Give coordinator time to start
sleep 5

export SCRIPT=worker_serve.sh

# Copy worker script and start serving on all workers
echo "Starting worker clients..."
gcloud compute tpus tpu-vm scp $SCRIPT finetune-70b:~/$SCRIPT --zone=us-central2-b --worker=all
gcloud compute tpus tpu-vm ssh finetune-70b --zone=us-central2-b --worker=all --command="cd ~/EasyLM && git fetch && git reset --hard HEAD && git checkout main && git pull && chmod +x ~/$SCRIPT && export WANDB_API_KEY='${WANDB_API_KEY}'  && export HF_TOKEN='${HF_TOKEN}' && ~/$SCRIPT"

echo "TPU pod serving deployment complete!"
echo "Coordinator server running with PID $COORDINATOR_PID"
