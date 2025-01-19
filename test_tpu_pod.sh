#!/bin/bash

# Exit on error and undefined variables
set -eu

echo "Starting TPU pod test deployment..."

# Push local changes to antra's repo
echo "Pushing local changes to repository..."
git pull
git push

# Copy script and run on all workers
echo "Starting test on all workers..."
gcloud compute tpus tpu-vm scp $INFER_SCRIPT $INFER_NAME:~/$INFER_SCRIPT --zone=$INFER_ZONE --worker=all
gcloud compute tpus tpu-vm ssh $INFER_NAME --zone=$INFER_ZONE --worker=all --command="cd ~/EasyLM && git fetch && git reset --hard HEAD && git checkout main && git pull && chmod +x ~/$INFER_SCRIPT && export INFER_DISK=${INFER_DISK} && export HF_TOKEN='${HF_TOKEN}' && ~/$INFER_SCRIPT"

echo "TPU pod test deployment complete!"
