#!/bin/bash

# Exit on error and undefined variables
set -eu

echo "Starting TPU pod flash attention test deployment..."

# Push local changes to antra's repo
echo "Pushing local changes to repository..."
git pull
git push antra

# Copy script and run on all workers
echo "Starting test on all workers..."
gcloud compute tpus tpu-vm scp test_flash_attention.sh $INFER_NAME:~/test_flash_attention.sh --zone=$INFER_ZONE --worker=all
gcloud compute tpus tpu-vm ssh $INFER_NAME --zone=$INFER_ZONE --worker=all --command="cd ~/EasyLM && git fetch && git reset --hard HEAD && git checkout main && git pull && chmod +x ~/test_flash_attention.sh && ~/test_flash_attention.sh"

echo "TPU pod flash attention test deployment complete!"
