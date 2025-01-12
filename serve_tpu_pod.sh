#!/bin/bash

# Exit on error and undefined variables
set -eu

# Set coordinator IP (default to localhost if not specified)
export COORDINATOR_IP="${COORDINATOR_IP:-"51.81.181.136"}"
echo "Using coordinator IP: ${COORDINATOR_IP}"

echo "Starting TPU pod serving deployment..."

# Push local changes to antra's repo
echo "Pushing local changes to repository..."
git pull
git push antra

## Start coordinator server in background
#echo "Starting coordinator server..."
#python -m EasyLM.coordinator &
#COORDINATOR_PID=$!
#
## Give coordinator time to start
#sleep 5
#
#export SCRIPT=worker_serve.sh
#

# Copy worker script and start serving on all workers
echo "Starting worker clients..."
gcloud compute tpus tpu-vm scp $SCRIPT $INFER_NAME:~/$SCRIPT --zone=$INFER_ZONE --worker=all
gcloud compute tpus tpu-vm ssh $INFER_NAME --zone=$INFER_ZONE --worker=all --command="cd ~/EasyLM && git fetch && git reset --hard HEAD && git checkout main && git pull && chmod +x ~/$SCRIPT && export WANDB_API_KEY='${WANDB_API_KEY}'  && export HF_TOKEN='${HF_TOKEN}' && export COORDINATOR_URL='http://${COORDINATOR_IP}:5010' && ~/$SCRIPT"

echo "TPU pod serving deployment complete!"
echo "Coordinator server running with PID $COORDINATOR_PID"
