#!/bin/bash

TPU_NODE="finetune70b"
ZONE="us-central2-b"
NUM_WORKERS=8

echo "Starting TPU pod cleanup..."

# Copy cleanup script to all workers
echo "Copying cleanup script to workers..."
gcloud compute tpus tpu-vm scp cleanup_worker.sh ${TPU_NODE}:~/cleanup_worker.sh --zone=${ZONE} --worker=all

# Run cleanup on all workers in parallel
echo "Running cleanup on all workers..."
gcloud compute tpus tpu-vm ssh ${TPU_NODE} --zone=${ZONE} --worker=all --command="chmod +x ~/cleanup_worker.sh && ~/cleanup_worker.sh"

echo "TPU pod cleanup complete!"
