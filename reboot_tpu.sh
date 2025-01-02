#!/bin/bash

# This script runs on the host to coordinate worker cleanup and TPU reboot

set -eu

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Get TPU name from environment or argument
TPU_NAME=${1:-${TPU_NAME}}
if [ -z "$TPU_NAME" ]; then
    log "Error: TPU_NAME not set and no argument provided"
    log "Usage: $0 <tpu-name>"
    exit 1
fi

log "Running on host, coordinating worker cleanup..."

# Copy worker cleanup script to all workers
log "Copying cleanup script to workers..."
for i in {0..7}; do
    gcloud compute tpus tpu-vm scp worker_cleanup.sh ${TPU_NAME}:~/worker_cleanup.sh --zone=us-central2-b --worker=$i &
done
wait

log "Running cleanup on all workers..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=us-central2-b --worker=all --command="chmod +x ~/worker_cleanup.sh && ~/worker_cleanup.sh"

# Now do the actual TPU reboot
log "Rebooting TPU: $TPU_NAME..."
if ! gcloud compute tpus tpu-vm stop $TPU_NAME --zone=us-central2-b; then
    log "Error stopping TPU"
    exit 1
fi

log "Waiting for TPU to stop completely..."
sleep 20

if ! gcloud compute tpus tpu-vm start $TPU_NAME --zone=us-central2-b; then
    log "Error starting TPU"
    exit 1
fi

log "Waiting for TPU to initialize..."
sleep 40

log "TPU reboot complete. Checking status..."
gcloud compute tpus tpu-vm describe $TPU_NAME --zone=us-central2-b

log "Reboot process completed successfully"
