#!/bin/bash

# This script should be run on the host machine, not on workers
# It will copy itself to workers and execute the worker portion there

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

if [[ -f /.dockerenv ]] || grep -q '/docker/' /proc/self/cgroup; then
    # We are on a worker - do worker cleanup
    log "Running on worker $(hostname)"
    
    # Clear TPU logs
    log "Clearing TPU logs..."
    sudo rm -rf /tmp/tpu_logs/*
    sudo mkdir -p /tmp/tpu_logs
    sudo chmod 777 /tmp/tpu_logs
    
    log "Worker cleanup complete on $(hostname)"
    exit 0
fi

# We are on the host - orchestrate the cleanup
log "Running on host, coordinating worker cleanup..."

# Copy this script to all workers and run it
log "Copying cleanup script to workers..."
for i in {0..31}; do
    gcloud compute tpus tpu-vm scp "$0" ${TPU_NAME}:~/reboot_tpu.sh --zone=us-central2-b --worker=$i &
done
wait

log "Running cleanup on all workers..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=us-central2-b --worker=all --command="chmod +x ~/reboot_tpu.sh && ~/reboot_tpu.sh"

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
