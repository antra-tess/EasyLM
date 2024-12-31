#!/bin/bash

export TPU_NAME=finetune-70b

set -eu  # Exit on error and undefined variables

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check for gcloud
if ! command_exists gcloud; then
    log "Error: gcloud command not found"
    exit 1
fi

# Get TPU name from environment or as argument
TPU_NAME=${1:-${TPU_NAME}}
if [ -z "$TPU_NAME" ]; then
    log "Error: TPU_NAME not set and no argument provided"
    log "Usage: $0 <tpu-name>"
    exit 1
fi

# Stop all python processes
log "Stopping all python processes..."
pkill -9 python || true  # Don't fail if no processes found

# Clear TPU logs
log "Clearing TPU logs..."
sudo rm -rf /tmp/tpu_logs/*
sudo mkdir -p /tmp/tpu_logs
sudo chmod 777 /tmp/tpu_logs

# Reboot TPU
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
