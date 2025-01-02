#!/bin/bash

# This script runs on TPU workers to clean up logs

set -eu

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Running cleanup on worker $(hostname)"

# Clear TPU logs
log "Clearing TPU logs..."
sudo rm -rf /tmp/tpu_logs/*
sudo mkdir -p /tmp/tpu_logs
sudo chmod 777 /tmp/tpu_logs

log "Worker cleanup complete on $(hostname)"
