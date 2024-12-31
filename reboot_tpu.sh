#!/bin/bash

# Stop all python processes
echo "Stopping all python processes..."
pkill -9 python

# Clear TPU logs
echo "Clearing TPU logs..."
sudo rm -rf /tmp/tpu_logs/*
sudo mkdir -p /tmp/tpu_logs
sudo chmod 777 /tmp/tpu_logs

# Get TPU name from environment or as argument
TPU_NAME=${1:-${TPU_NAME}}
if [ -z "$TPU_NAME" ]; then
    echo "Error: TPU_NAME not set and no argument provided"
    exit 1
fi

echo "Rebooting TPU: $TPU_NAME..."
gcloud compute tpus tpu-vm stop $TPU_NAME --zone=us-central2-b
sleep 10
gcloud compute tpus tpu-vm start $TPU_NAME --zone=us-central2-b
sleep 30

echo "TPU reboot complete"
