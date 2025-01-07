#!/bin/bash

# Check if instance name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <instance-name>"
    echo "Example: $0 node-001"
    exit 1
fi

INSTANCE_NAME=$1

# SSH to the main TPU VM and run mount commands directly
echo "Connecting to ${INSTANCE_NAME} to mount NFS..."
gcloud compute tpus tpu-vm ssh "${INSTANCE_NAME}" --zone=us-central2-b --command="
    sudo mkdir -p /mnt/disk2
    if ! mount | grep -q '/mnt/disk2'; then
        sudo apt-get update
        sudo apt-get install -y nfs-common
        sudo mount -t nfs 10.96.49.202:/ftshare /mnt/disk2
    fi
    sudo chown -R \$(whoami):\$(whoami) /mnt/disk2
    sudo chmod -R 777 /mnt/disk2
"
