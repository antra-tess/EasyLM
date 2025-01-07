#!/bin/bash

# Check if instance name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <instance-name>"
    echo "Example: $0 node-001"
    exit 1
fi

INSTANCE_NAME=$1
JUMP_HOST="finetune-70b"

# First SSH to the jump host, then to target instance
echo "Using ${JUMP_HOST} as jump host to reach ${INSTANCE_NAME}..."
gcloud compute tpus tpu-vm ssh "${JUMP_HOST}" --zone=us-central2-b --command="
    echo 'Connecting to ${INSTANCE_NAME} workers...'
    for i in {0..15}; do
        echo \"Configuring worker \$i...\"
        ssh worker\$i.${INSTANCE_NAME} \"
            sudo mkdir -p /mnt/disk2
            if ! mount | grep -q '/mnt/disk2'; then
                sudo apt-get update
                sudo apt-get install -y nfs-common
                sudo mount -t nfs 10.96.49.202:/ftshare /mnt/disk2
            fi
            sudo chown -R \$(whoami):\$(whoami) /mnt/disk2
            sudo chmod -R 777 /mnt/disk2
        \"
    done
"
