#!/bin/bash

# Check if instance name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <instance-name>"
    echo "Example: $0 node-001"
    exit 1
fi

INSTANCE_NAME=$1
JUMP_HOST="finetune-70b"

# Get the list of worker IPs for the target instance
echo "Getting worker IPs for ${INSTANCE_NAME}..."
WORKER_IPS=$(gcloud compute tpus tpu-vm list --zone=us-central2-b --filter="name~'${INSTANCE_NAME}'" --format="csv[no-heading](networkEndpoints[].ipAddress)")

# Convert semicolon-separated list to space-separated
WORKER_IPS=$(echo "$WORKER_IPS" | tr ';' ' ')

# First SSH to the jump host, then to target instance using IPs
echo "Using ${JUMP_HOST} as jump host to reach ${INSTANCE_NAME}..."
gcloud compute tpus tpu-vm ssh "${JUMP_HOST}" --zone=us-central2-b -A --command="
    echo 'Connecting to ${INSTANCE_NAME} workers...'
    read -r -a IPS <<< \"${WORKER_IPS}\"
    for i in {0..15}; do
        IP=\${IPS[\$i]}
        echo \"Configuring worker \$i (\$IP)...\"
        ssh -A -o StrictHostKeyChecking=no \$IP \"
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
