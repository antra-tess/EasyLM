#!/bin/bash

# Check if instance name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <instance-name>"
    echo "Example: $0 node-001"
    exit 1
fi

INSTANCE_NAME=$1

# First SSH to finetune-70b, then use it to reach target workers
echo "Using finetune-70b as jump host to reach ${INSTANCE_NAME} workers..."
gcloud compute tpus tpu-vm ssh "finetune-70b" --zone=us-central2-b --account=antra@tesserae.cc --command="
    for i in {0..15}; do
        echo \"Configuring worker \$i...\"
        gcloud compute tpus tpu-vm ssh \"${INSTANCE_NAME}\" --zone=us-central2-b --account=antra@tesserae.cc --worker=\$i --command=\"
            sudo mkdir -p /mnt/disk2
            if ! mount | grep -q '/mnt/disk2'; then
                sudo apt-get update
                sudo apt-get install -y nfs-common
                sudo mount -t nfs 10.96.49.202:/ftshare /mnt/disk2
            fi
            sudo chown -R \\\$(whoami):\\\$(whoami) /mnt/disk2
            sudo chmod -R 777 /mnt/disk2
        \"
    done
"
