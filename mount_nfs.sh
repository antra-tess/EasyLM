#!/bin/bash

# Get the list of TPU worker IPs
WORKER_IPS=$(gcloud compute tpus tpu-vm list --zone=us-central2-b --filter="name~'finetune-70b'" --format="csv[no-heading](networkEndpoints[].ipAddress)")

# Function to mount NFS on a worker
mount_nfs() {
    local worker_ip=$1
    local worker_index=$2
    echo "Configuring NFS mount on worker ${worker_index}: ${worker_ip}"

    # Use SSH to execute commands on the worker
    gcloud compute tpus tpu-vm ssh "finetune-70b" --zone=us-central2-b --worker=${worker_index} --command="
        # Create mount directory if it doesn't exist
        sudo mkdir -p /mnt/disk2

        # Check if already mounted to avoid duplicate mounts
        if ! mount | grep -q '/mnt/disk2'; then
            # Install NFS client if not already installed
            if ! command -v mount.nfs &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y nfs-common
            fi

            # Mount the NFS share
            sudo mount -t nfs 10.96.49.202:/ftshare /mnt/disk2
        fi

        # Always ensure proper permissions
        sudo chown -R $(whoami):$(whoami) /mnt/disk2
        sudo chmod -R 777 /mnt/disk2

        echo 'NFS mount configured on worker ${worker_ip}'
    "
}

# Mount NFS on each worker
echo "Starting NFS mount process across workers..."
worker_index=0
for worker_ip in $(echo $WORKER_IPS | tr ';' '\n'); do
    mount_nfs "$worker_ip" "$worker_index" &
    ((worker_index++))
done

# Wait for all mounts to complete
wait

echo "NFS mount process completed on all workers"
