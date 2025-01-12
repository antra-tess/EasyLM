#!/bin/bash

# Get the list of TPU worker IPs
WORKER_IPS=$(gcloud compute tpus tpu-vm list --zone=$MOUNT_ZONE --filter="name~'$MOUNT_NAME'" --format="csv[no-heading](networkEndpoints[].ipAddress)")

# if disk is 'disk2' then ip is 10.96.49.202 else ip is 10.127.194.242
if [ "$DISK" == "disk2" ]; then
    DISK_PATH="10.96.49.202:/ftshare"
else
    DISK_PATH="10.127.194.242:/disk3"

# Function to mount NFS on a worker
mount_nfs() {
    local worker_ip=$1
    local worker_index=$2
    echo "Configuring NFS mount on worker ${worker_index}: ${worker_ip}"

    # Use SSH to execute commands on the worker
    gcloud compute tpus tpu-vm ssh "$MOUNT_NAME" --zone=$MOUNT_ZONE --worker=${worker_index} --command="
        # Create mount directory if it doesn't exist
        sudo mkdir -p /mnt/$DISK

        # Check if already mounted to avoid duplicate mounts
        if ! mount | grep -q '/mnt/$DISK'; then
            # Install NFS client if not already installed
            if ! command -v mount.nfs &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y nfs-common
            fi

            # Mount the NFS share
            sudo mount -t nfs $DISK_PATH /mnt/$DISK
        fi

        # Always ensure proper permissions
        # sudo chown -R $(whoami):$(whoami) /mnt/$DISK
        sudo chmod -R 777 /mnt/$DISK

        echo 'NFS mount configured on worker ${worker_ip}'
    "
}
#10.96.49.202

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
