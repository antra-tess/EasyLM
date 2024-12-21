#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

echo "Starting TPU pod cleanup..."

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Cleaning worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- '
    echo "Removing core dumps..."
    sudo rm -f /tmp/tpu_logs/core*
    
    echo "Cleaning up mlxu temporary files..."
    rm -rf /tmp/mlxu
    
    echo "Cleanup complete for worker '$i'"
    '
done

echo "TPU pod cleanup complete!"
