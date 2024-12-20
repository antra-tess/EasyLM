#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Cleaning processes on worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- '
    echo "Looking for EasyLM processes..."
    
    # Find Python processes containing EasyLM but exclude grep and the kill script itself
    WORKER_PROCS=$(ps aux | grep "[E]asyLM" | grep "python" | grep -v "kill_all" | awk "{print \$2}")
    
    if [ -z "$WORKER_PROCS" ]; then
      echo "No EasyLM processes found"
    else
      echo "Found processes:"
      ps aux | grep "[E]asyLM" | grep "python" | grep -v "kill_all"
      
      # Kill each process
      for pid in $WORKER_PROCS; do
        echo "Killing process $pid"
        sudo kill -9 $pid 2>/dev/null || true
      done
      echo "Processes killed"
    fi

    # Remove the TPU lockfile if it exists
    if [ -f /tmp/libtpu_lockfile ]; then
      echo "Removing lockfile"
      sudo rm -f /tmp/libtpu_lockfile
    fi
    '
done

echo "Completed processing all workers"
