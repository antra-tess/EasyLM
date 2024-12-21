#!/bin/bash

TPU_NODE="finetune-70b"
ZONE="us-central2-b"

echo "=== Cleaning processes on all workers ==="

# Execute cleanup on all workers in parallel
gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
  --zone="$ZONE" \
  --worker=all \
  --command='
    echo "Looking for EasyLM processes on $(hostname)..."
    
    # Find Python processes containing EasyLM but exclude grep and the kill script itself
    WORKER_PROCS=$(ps aux | grep "[E]asyLM" | grep "python" | grep -v "kill_all" | awk "{print \$2}")
    
    if [ -z "$WORKER_PROCS" ]; then
      echo "No EasyLM processes found on $(hostname)"
    else
      echo "Found processes on $(hostname):"
      ps aux | grep "[E]asyLM" | grep "python" | grep -v "kill_all"
      
      # Kill each process
      for pid in $WORKER_PROCS; do
        echo "Killing process $pid on $(hostname)"
        sudo kill -9 $pid 2>/dev/null || true
      done
      echo "Processes killed on $(hostname)"
    fi

    # Remove the TPU lockfile if it exists
    if [ -f /tmp/libtpu_lockfile ]; then
      echo "Removing lockfile on $(hostname)"
      sudo rm -f /tmp/libtpu_lockfile
    fi
  '

echo "Completed processing all workers"
