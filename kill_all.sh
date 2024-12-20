TPU_NODE="finetune-70b"
ZONE="us-central2-b"
NUM_WORKERS=8

for i in $(seq 0 $((NUM_WORKERS-1))); do
  echo "=== Cleaning processes on worker $i ==="
  gcloud compute tpus tpu-vm ssh "$TPU_NODE" \
    --zone="$ZONE" \
    --worker="$i" -- '
    echo "Looking for Python processes under worker account on worker '$i'..."
    
    # Find all Python processes under service accounts and their children
    WORKER_PROCS=$(ps aux | grep "^sa_" | grep python | awk "{print \$2}")
    
    if [ -z "$WORKER_PROCS" ]; then
      echo "No Python processes found under worker account on worker '$i'"
    else
      echo "Found processes on worker '$i':"
      ps aux | grep "^sa_" | grep python
      
      # Kill each process and its children
      for pid in $WORKER_PROCS; do
        echo "Killing process tree for PID: $pid on worker '$i'"
        # Kill children first
        sudo pkill -9 -P $pid
        # Then kill the parent
        sudo kill -9 $pid
      done
      echo "Processes killed on worker '$i'"
    fi

    # Remove the TPU lockfile if it exists
    if [ -f /tmp/libtpu_lockfile ]; then
      echo "Removing lockfile on worker '$i'"
      sudo rm -f /tmp/libtpu_lockfile
    fi
    ' 2>&1 | tee worker_${i}_kill.log &
done

# Wait for all background processes to complete
wait

echo "Completed processing all workers"
reset
