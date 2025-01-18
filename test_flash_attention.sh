#!/bin/bash

set -eu

# Get worker number from hostname
WORKER_NUM=$(hostname | grep -o '[0-9]*$')

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

cd ~/EasyLM

#if [ "$WORKER_NUM" = "0" ]; then
#    echo "Starting flash attention test on worker 0..."
    # Run flash attention tests with output
    python -m EasyLM.attention.test_flash_attention
#else
    # Run tests but suppress output
#    python -m EasyLM.attention.test_flash_attention > /dev/null 2>&1
#fi
