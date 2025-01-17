#!/bin/bash

set -eu

echo "Starting flash attention test on worker..."

# Fix TPU logs permissions first
sudo mkdir -p /tmp/tpu_logs && sudo chown -R $USER:$USER /tmp/tpu_logs && sudo chmod 755 /tmp/tpu_logs

cd ~/EasyLM

# Run flash attention tests
python -m EasyLM.attention.test_flash_attention
