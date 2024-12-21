#!/bin/bash

echo "Starting worker cleanup..."

echo "Removing core dumps..."
sudo rm -f /tmp/tpu_logs/core*

echo "Cleaning up mlxu temporary files..."
rm -rf /tmp/mlxu

echo "Cleanup complete!"
