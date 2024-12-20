#!/bin/bash

# Exit on error
set -e

echo "Setting up TPU pod for EasyLM..."

# Clone repository
if [ ! -d "EasyLM" ]; then
    git clone git@github.com:antra-tess/EasyLM.git
fi

cd EasyLM

# Run TPU setup script
./scripts/tpu_vm_setup.sh

# Set up environment variables
echo "export PYTHONPATH=${PWD}:${PYTHONPATH}" >> ~/.bashrc
echo "export WANDB_PROJECT=levanter-sft" >> ~/.bashrc

# Mount GCS bucket if not already mounted
if [ ! -d "/mnt/gcs-bucket" ]; then
    sudo mkdir -p /mnt/gcs-bucket
    sudo mount -o nonempty {bucket-name} /mnt/gcs-bucket
fi

echo "TPU pod setup complete!"
