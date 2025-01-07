#!/bin/bash

# Get worker0's hostname from finetune-70b
echo "Getting proxy hostname from finetune-70b worker0..."
PROXY_HOSTNAME=$(gcloud compute tpus tpu-vm ssh "finetune-70b" --zone=us-central2-b --worker=0 --account=antra@tesserae.cc --command="hostname -f")

echo "Setting up SOCKS proxy on ${PROXY_HOSTNAME}..."
gcloud compute tpus tpu-vm ssh "finetune-70b" --zone=us-central2-b --worker=0 --account=antra@tesserae.cc --command="
    # Kill any existing proxy
    pkill -f 'ssh.*1080'
    # Set up new SOCKS proxy
    ssh -D 1080 -f -C -q -N localhost
"

echo "Proxy setup complete. Use these settings:"
echo "export http_proxy=socks5://${PROXY_HOSTNAME}:1080"
echo "export https_proxy=socks5://${PROXY_HOSTNAME}:1080"
