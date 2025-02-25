#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <target-instance>"
    echo "Example: $0 node-001"
    exit 1
fi

TARGET_INSTANCE=$1

# Get worker0's hostname from finetune70b
echo "Getting proxy hostname from finetune70b worker0..."
PROXY_HOSTNAME=$(gcloud compute tpus tpu-vm ssh "finetune70b" --zone=us-central2-b --worker=0 --account=antra@tesserae.cc --command="hostname -f")

echo "Setting up SOCKS proxy on ${PROXY_HOSTNAME}..."
gcloud compute tpus tpu-vm ssh "finetune70b" --zone=us-central2-b --account=antra@tesserae.cc --command='pkill -f "ssh.*1080" || true; ssh -D 0.0.0.0:1080 -N -f -v localhost'


# Configure proxy on each worker of target instance
echo "Configuring proxy on ${TARGET_INSTANCE} workers..."
for i in {0..15}; do
    echo "Setting up proxy on worker $i..."
    gcloud compute tpus tpu-vm ssh "${TARGET_INSTANCE}" --zone=us-central2-b --worker=$i --account=antra@tesserae.cc --command="
        echo 'export http_proxy=socks5://${PROXY_HOSTNAME}:1080' >> ~/.bashrc
        echo 'export https_proxy=socks5://${PROXY_HOSTNAME}:1080' >> ~/.bashrc
        export http_proxy=socks5://${PROXY_HOSTNAME}:1080
        export https_proxy=socks5://${PROXY_HOSTNAME}:1080
    "
done

echo "Proxy setup complete on all workers."
