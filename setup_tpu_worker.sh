#!/bin/bash

# Exit on error and undefined variables
set -eu

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python package installation
check_package() {
    python3 -c "import $1" >/dev/null 2>&1
}

echo "Starting TPU pod setup for EasyLM..."


# Check for HF token
if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN not found in environment"
    read -p "Please enter your HuggingFace token: " token
    echo "export HF_TOKEN='${token}'" >> ~/.bashrc
    export HF_TOKEN="${token}"
fi


# Remove existing EasyLM directory if exists
if [ -d "EasyLM" ]; then
    echo "Removing existing EasyLM directory..."
    rm -rf EasyLM
fi

echo "Cloning EasyLM repository..."
git clone https://github.com/antra-tess/EasyLM.git
cd EasyLM

# Run TPU setup script if packages not installed
if ! check_package "python-socketio"; then
    echo "Installing required packages..."
    ./scripts/tpu_vm_setup.sh
else
    echo "JAX TPU support already installed"
fi

# Set up environment variables
echo "Setting up environment variables..."
mkdir -p ~/.env
cat > ~/.env/easylm.sh <<EOF
export WANDB_PROJECT="levanter-sft"
export WANDB_API_KEY="${WANDB_API_KEY}"
export HF_TOKEN="${HF_TOKEN}"
EOF

# Add source to bashrc if not already present
if ! grep -q "source ~/.env/easylm.sh" ~/.bashrc; then
    echo "source ~/.env/easylm.sh" >> ~/.bashrc
fi

echo "TPU pod setup complete!"
echo "Please run 'source ~/.bashrc' to load the environment variables"
