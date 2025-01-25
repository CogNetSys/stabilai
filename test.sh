#!/bin/bash

# Set the required versions (customize as needed)
REQUIRED_DRIVER_VERSION=535
REQUIRED_CUDA_VERSION="12.4"
REQUIRED_TORCH_CUDA="cu124"

# Helper function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# 1. Check NVIDIA driver version
check_driver() {
    echo "Checking NVIDIA driver..."
    if command_exists nvidia-smi; then
        INSTALLED_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d "." -f 1)
        if [ "$INSTALLED_DRIVER_VERSION" -ge "$REQUIRED_DRIVER_VERSION" ]; then
            echo "NVIDIA driver is up-to-date (version $INSTALLED_DRIVER_VERSION)."
        else
            echo "Updating NVIDIA driver to version $REQUIRED_DRIVER_VERSION..."
            sudo apt-get update && sudo apt-get install -y nvidia-driver-$REQUIRED_DRIVER_VERSION
        fi
    else
        echo "NVIDIA driver not found. Installing..."
        sudo apt-get update && sudo apt-get install -y nvidia-driver-$REQUIRED_DRIVER_VERSION
    fi
}

# 2. Check CUDA version
check_cuda() {
    echo "Checking CUDA..."
    if command_exists nvcc; then
        INSTALLED_CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9.]+" | head -n 1)
        if [ "$INSTALLED_CUDA_VERSION" == "$REQUIRED_CUDA_VERSION" ]; then
            echo "CUDA is up-to-date (version $INSTALLED_CUDA_VERSION)."
        else
            echo "CUDA version mismatch (found $INSTALLED_CUDA_VERSION, need $REQUIRED_CUDA_VERSION). Updating..."
            install_cuda
        fi
    else
        echo "CUDA not found. Installing..."
        install_cuda
    fi
}

install_cuda() {
    CUDA_INSTALLER=cuda_${REQUIRED_CUDA_VERSION}_linux.run
    wget https://developer.download.nvidia.com/compute/cuda/${REQUIRED_CUDA_VERSION}/local_installers/${CUDA_INSTALLER}
    sudo sh ${CUDA_INSTALLER} --silent --toolkit --override
    rm -f ${CUDA_INSTALLER}
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
}

# 3. Check PyTorch installation
check_pytorch() {
    echo "Checking PyTorch..."
    python3 -c "import torch; print(torch.__version__); print(torch.version.cuda)" 2>/dev/null | grep "$REQUIRED_TORCH_CUDA" &> /dev/null
    if [ $? -eq 0 ]; then
        echo "PyTorch with CUDA $REQUIRED_TORCH_CUDA is already installed."
    else
        echo "Installing PyTorch with CUDA $REQUIRED_TORCH_CUDA..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${REQUIRED_TORCH_CUDA}
    fi
}

# Run checks
check_driver
check_cuda
check_pytorch

# Final message
echo "Environment setup complete!"
