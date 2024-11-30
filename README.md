# Blender-ML-Thesis
My master's thesis repo

## Setup
> The setup described below is for Linux Ubuntu 24.01.

### Ubuntu Prerequisites

1. **Install Docker**: Follow the Docker install instructions [here](https://docs.docker.com/desktop/setup/install/linux/ubuntu/)
    ```bash
    # Install Docker on Ubuntu
    sudo apt-get update
    sudo apt-get install docker-compose-plugin
    ```

2. **Nvidia Toolkit**
    ```bash
    sudo apt-get update && sudo apt-get install -y nvidia-cuda-toolkit
    ```
    - **Install container toolkit**: Follow the install instructions from Nvidia [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#setting-up-nvidia-container-toolkit)
    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    ```

3. **Allow X11 Forwarding** (this step might be deprecated soon):
    ```bash
    # Allow non-network local connections to the X server
    xhost +local:docker
    ```
    **Shell output**:
    ```
    non-network local connections being added to access control list
    ```

### Test The Repo

4. **Run Docker Compose**
    ```bash
    docker compose up
    ```

## Python Dependencies

1. Download PyTorch for GPU: Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install the appropriate version for your system.