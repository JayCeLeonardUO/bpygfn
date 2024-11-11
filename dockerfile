# Use the official NVIDIA CUDA image as a base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    software-properties-common \
    && add-apt-repository ppa:graphics-drivers/ppa \
    && apt-get update && apt-get install -y \
    blender \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the entrypoint to Blender
ENTRYPOINT ["blender"]