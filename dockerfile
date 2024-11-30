# Use the official NVIDIA CUDA image as a base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies (to make blender happy)
RUN apt-get update && apt-get install -y \
    wget \
    software-properties-common \
    libx11-6 \
    libx11-dev \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxi6 \
    libxrender1 \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    libxxf86vm1 \
    libxkbcommon0 \
    libsm6 \
    libice6 \
    snapd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Blender 4
RUN wget https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz \
    && tar -xvf blender-4.0.0-linux-x64.tar.xz \
    && mv blender-4.0.0-linux-x64 /opt/blender \
    && ln -s /opt/blender/blender /usr/local/bin/blender


# Set the entrypoint to Blender
ENTRYPOINT ["blender"]