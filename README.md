# Blender-ML-Thesis
My master's thesis repo
# Setup
> The setup described below is for Linux Ubuntu 24.01.

Ubuntu preRecs
---

1. **Install docker**: Docker install instructions [here](https://docs.docker.com/desktop/setup/install/linux/ubuntu/) 
```bash
% install docker on whatever OS you are on
$ sudo apt-get update
$ sudo apt-get install docker-compose-plugin
```
2. **Nvidia Toolkit**
```bash
    $ sudo apt-get update && sudo apt-get install -y nvidia-cuda-toolkit`
```
- **install container toolkit**: Install instructions from Nvidia [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#setting-up-nvidia-container-toolkit)
```bash
$ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
$ sudo apt-get update
$ sudo apt-get install -y nvidia-container-toolkit
```
3. 