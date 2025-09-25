git submodule update --init --recursive

# notice uv should be install standalone to avoid conflicts; do not ues pip install
curl -LsSf https://astral.sh/uv/install.sh | sh


GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

# uv add  opencv-python-headless
sudo apt-get update && sudo apt-get install -y ffmpeg

# Fix: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1

uv pip install numpy==1.24.4 mediapy
uv pip install accelerate>=0.21.0
uv pip install -e SimplerEnv/ManiSkill2_real2sim
uv pip install -e SimplerEnv



# for SimplerENV rendering
sudo apt-get install vulkan-tools libvulkan1 xvfb libglvnd-dev -y


echo "[2/4] Ensure directories exist"
sudo mkdir -p /usr/share/vulkan/icd.d
sudo mkdir -p /usr/share/glvnd/egl_vendor.d
sudo mkdir -p /etc/vulkan/implicit_layer.d

echo "[3/4] Create or update JSON files"


# /usr/share/vulkan/icd.d/nvidia_icd.json
sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json >/dev/null <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155"
    }
}
EOF
echo "Wrote /usr/share/vulkan/icd.d/nvidia_icd.json"

# /usr/share/glvnd/egl_vendor.d/10_nvidia.json
sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
echo "Wrote /usr/share/glvnd/egl_vendor.d/10_nvidia.json"

# /etc/vulkan/implicit_layer.d/nvidia_layers.json
sudo tee /etc/vulkan/implicit_layer.d/nvidia_layers.json >/dev/null <<'EOF'
{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }
}
EOF
echo "Wrote /etc/vulkan/implicit_layer.d/nvidia_layers.json"

uv pip install sapien
uv run python -m sapien.example.offscreen

# copy pi05_base_pytorch to /dev/shm/
cp -r /mnt/hdfs/wenbo/vla/pi05/pi05_base_pytorch /dev/shm/

# copy dataset to /opt/tiger/openpi/
cp -r /mnt/hdfs/wenbo/vla/lerobot-pi0-bridge.tar /opt/tiger/openpi/

# untar dataset
tar -xvf /mnt/hdfs/wenbo/vla/lerobot-pi0-bridge.tar -C /dev/shm/