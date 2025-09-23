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
uv pip install -e SimplerEnv/ManiSkill2_real2sim
uv pip install -e SimplerEnv