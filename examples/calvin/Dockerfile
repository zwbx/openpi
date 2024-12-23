# THIS DOCKERFILE DOES NOT YET WORK
# Dockerfile for the CALVIN benchmark.

# Build the container:
# docker build . -t calvin -f examples/calvin/Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --privileged --gpus all calvin /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    make \
    g++ \
    git \
    wget \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    unzip \
    ffmpeg

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR
ENV PATH=$CONDA_DIR/bin:$PATH

# Submodules don't work with calvin because it internally parses git metadata.
# So we have to clone it directly.
RUN git clone --recurse-submodules https://github.com/mees/calvin.git /root/calvin

RUN conda create -n calvin python=3.8
RUN source /opt/conda/bin/activate calvin && \
    pip install setuptools==57.5.0 && \
    cd /root/calvin && \
    ./install.sh && \
    pip install \
    imageio[ffmpeg] \
    moviepy \
    numpy==1.23.0 \
    tqdm \
    tyro \
    websockets \
    msgpack

ENV PYTHONPATH=/app:/app/packages/openpi-client/src

# Download CALVIN dataset, see https://github.com/mees/calvin/blob/main/dataset/download_data.sh
RUN mkdir -p /datasets && cd /datasets && \
    wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip && \
    unzip calvin_debug_dataset.zip && \
    rm calvin_debug_dataset.zip

WORKDIR /app
CMD ["/bin/bash", "-c", "source /opt/conda/bin/activate calvin && python examples/calvin/main.py"]
