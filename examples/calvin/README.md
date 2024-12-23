# CALVIN Benchmark

This example runs the CALVIN benchmark: https://github.com/mees/calvin

## With Docker

```bash
export SERVER_ARGS="--env CALVIN"
docker compose -f examples/calvin/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
cd $OPENPI_ROOT
conda create -n calvin python=3.8
conda activate calvin

git clone --recurse-submodules https://github.com/mees/calvin.git
cd calvin
pip install setuptools==57.5.0
./install.sh

pip install imageio[ffmpeg] moviepy numpy==1.23.0 tqdm tyro websockets msgpack
ENV PYTHONPATH=$PYTHONPATH:$OPENPI_ROOT/packages/openpi-client/src

# Download CALVIN dataset, see https://github.com/mees/calvin/blob/main/dataset/download_data.sh
export CALVIN_DATASETS_DIR=~/datasets
export CALVIN_DATASET=calvin_debug_dataset
mkdir -p $CALVIN_DATASETS_DIR && cd $CALVIN_DATASETS_DIR
wget http://calvin.cs.uni-freiburg.de/dataset/$CALVIN_DATASET.zip
unzip $CALVIN_DATASET.zip
rm $CALVIN_DATASET.zip

# Run the simulation
cd $OPENPI_ROOT
python examples/calvin/main.py --args.calvin_data_path=$CALVIN_DATASETS_DIR
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env CALVIN
```
