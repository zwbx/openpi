# Run Aloha (Real Robot)

This example demonstrates how to run with a real robot using an [ALOHA setup](https://github.com/tonyzhaozh/aloha).

## Prerequisites

This repo uses a fork of the ALOHA repo, with very minor modifications to use Realsense cameras.

1. Follow the [hardware installation instructions](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#hardware-installation) in the ALOHA repo.
1. Modify the `third_party/aloha/aloha_scripts/realsense_publisher.py` file to use serial numbers for your cameras.

## With Docker

```bash
export SERVER_ARGS="--env ALOHA --default_prompt='toast out of toaster'"
docker compose -f examples/aloha_real/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_real/.venv
source examples/aloha_real/.venv/bin/activate
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e packages/openpi-client

# Run the robot
python examples/aloha_real/main.py
```

Terminal window 2:

```bash
roslaunch --wait aloha ros_nodes.launch
```

Terminal window 3:

```bash
uv run scripts/serve_policy.py --env ALOHA --default_prompt='toast out of toaster'
```

## Model Guide
The Pi0 Base Model is an out-of-the-box model for general tasks. You can find more details in the [technical report](https://www.physicalintelligence.company/download/pi0.pdf).

While we strongly recommend fine-tuning the model to your own data to adapt it to particular tasks, it may be possible to prompt the model to attempt some tasks that were in the pre-training data. For example, below is a video of the model attempting the "toast out of toaster" task.

<p align="center"> 
  <img src="https://github.com/Physical-Intelligence/openpi/blob/main/examples/aloha_real/toast.gif" alt="toast out of toaster"/> 
</p>

## Training on your own Aloha dataset

OpenPI suppports training on data collected in the default aloha hdf5 format. To do so you must first convert the data to the huggingface format. We include `scripts/aloha_hd5.py` to help you do this. Once the dataset is converted, add a new `TrainConfig` to `src/openpi/training/configs.py` and replace repo id with the id assigned to your dataset during conversion.

```python
TrainConfig(
    name=<your-config-name>,
    data=LeRobotAlohaDataConfig(
        repo_id=<your-repo-id>,
        delta_action_mask=[True] * 6 + [False] + [True] * 6 + [False],
    ),
),
```

Run the training script:

```bash
uv run scripts/train.py <your-config-name>
```
