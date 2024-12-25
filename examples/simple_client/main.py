import dataclasses
import enum
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    CALVIN = "calvin"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    env: EnvMode = EnvMode.ALOHA_SIM


def main(args: Args) -> None:
    obs_fn = {
        EnvMode.ALOHA: _random_observation_aloha,
        EnvMode.ALOHA_SIM: _random_observation_aloha,
        EnvMode.DROID: _random_observation_droid,
        EnvMode.CALVIN: _random_observation_calvin,
        EnvMode.LIBERO: _random_observation_libero,
    }[args.env]

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    # Send 1 observation to make sure the model is loaded.
    policy.infer(obs_fn())

    start = time.time()
    for _ in range(100):
        policy.infer(obs_fn())
    end = time.time()

    print(f"Total time taken: {end - start}")
    # Note that each inference returns many action chunks.
    print(f"Inference rate: {100 / (end - start)} Hz")


def _random_observation_aloha() -> dict:
    return {
        "qpos": np.ones((14,)),
        "image": np.random.rand(4, 3, 480, 640).astype(np.float32),
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_calvin() -> dict:
    return {
        "observation/state": np.random.rand(15),
        "observation/rgb_static": np.random.rand(4, 3, 480, 640).astype(np.float32),
        "observation/rgb_gripper": np.random.rand(4, 3, 480, 640).astype(np.float32),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.rand(4, 3, 480, 640).astype(np.float32),
        "observation/wrist_image": np.random.rand(4, 3, 480, 640).astype(np.float32),
        "prompt": "do something",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
