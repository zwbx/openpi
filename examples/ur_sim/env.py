import argparse
import time
import sys
import logging
logging.getLogger('gymnasium').setLevel(logging.ERROR)


import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from omni.isaac.lab.app import AppLauncher

# add argparse arguments 
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.") 
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

args_cli, other_args = parser.parse_known_args() 
sys.argv = [sys.argv[0]] + other_args # clear out sys.argv for hydra



# launch omniverse app
args_cli.enable_cameras = True
# args_cli.headless = True
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cv2
import h5py
import torch
# torch.set_printoptions(precision=3, threshold=10, edgeitems=3)

import gymnasium
import numpy as np
from pathlib import Path
from openpi_client.runtime import environment as _environment
from typing_extensions import override
from scipy.spatial.transform import Rotation as R
import real2simeval.environments
from real2simeval.splat_render.render import SplatRenderer
from real2simeval.utils import get_transform_from_txt, scalar_last, decrease_brightness 

from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.core.prims import GeometryPrimView
import omni.isaac.lab.utils.math as math


DATA_PATH = Path(__file__).parent.parent.parent.parent.parent / "data"

class URSimEnvironment(_environment.Environment):
    """An environment for an Aloha robot in simulation."""

    def __init__(self, task: str, seed: int = 0) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        self.file = h5py.File("data/episode.h5", "r")
        self.step = 0

        env_cfg = parse_env_cfg(
            task,
            device= args_cli.device,
            num_envs=1,
            use_fabric=True,
        )

        sim_assets = {
                "pi_scene_v2_static": DATA_PATH/"pi_scene_v2",
                "bottle": DATA_PATH/"pi_objects/bottle",
                "plate": DATA_PATH/"pi_objects/plate",
                "robot": DATA_PATH/"pi_robot/",
                }
        env_cfg.setup_scene(sim_assets)


        self._gym = gymnasium.make(task, cfg = env_cfg)

        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0

    @override
    def reset(self) -> None:
        gym_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))

        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = False
        self._episode_reward = 0.0


    @override
    def done(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        return self._last_obs  # type: ignore

    @override
    def apply_action(self, action: dict) -> None:
        action = action["actions"]

        # ur5e = self.file["observation/ur5e/joints/position"][self.step]
        # robotiq = self.file["observation/robotiq_gripper/gripper/position"][self.step]
        # action = np.concatenate([ur5e, robotiq], axis=-1)

        # scale gripper from [0,1] to [-1,1]
        action = action.copy()
        action[-1] = action[-1] * 2 - 1

        #####
        # action = np.zeros(7)
        # action[-1] = -1
        ####

        action = torch.tensor(action, dtype=torch.float32)[None]
        # print(action)
        gym_obs, reward, terminated, truncated, info = self._gym.step(action)


        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = terminated or truncated
        # self._episode_reward = max(self._episode_reward, reward)

        img1 = self._last_obs["observation/base_0_camera/rgb/image"]
        img2 = self._last_obs["observation/wrist_0_camera/rgb/image"]
        big_img = np.concatenate([img1, img2], axis=1)
        cv2.imshow("big_img", cv2.cvtColor(big_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        self.step += 1



    def _convert_observation(self, gym_obs: dict) -> dict:
        # Convert axis order from [H, W, C] --> [C, H, W]
        # img = np.transpose(gym_obs["pixels"]["top"], (2, 0, 1))
        data = {}
        data["observation/ur5e/joints/position"] = gym_obs["policy"]["joints"][:6].detach().cpu().numpy()
        data["observation/robotiq_gripper/gripper/position"] = gym_obs["policy"]["joints"][6:].detach().cpu().numpy()
        data["observation/base_0_camera/rgb/image"] = gym_obs["splat"]["base_cam"]
        data["observation/wrist_0_camera/rgb/image"] = gym_obs["splat"]["wrist_cam"]

        # data["observation/base_0_camera/rgb/image"] = (self.file["observation/base_0_camera/rgb/image_224_224"][self.step])
        # data["observation/wrist_0_camera/rgb/image"] = (self.file["observation/wrist_0_camera/rgb/image_224_224"][self.step])
        # data["observation/base_0_camera/rgb/image"] = (self.file["observation/base_0_camera/rgb/image_256_320"][self.step])
        # data["observation/wrist_0_camera/rgb/image"] = (self.file["observation/wrist_0_camera/rgb/image_256_320"][self.step])
        # data["observation/ur5e/joints/position"] = self.file["observation/ur5e/joints/position"][self.step]
        # data["observation/robotiq_gripper/gripper/position"] = self.file["observation/robotiq_gripper/gripper/position"][self.step]
        #
        # print(data["observation/ur5e/joints/position"])

        return data

