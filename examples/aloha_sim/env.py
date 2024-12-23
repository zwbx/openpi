import gym_aloha  # noqa: F401
import gymnasium
import numpy as np
from openpi_client.runtime import environment as _environment
from typing_extensions import override


class AlohaSimEnvironment(_environment.Environment):
    """An environment for an Aloha robot in simulation."""

    def __init__(self, task: str, obs_type: str = "pixels_agent_pos", seed: int = 0) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        self._gym = gymnasium.make(task, obs_type=obs_type)

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
        gym_obs, reward, terminated, truncated, info = self._gym.step(action["qpos"])
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = terminated or truncated
        self._episode_reward = max(self._episode_reward, reward)

    def _convert_observation(self, gym_obs: dict) -> dict:
        # Convert axis order from [H, W, C] --> [C, H, W]
        img = np.transpose(gym_obs["pixels"]["top"], (2, 0, 1))

        # Add multi-camera dimension, to match the way real aloha provides images as [cam_idx, C, H, W].
        imgs = np.expand_dims(img, axis=0)

        return {
            "qpos": gym_obs["agent_pos"],
            "image": imgs,
        }
