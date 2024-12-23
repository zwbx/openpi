import einops
import numpy as np
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.aloha_real import real_env as _real_env


class AlohaRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(self, render_height: int = 480, render_width: int = 640) -> None:
        self._env = _real_env.make_real_env(init_node=True)
        self._render_height = render_height
        self._render_width = render_width

        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def done(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        images = []
        for cam_name in obs["images"]:
            curr_image = obs["images"][cam_name]
            curr_image = einops.rearrange(curr_image, "h w c -> c h w")
            images.append(curr_image)
        stacked_images = np.stack(images, axis=0).astype(np.uint8)

        # TODO: Consider removing these transformations.
        return {
            "qpos": obs["qpos"],
            "image": stacked_images,
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["qpos"])
