import logging
import pathlib

import imageio
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
# import openpi.transforms as transforms
from typing_extensions import override


class VideoSaver(_subscriber.Subscriber):
    """Saves episode data."""

    def __init__(self, out_path: pathlib.Path, subsample: int = 1) -> None:
        self._out_path = out_path
        self._images: list[np.ndarray] = []
        self._subsample = subsample

    @override
    def on_episode_start(self) -> None:
        self._images = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        img1 = observation["observation/base_0_camera/rgb/image"]
        img2 = observation["observation/wrist_0_camera/rgb/image"]
        # img1 = observation["observation/base_0_camera/rgb/image"]
        # img2 = observation["observation/wrist_0_camera/rgb/image"]
        # img3 = observation["base"][0]
        # img4 = observation["wrist"][0]

        big_img = np.concatenate([img1, img2,], axis=1)
        self._images.append(big_img)
        # im = observation["image"][0]  # [C, H, W]
        # im = np.transpose(im, (1, 2, 0))  # [H, W, C]
        # self._images.append(im)

    @override
    def on_episode_end(self) -> None:
        logging.info(f"Saving video to {self._out_path}")
        imageio.mimwrite(
            self._out_path,
            [np.asarray(x) for x in self._images[:: self._subsample]],
            fps=20 // max(1, self._subsample),
        )
