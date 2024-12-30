from collections.abc import Sequence

import numpy as np

from openpi import transforms


class URInputs(transforms.DataTransformFn):
    def __init__(self, action_dim: int, *, delta_action_mask: Sequence[bool] | None = None):
        self._action_dim = action_dim
        self._delta_action_mask = delta_action_mask

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([
            data["observation/ur5e/joints/position"], 
            data["observation/robotiq_gripper/gripper/position"]
            ], axis=1)
        state = transforms.pad_to_dim(state, self._action_dim)
        print(f"state: {state}")

        base_image = data["observation/base_0_camera/rgb/image"]

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": data["observation/base_0_camera/rgb/image"],
                "left_wrist_0_rgb": data["observation/wrist_0_camera/rgb/image"],
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.ones(1, dtype=np.bool_),
                "left_wrist_0_rgb": np.ones(1, dtype=np.bool_),
                "right_wrist_0_rgb": np.zeros(1, dtype=np.bool_),
            },
        }

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]


        return inputs


class UROutputs(transforms.DataTransformFn):
    def __init__(self, *, delta_action_mask: Sequence[bool] | None = None):
        self._delta_action_mask = delta_action_mask

    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        actions = np.asarray(data["actions"][..., :7])

        # Apply the delta action mask.
        if self._delta_action_mask is not None:
            state = np.asarray(data["state"][..., :7])
            mask = np.asarray(self._delta_action_mask[:7])
            actions = actions + np.expand_dims(np.where(mask, state, 0), axis=-2)

        return {"actions": actions}
