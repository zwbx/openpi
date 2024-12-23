from collections.abc import Sequence

import numpy as np

from openpi import transforms


class DroidInputs(transforms.DataTransformFn):
    def __init__(self, action_dim: int, *, delta_action_mask: Sequence[bool] | None = None):
        self._action_dim = action_dim
        self._delta_action_mask = delta_action_mask

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([data["observation/joint_position"], data["observation/gripper_position"]], axis=1)
        state = transforms.pad_to_dim(state, self._action_dim)

        base_image = data["observation/exterior_image_1_left"]

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": data["observation/exterior_image_1_left"],
                "left_wrist_0_rgb": data["observation/wrist_image_left"],
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


class DroidOutputs(transforms.DataTransformFn):
    def __init__(self, *, delta_action_mask: Sequence[bool] | None = None):
        self._delta_action_mask = delta_action_mask

    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        actions = np.asarray(data["actions"][..., :8])

        # Apply the delta action mask.
        if self._delta_action_mask is not None:
            state = np.asarray(data["state"][..., :8])
            mask = np.asarray(self._delta_action_mask[:8])
            actions = actions + np.expand_dims(np.where(mask, state, 0), axis=-2)

        return {"actions": actions}
