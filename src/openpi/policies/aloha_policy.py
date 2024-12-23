from collections.abc import Sequence

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    return {
        "qpos": np.ones((14,)),
        "image": np.random.rand(4, 3, 480, 640).astype(np.float32),
    }


class ActInputsRepack(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # images is [..., num_cams, channel, height, width] of type uint8.
        # number of cameras (num_cams) depends on the environment.
        images = np.asarray(data["image"])

        num_cams = images.shape[-4]
        if num_cams == 4:
            cam_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
        elif num_cams == 1:
            cam_names = ["cam_high"]
        else:
            raise ValueError(f"Expected 1 or 4 cameras, got {num_cams}")

        # `images` have shape [..., cam_idx, channel, height, width].
        image_splits = [np.squeeze(x, axis=-4) for x in np.split(images, num_cams, axis=-4)]
        images_dict = dict(zip(cam_names, image_splits, strict=True))

        return {
            "images": images_dict,
            "state": data["qpos"],
        }


class ActOutputsRepack(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"qpos": data["actions"]}


class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [..., channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [..., 14]
    - actions: [..., action_horizon, action_dim]

    Args:
        action_dim: The dimension of the action space.
        delta_action_mask: A boolean mask for the action dimensions. If None, absolute actions are used.
        adapt_to_pi: If true, will adapt the joint and gripper values to match the pi runtime.
    """

    EXPECTED_CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __init__(self, action_dim: int, *, delta_action_mask: Sequence[bool] | None = None, adapt_to_pi: bool = False):
        self._action_dim = action_dim
        self._delta_action_mask = delta_action_mask
        self._adapt_to_pi = adapt_to_pi

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self._adapt_to_pi)

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self._action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]
        batch_size = base_image.shape[:-3]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.ones(batch_size, dtype=np.bool_),
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.ones(batch_size, dtype=np.bool_)
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.zeros(batch_size, dtype=np.bool_)

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self._adapt_to_pi)

            if self._delta_action_mask is not None:
                mask = np.asarray(self._delta_action_mask[:14])
                actions = actions - np.expand_dims(np.where(mask, state[..., :14], 0), axis=-2)

            inputs["actions"] = transforms.pad_to_dim(actions, self._action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy.

    Args:
        delta_action_mask: A boolean mask for the action dimensions. If None, absolute actions are used.
        adapt_to_pi: If true, will adapt the joint and gripper values to match the pi runtime.
    """

    def __init__(self, *, delta_action_mask: Sequence[bool] | None = None, adapt_to_pi: bool = False):
        self._delta_action_mask = delta_action_mask
        self._adapt_to_pi = adapt_to_pi

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][..., :14])

        # Apply the delta action mask.
        if self._delta_action_mask is not None:
            state = np.asarray(data["state"][..., :14])
            mask = np.asarray(self._delta_action_mask[:14])
            actions = actions + np.expand_dims(np.where(mask, state, 0), axis=-2)

        return {"actions": _encode_actions(actions, adapt_to_pi=self._adapt_to_pi)}


def joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [..., channel, height, width] to [..., height, width, channel].
        return einops.rearrange(img, "... c h w -> ... h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = joint_flip_mask() * state

        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[..., 6] = gripper_to_angular(state[..., 6])
        state[..., 13] = gripper_to_angular(state[..., 13])

    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = joint_flip_mask() * actions

        actions[..., 6] = gripper_from_angular(actions[..., 6])
        actions[..., 13] = gripper_from_angular(actions[..., 13])

    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = joint_flip_mask() * actions

        actions[..., 6] = gripper_from_angular_inv(actions[..., 6])
        actions[..., 13] = gripper_from_angular_inv(actions[..., 13])

    return actions
