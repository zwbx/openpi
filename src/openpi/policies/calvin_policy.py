import jax.numpy as jnp

from openpi import transforms


class CalvinInputs(transforms.DataTransformFn):
    def __init__(self, action_dim: int):
        self._action_dim = action_dim

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["observation/state"], self._action_dim)

        inputs = {
            "state": state,
            "image": {
                "rgb_static": data["observation/rgb_static"],
                "rgb_gripper": data["observation/rgb_gripper"],
            },
            "image_mask": {
                "rgb_static": jnp.ones(1, dtype=jnp.bool_),
                "rgb_gripper": jnp.ones(1, dtype=jnp.bool_),
            },
        }

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


class CalvinOutputs(transforms.DataTransformFn):
    def __init__(self):
        pass

    def __call__(self, data: dict) -> dict:
        # Only return the first 15 dims.
        actions = jnp.asarray(data["actions"][..., :15])
        return {"actions": actions}
