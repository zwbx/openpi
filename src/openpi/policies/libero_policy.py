import jax.numpy as jnp

from openpi import transforms


class LiberoInputs(transforms.DataTransformFn):
    def __init__(self, action_dim: int):
        self._action_dim = action_dim

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["observation/state"], self._action_dim)

        inputs = {
            "state": state,
            "image": {
                "image": data["observation/image"],
                "wrist_image": data["observation/wrist_image"],
            },
            "image_mask": {
                "image": jnp.ones(1, dtype=jnp.bool_),
                "wrist_image": jnp.ones(1, dtype=jnp.bool_),
            },
        }

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


class LiberoOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        actions = jnp.asarray(data["actions"][..., :8])
        return {"actions": actions}
