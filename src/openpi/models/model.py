import abc
from collections.abc import Sequence
import dataclasses
import logging
import pathlib

import augmax
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from typing_extensions import override

from openpi.models import common
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


def preprocess_observation(
    rng: at.KeyArrayLike,
    observation: common.Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> common.Observation:
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return common.Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
    )


@struct.dataclass
class BaseModel(abc.ABC):
    # Action space dimension.
    action_dim: int = struct.field(pytree_node=False)
    # Action sequence length.
    action_horizon: int = struct.field(pytree_node=False)
    # Tokenized prompt maximum length.
    max_token_len: int = struct.field(pytree_node=False)

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: common.Observation,
        actions: common.Actions,
        *,
        train: bool = False,
        params: at.Params | None = None,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: common.Observation,
        **sample_kwargs,
    ) -> common.Actions: ...


@struct.dataclass
class Model(BaseModel):
    module: common.BaseModule = struct.field(pytree_node=False)
    params: at.Params | None = None

    def init_params(self, rng: at.KeyArrayLike, observation: common.Observation, actions: common.Actions) -> at.Params:
        """Initialize and return the parameters by tracing the module's `compute_loss` function."""
        preprocess_rng, init_rng = jax.random.split(rng)
        obs = preprocess_observation(preprocess_rng, observation)

        return self.module.init(init_rng, obs, actions, method=self.module.compute_loss)["params"]

    @at.typecheck
    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: common.Observation,
        actions: common.Actions,
        params: at.Params | None = None,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, ""]:
        if params is None:
            if self.params is None:
                raise ValueError(
                    "No parameters found. Either bind the model to parameters using `set_params` or provide params directly."
                )
            params = self.params

        loss_rng, preprocess_rng = jax.random.split(rng)

        obs = preprocess_observation(preprocess_rng, observation, train=train)
        loss_args = (obs, actions)

        return jnp.mean(
            self.module.apply({"params": params}, *loss_args, rngs={"loss": loss_rng}, method=self.module.compute_loss)  # type: ignore
        )

    @jax.jit
    @at.typecheck
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: common.Observation,
        **sample_kwargs,
    ) -> common.Actions:
        if self.params is None:
            raise ValueError(
                "No parameters found. Bind the model to parameters using `set_params` before calling `sample_actions`."
            )

        preprocess_rng, sample_rng = jax.random.split(rng)

        obs = preprocess_observation(preprocess_rng, observation)
        sample_args = (self.action_horizon, self.action_dim, obs)

        actions, _ = self.module.apply(
            {"params": self.params},
            *sample_args,
            rngs={"sample": sample_rng},
            method=self.module.sample_actions,
            mutable=["cache"],
            **sample_kwargs,
        )
        return actions

    def set_params(self, params: at.Params) -> "Model":
        """Returns a copy of the model bound to `params`."""
        return dataclasses.replace(self, params=params)

    def fake_obs(self, batch_size: int = 1) -> common.Observation:
        observation_spec, _ = create_inputs_spec(self, batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> common.Actions:
        _, action_spec = create_inputs_spec(self, batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), action_spec)


def restore_params(
    params_path: pathlib.Path | str,
    *,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint. This works with checkpoints saved with `save_state` during
    openpi training (see `training/checkpoints.py`) as well as pre-trained checkpoints released for openpi.
    """
    params_path = pathlib.Path(params_path).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Model params not found at: {params_path}")

    restore_type = np.ndarray if sharding is None else jax.Array

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        # Use EMA params if they exist, otherwise regular params. See `training.utils.TrainState`.
        params_name = "ema_params" if metadata.get("ema_params") is not None else "params"
        item = {params_name: metadata[params_name]}

        return ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
                transforms={},  # required to load a partial PyTree (e.g., only params from a full TrainState)
            ),
        )[params_name]


def create_inputs_spec(model: Model, *, batch_size: int = 1) -> tuple[common.Observation, at.Float[at.Array, "ah ad"]]:
    image_spec = jax.ShapeDtypeStruct([batch_size, 224, 224, 3], jnp.float32)
    image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

    with at.disable_typechecking():
        observation_spec = common.Observation(
            images={
                "base_0_rgb": image_spec,
                "left_wrist_0_rgb": image_spec,
                "right_wrist_0_rgb": image_spec,
            },
            image_masks={
                "base_0_rgb": image_mask_spec,
                "left_wrist_0_rgb": image_mask_spec,
                "right_wrist_0_rgb": image_mask_spec,
            },
            state=jax.ShapeDtypeStruct([batch_size, model.action_dim], jnp.float32),
            tokenized_prompt=jax.ShapeDtypeStruct([batch_size, model.max_token_len], jnp.int32),
            tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, model.max_token_len], jnp.int32),
        )
    action_spec = jax.ShapeDtypeStruct([batch_size, model.action_horizon, model.action_dim], jnp.float32)

    return observation_spec, action_spec
