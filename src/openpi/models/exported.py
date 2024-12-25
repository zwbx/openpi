"""Functionality to handle internal pi checkpoints.

Used to test internal pi checkpoints and provides utilities to convert them to openpi checkpoints.
"""

import pathlib
from typing import Any

import flax.serialization
import flax.struct as struct
import jax
import jax.export
import jax.numpy as jnp
import orbax.checkpoint as ocp
from typing_extensions import override

from openpi.models import common
from openpi.models import model as _model
from openpi.shared import image_tools
from openpi.shared import normalize as _normalize
import openpi.shared.array_typing as at
import openpi.shared.download as download


def convert_to_openpi(
    ckpt_dir: pathlib.Path | str, processor: str, out_dir: pathlib.Path | str, param_path: str = "decoder"
) -> None:
    """Convert a monopi checkpoint to an openpi checkpoint."""
    out_dir = pathlib.Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load params and norm stats.
    ckpt_dir = download.maybe_download(str(ckpt_dir))
    sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    params = _load_params(ckpt_dir, sharding=sharding)
    norm_stats = _import_norm_stats(ckpt_dir, processor)

    for part in param_path.split("/"):
        if part not in params:
            raise ValueError(f"{part} not found in the checkpoint. Available keys: {list(params)}")
        params = params[part]

    # Load the monopi model.
    # Save params.
    ckpt = ocp.StandardCheckpointer()
    ckpt.save(out_dir / "params", {"params": params})
    ckpt.wait_until_finished()

    # Save norm stats.
    _normalize.save(out_dir / "assets", norm_stats)


@struct.dataclass
class PiModel(_model.BaseModel):
    """A model loaded from a monopi checkpoint model directory."""

    params: at.Params

    exported: jax.export.Exported = struct.field(pytree_node=False)
    example_spec: Any = struct.field(pytree_node=False)
    sample_spec: Any = struct.field(pytree_node=False)
    ckpt_dir: pathlib.Path = struct.field(pytree_node=False)

    @classmethod
    def from_checkpoint(cls, ckpt_dir: pathlib.Path | str) -> "PiModel":
        """Load a model from a monopi model checkpoint directory. Must point at the "model" sub-directory."""
        ckpt_dir = download.maybe_download(str(ckpt_dir))
        with (ckpt_dir / "graph").open("rb") as f:
            exported = jax.export.deserialize(f.read())

        input_spec = jax.tree.unflatten(exported.in_tree, exported.in_avals)[0]
        params = _load_params(ckpt_dir, input_spec[0])
        example_spec = input_spec[2]
        sample_spec = input_spec[3]

        # Extract the action properties from the output spec.
        output_spec = jax.tree.unflatten(exported.out_tree, exported.out_avals)
        actions_spec = output_spec["actions"]
        action_horizon, action_dim = actions_spec.shape

        max_token_len = example_spec["prompt_tokens"].shape[-1]

        return cls(
            params=params,
            exported=exported,
            example_spec=example_spec,
            sample_spec=sample_spec,
            ckpt_dir=ckpt_dir,
            action_horizon=action_horizon,
            action_dim=action_dim,
            max_token_len=max_token_len,
        )

    @jax.jit
    @override
    def sample_actions(self, rng: at.KeyArrayLike, observation: common.Observation, **sample_kwargs) -> common.Actions:
        if observation.state.ndim == 2 and observation.state.shape[0] != 1:
            raise ValueError("Only batch_size=1 is supported.")

        # Convert to the example format.
        example = _obs_to_example(observation, self.example_spec)
        example = _unbatch(example)

        # Resize the input images if needed.
        def resize_if_needed(key, image):
            target_shape = self.example_spec["image"][key].shape
            if len(target_shape) == 3 and image.shape != target_shape:
                return image_tools.resize_with_pad(image, *target_shape[-3:-1])
            return image

        example["image"] = {key: resize_if_needed(key, value) for key, value in example["image"].items()}

        if set(sample_kwargs) != set(self.sample_spec):
            raise ValueError(
                f"Sample args {list(sample_kwargs)} do not match the expected args {list(self.sample_spec)}"
            )

        rng_data = jax.random.key_data(rng)
        result = self.exported.call(self.params, rng_data, example, sample_kwargs)

        return _make_batch(result)["actions"]

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: common.Observation,
        actions: common.Actions,
        *,
        train: bool = False,
        params: at.Params | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        raise NotImplementedError("Not implemented.")

    def fake_obs(self) -> common.Observation:
        example = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), self.example_spec)
        return _example_to_obs(_make_batch(example))

    def norm_stats(self, processor_name: str) -> dict[str, _normalize.NormStats]:
        """Load the norm stats from the checkpoint."""
        return _import_norm_stats(self.ckpt_dir, processor_name)

    def processor_names(self) -> list[str]:
        """List of processor names available in the checkpoint."""
        processor_dir = self.ckpt_dir / "processors"
        return [x.name for x in processor_dir.iterdir() if x.is_dir()]

    def set_module(self, module: common.BaseModule, param_path: str) -> _model.Model:
        """Creates a new model that uses the same parameters but a different module.

        Args:
            module: The module to use for the model.
            param_path: Location of the parameter sub-tree that should be loaded (e.g., decoder).
                Can include "/" to support nesting.

        Returns:
            A new model with the parameters loaded from the checkpoint.
        """
        params = self.params
        for part in param_path.split("/"):
            if part not in params:
                raise ValueError(f"{part} not found in the checkpoint. Available keys: {list(params)}")
            params = params[part]
        return _model.Model(
            module=module,
            params=params,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.max_token_len,
        )


def _load_params(
    path: pathlib.Path, params_spec: at.PyTree | None = None, sharding: jax.sharding.Sharding | None = None
):
    if sharding is None:
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def to_restore_args(tree):
        return jax.tree.map(lambda x: ocp.ArrayRestoreArgs(dtype=x.dtype, sharding=sharding), tree)

    with ocp.PyTreeCheckpointer() as ckptr:
        if params_spec is None:
            params_spec = ckptr.metadata(path)["params"]
        item = {"params": params_spec}
        return ckptr.restore(
            path,
            args=ocp.args.PyTreeRestore(
                item=item,
                restore_args=to_restore_args(item),
                # This is needed to read a partial checkpoint.
                transforms={},
            ),
        )["params"]


def _obs_to_example(obs: common.Observation, example_spec: dict) -> dict:
    def to_uint8(v):
        return (255.0 * (v + 1.0) / 2.0).astype(jnp.uint8)

    images = {k: to_uint8(v) for k, v in obs.images.items()}
    image_masks = {f"{k}_mask": v for k, v in obs.image_masks.items()}

    result = {
        "image": {**images, **image_masks},
        "state": obs.state,
        "prompt_tokens": obs.tokenized_prompt,
    }

    # NOTE(ury): This is used to support the new version with DCT co-training.
    if "mask_prompt_input" in example_spec:
        allow_action_diffusion_attention = example_spec["allow_action_diffusion_attention"]
        mask_ar = example_spec["mask_ar"]

        result = {
            **result,
            "mask_prompt_input": obs.tokenized_prompt_mask,
            # NOTE(ury): These values are likely wrong. Put something for now
            # to make sure that the model doesn't crash.
            "allow_action_diffusion_attention": _make_batch(
                jnp.zeros(allow_action_diffusion_attention.shape, allow_action_diffusion_attention.dtype)
            ),
            "mask_ar": _make_batch(jnp.ones(mask_ar.shape, mask_ar.dtype)),
        }
    else:
        result = {
            **result,
            "mask_input": obs.tokenized_prompt_mask,
        }

    return result


def _example_to_obs(example: dict) -> common.Observation:
    images, image_masks = {}, {}
    for k, v in example["image"].items():
        if k.endswith("_mask"):
            image_masks[k.removesuffix("_mask")] = v
        else:
            images[k] = v

    # NOTE(ury): This is used to support the new version with DCT co-training.
    if "mask_prompt_input" in example:
        example["mask_input"] = example["mask_prompt_input"]

    return common.Observation.from_dict(
        {
            "image": images,
            "image_mask": image_masks,
            "state": example["state"],
            "tokenized_prompt": example["prompt_tokens"],
            "tokenized_prompt_mask": example["mask_input"],
        }
    )


def _import_norm_stats(ckpt_dir: pathlib.Path | str, processor_name: str) -> dict[str, _normalize.NormStats]:
    ckpt_dir = pathlib.Path(ckpt_dir).resolve()

    path = ckpt_dir / "processors" / processor_name
    if not path.exists():
        raise FileNotFoundError(f"Processor {processor_name} not found in {ckpt_dir}")

    if not (found_files := list(path.glob("*/norm_stats.msgpack"))):
        raise FileNotFoundError(f"norm_stats.msgpack not found in {path}")

    outputs = []

    for file in sorted(found_files):
        with file.open("rb") as f:
            norm_stats = flax.serialization.msgpack_restore(f.read())

        # This is the new Normalize processor.
        if "input_norms" in norm_stats:
            actions = norm_stats["output_norms"]["actions"]
            outputs.append(_normalize.NormStats(mean=actions["mean"], std=actions["std"]))

            state = norm_stats["input_norms"]["state"]
            outputs.append(_normalize.NormStats(mean=state["mean"], std=state["std"]))

        # This is to support the old NormalizeActions / NormalizeState processor combo.
        else:
            outputs.append(_normalize.NormStats(mean=norm_stats["mean"], std=norm_stats["std"]))

    return {
        "actions": outputs[0],
        "state": outputs[1],
    }


def _make_batch(data: at.PyTree) -> at.PyTree:
    return jax.tree.map(lambda x: x[jnp.newaxis, ...], data)


def _unbatch(data: at.PyTree) -> at.PyTree:
    return jax.tree.map(lambda x: x[0, ...], data)
