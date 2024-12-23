from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import common
from openpi.models import model as _model
from openpi.shared import array_typing as at

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
    ):
        self._model = model
        self._input_transform = _transforms.CompositeTransform(transforms)
        self._output_transform = _transforms.CompositeTransform(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {"num_steps": 10}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        inputs = self._input_transform(_make_batch(obs))
        inputs = jax.tree_util.tree_map(lambda x: jnp.asarray(x), inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._model.sample_actions(
                sample_rng, common.Observation.from_dict(inputs), **self._sample_kwargs
            ),
        }
        outputs = self._output_transform(outputs)
        return _unbatch(jax.device_get(outputs))


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results


def _make_batch(data: at.PyTree[np.ndarray]) -> at.PyTree[np.ndarray]:
    def _transform(x: np.ndarray) -> np.ndarray:
        return np.asarray(x)[np.newaxis, ...]

    return jax.tree_util.tree_map(_transform, data)


def _unbatch(data: at.PyTree[np.ndarray]) -> at.PyTree[np.ndarray]:
    return jax.tree_util.tree_map(lambda x: np.asarray(x[0, ...]), data)
