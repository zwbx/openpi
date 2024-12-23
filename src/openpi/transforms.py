from collections.abc import Callable, Sequence
import dataclasses
from typing import Protocol, TypeAlias, TypeVar

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

Batch: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


class DataTransformFn(Protocol):
    def __call__(self, data: Batch) -> Batch: ...


@dataclasses.dataclass(frozen=True)
class Group:
    inputs: Sequence[DataTransformFn] = ()
    outputs: Sequence[DataTransformFn] = ()


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    transforms: Sequence[DataTransformFn]

    def __call__(self, data: Batch) -> Batch:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, item) -> dict:
        flat_item = flatten_dict(item)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: dict) -> dict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = self.prompt
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    strict: bool = False

    def __call__(self, data: dict) -> dict:
        def normalize(x, stats: NormStats):
            return (x - stats.mean) / (stats.std + 1e-6)

        if self.norm_stats is None:
            return data

        return apply_tree(data, self.norm_stats, normalize, strict=self.strict)


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None

    def __call__(self, data: dict) -> dict:
        def unnormalize(x, stats: NormStats):
            return x * (stats.std + 1e-6) + stats.mean

        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(data, self.norm_stats, unnormalize, strict=True)


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, item) -> dict:
        item["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in item["image"].items()}
        return item


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: dict) -> dict:
        data["actions"] = data["actions"][:: self.stride]
        return data


class TokenizePrompt(DataTransformFn):
    # This is the default text prompt for the model.
    DEFAULT_PROMPT = "be a good robot"

    def __init__(self, tokenizer: _tokenizer.Tokenizer, *, default_prompt: str | None = None):
        self._tokenizer = tokenizer
        self._default_prompt = default_prompt or self.DEFAULT_PROMPT

    def __call__(self, data: dict) -> dict:
        if "prompt" not in data:
            batch_size = data["state"].shape[:-1]
            prompt = np.full(batch_size, self._default_prompt)
        else:
            prompt = np.asarray(data.pop("prompt"))

        # TODO(ury): Adjust the tokenizer to take a single element instead.
        shape = prompt.shape
        if len(shape) == 0:
            prompt = prompt[np.newaxis, ...]
        tokens, token_masks = self._tokenizer.tokenize(prompt)
        if len(shape) == 0:
            tokens = tokens[0]
            token_masks = token_masks[0]

        return {**data, "tokenized_prompt": np.asarray(tokens), "tokenized_prompt_mask": np.asarray(token_masks)}


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x
