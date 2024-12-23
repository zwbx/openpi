import abc
import dataclasses
from typing import TypeAlias

from flax import struct
import flax.linen as nn
import numpy as np

from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class Observation:
    """Holds observations, i.e., inputs to the model."""

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[at.Array, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[at.Array, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[at.Array, "*b s"]
    # Tokenized prompt.
    tokenized_prompt: at.Int[at.Array, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Int[at.Array, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[at.ArrayLike]) -> "Observation":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
        )

    def to_dict(self) -> at.PyTree[at.ArrayLike]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        # TODO(ury): This is awkward. Adjust the names to be the same.
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


Actions: TypeAlias = at.Float[at.ArrayLike, "*b ah ad"]


class BaseModule(nn.Module, abc.ABC):
    @at.typecheck
    @abc.abstractmethod
    def compute_loss(
        self,
        obs: Observation,
        target_actions: Actions,
        *,
        timestep: at.Float[at.Array, " b"] | None = None,
    ) -> at.Float[at.Array, "b ah"]: ...

    @at.typecheck
    @abc.abstractmethod
    def sample_actions(
        self,
        action_horizon: int,
        action_dim: int,
        obs: Observation,
        **sample_kwargs,
    ) -> Actions: ...
