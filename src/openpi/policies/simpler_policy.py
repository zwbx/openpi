import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model



def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image



@dataclasses.dataclass(frozen=True)
class SimplerInputs(transforms.DataTransformFn):
    """
    Simpler input processing for basic datasets.
    """
    
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) format
        base_image = _parse_image(data["image"])

        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                # "left_wrist_0_rgb": np.zeros_like(base_image),
                # "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # "left_wrist_0_rgb": np.False_,
                # "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SimplerOutputs(transforms.DataTransformFn):
    """
    Simpler output processing for basic datasets.
    """

    def __call__(self, data: dict) -> dict:
        # Return all actions without dimension limiting
        return {"actions": np.asarray(data["actions"])}
