from collections.abc import Sequence
import logging

import torch
import kornia.augmentation as K

from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)

# Kornia augmentation pipeline cache（按设备 + 视角类型存储）
# same_on_batch=False ensures each sample in batch gets different augmentation
_train_aug_pipelines: dict[str, K.AugmentationSequential] = {}


def _pipeline_cache_key(device: torch.device | None, view_type: str) -> str:
    device_key = str(device) if device is not None else "cpu"
    return f"{view_type}:{device_key}"


def _default_aug_descriptor() -> dict[str, float | int]:
    return {
        "crop_scale": 1.0,
        "crop_ratio": 1.0,
        "rotation_deg": 0.0,
        "flip": 0,
        "brightness": 1.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "hue": 0.0,
    }


def _format_aug_descriptor(descriptor: dict[str, float | int]) -> str:
    return (
        f"crop_{descriptor['crop_scale']:.2f}_ratio_{descriptor['crop_ratio']:.2f}_"
        f"rot_{descriptor['rotation_deg']:.1f}_flip_{descriptor['flip']}_"
        f"b_{descriptor['brightness']:.2f}_c_{descriptor['contrast']:.2f}_"
        f"s_{descriptor['saturation']:.2f}_h_{descriptor['hue']:.2f}"
    )


def get_train_aug_pipeline(device: torch.device | None, view_type: str = "default"):
    """Get or create the training augmentation pipeline for the given device and view type."""

    cache_key = _pipeline_cache_key(device, view_type)
    pipeline = _train_aug_pipelines.get(cache_key)
    if pipeline is None:
        if view_type == "default":
            pipeline = K.AugmentationSequential(
                K.RandomResizedCrop(
                    size=IMAGE_RESOLUTION,
                    scale=(0.8, 1.0),
                    ratio=(0.95, 1.05),
                    same_on_batch=False,
                    p=0.3,
                ),
                K.RandomRotation(
                    degrees=20.0,
                    same_on_batch=False,
                    p=0.3,
                ),
                K.RandomHorizontalFlip(
                    same_on_batch=False,
                    p=0.1,
                ),
                K.ColorJitter(
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    same_on_batch=False,
                    p=0.8,
                ),
                data_keys=["input"],
                same_on_batch=False,
            )
        elif view_type == "wrist":
            pipeline = K.AugmentationSequential(
                K.ColorJitter(
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    same_on_batch=False,
                    p=0.8,
                ),
                data_keys=["input"],
                same_on_batch=False,
            )
        else:
            raise ValueError(f"Unsupported view_type '{view_type}' for augmentation pipeline")

        target_device = device if device is not None else torch.device("cpu")
        pipeline = pipeline.to(target_device)
        _train_aug_pipelines[cache_key] = pipeline

    return pipeline


def _collect_aug_params(
    pipeline: K.AugmentationSequential,
    batch_size: int,
    view_type: str,
) -> list[dict[str, float | int]]:
    """Extract per-sample augmentation参数，用于构建embodiment key。

    TODO: wrist 视角目前只返回颜色扰动，后续如果需要可在保持几何稳定的前提下记录更多信息。
    """

    descriptors = [_default_aug_descriptor() for _ in range(batch_size)]

    for module in pipeline.children():
        params = getattr(module, "_params", None)
        if not params:
            continue

        batch_prob = params.get("batch_prob")
        if batch_prob is None:
            applied_mask = torch.ones(batch_size, dtype=torch.bool)
        else:
            applied_mask = batch_prob.detach().bool().cpu().reshape(batch_size, -1)
        applied_mask = applied_mask[:, 0]
        applied_list = applied_mask.tolist()

        if isinstance(module, K.RandomResizedCrop):
            if view_type == "wrist":
                continue  # wrist 不做几何增广
            scale = params.get("scale")
            ratio = params.get("ratio")
            if scale is not None:
                scale = scale.detach().cpu().reshape(batch_size, -1)
            if ratio is not None:
                ratio = ratio.detach().cpu().reshape(batch_size, -1)

            for idx in range(batch_size):
                if idx >= len(descriptors) or not applied_list[idx]:
                    continue
                if scale is not None:
                    descriptors[idx]["crop_scale"] = float(scale[idx].mean().item())
                if ratio is not None:
                    descriptors[idx]["crop_ratio"] = float(ratio[idx].mean().item())

        elif isinstance(module, K.RandomRotation):
            if view_type == "wrist":
                continue
            angle = params.get("angle")
            if angle is not None:
                angle = angle.detach().cpu().reshape(batch_size, -1)
                for idx in range(batch_size):
                    if idx >= len(descriptors) or not applied_list[idx]:
                        continue
                    descriptors[idx]["rotation_deg"] = float(angle[idx].mean().item())

        elif isinstance(module, K.RandomHorizontalFlip):
            if view_type == "wrist":
                continue
            for idx in range(batch_size):
                if idx >= len(descriptors) or not applied_list[idx]:
                    continue
                descriptors[idx]["flip"] = 1

        elif isinstance(module, K.ColorJitter):
            brightness = params.get("brightness_factor")
            contrast = params.get("contrast_factor")
            saturation = params.get("saturation_factor")
            hue = params.get("hue_factor")

            if brightness is not None:
                brightness = brightness.detach().cpu().reshape(batch_size, -1)
            if contrast is not None:
                contrast = contrast.detach().cpu().reshape(batch_size, -1)
            if saturation is not None:
                saturation = saturation.detach().cpu().reshape(batch_size, -1)
            if hue is not None:
                hue = hue.detach().cpu().reshape(batch_size, -1)

            for idx in range(batch_size):
                if idx >= len(descriptors) or not applied_list[idx]:
                    continue
                if brightness is not None:
                    descriptors[idx]["brightness"] = float(brightness[idx].mean().item())
                if contrast is not None:
                    descriptors[idx]["contrast"] = float(contrast[idx].mean().item())
                if saturation is not None:
                    descriptors[idx]["saturation"] = float(saturation[idx].mean().item())
                if hue is not None:
                    descriptors[idx]["hue"] = float(hue[idx].mean().item())

    return descriptors


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    aug_config: dict | list[dict] | None = None,  # Kept for backward compatibility, not used with Kornia
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    return_aug_params: bool = False,  # New parameter to return augmentation parameters
):
    """Preprocess observation with Kornia-based per-sample augmentation.

    Args:
        observation: Input observation
        train: Whether in training mode (if True, applies Kornia augmentations)
        aug_config: Deprecated, kept for backward compatibility (Kornia handles augmentation automatically)
        image_keys: Keys for images to process
        image_resolution: Target image resolution
        return_aug_params: If True, return augmentation parameters from Kornia

    Returns:
        If return_aug_params=False:
            Preprocessed observation with augmentations applied
        If return_aug_params=True:
            (processed_obs, {
                "params": List[Dict[str, descriptor]],  # 每个样本的视角 -> 增广参数
                "keys": List[str],  # 聚合后的字符串 key，用于构建 embodiment_keys
            })
    """
    # 根据实际observation中的图像动态设置image_keys
    if observation.images:
        image_keys = list(observation.images.keys())

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]
    out_images = {}
    per_sample_view_params: list[dict[str, dict[str, float | int]]] | None = None
    per_sample_key_components: list[list[str]] | None = None

    for key in image_keys:
        image = observation.images[key]
        is_wrist_view = "wrist" in key

        # Handle both [B, C, H, W] and [B, H, W, C] formats
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            pipeline = get_train_aug_pipeline(image.device, view_type="wrist" if is_wrist_view else "default")
            pipeline.train()

            # Convert from [-1, 1] to [0, 1] for Kornia
            image = image / 2.0 + 0.5

            # Convert from [B, H, W, C] to [B, C, H, W] for Kornia
            image = image.permute(0, 3, 1, 2)

            # Apply Kornia augmentations (automatically per-sample)
            # Each sample in the batch gets different random augmentation parameters
            image = pipeline(image)

            if return_aug_params:
                batch_size = image.shape[0]
                params = _collect_aug_params(
                    pipeline,
                    batch_size,
                    "wrist" if is_wrist_view else "default",
                )

                if per_sample_view_params is None:
                    per_sample_view_params = [{} for _ in range(batch_size)]
                    per_sample_key_components = [[] for _ in range(batch_size)]

                if is_wrist_view:
                    # TODO: wrist 视角仅记录颜色扰动，如需纳入 key 可在此扩展策略
                    for idx, descriptor in enumerate(params):
                        per_sample_view_params[idx][key] = descriptor
                else:
                    for idx, descriptor in enumerate(params):
                        per_sample_view_params[idx][key] = descriptor
                        per_sample_key_components[idx].append(
                            f"{key}:{_format_aug_descriptor(descriptor)}"
                        )

            # Convert back to [B, H, W, C]
            image = image.permute(0, 2, 3, 1)

            # Clamp and convert back to [-1, 1]
            image = torch.clamp(image, 0, 1)
            image = image * 2.0 - 1.0

        # Convert back to [B, C, H, W] format if it was originally channels-first
        # if is_channels_first:
        image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]
    
    # HACK
    # Ensure state is float32 to avoid unintended float64 promotion downstream
    state = observation.state
    if isinstance(state, torch.Tensor) and state.dtype != torch.float32:
        state = state.to(torch.float32)

    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    processed_obs = SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )

    if return_aug_params:
        if per_sample_view_params is None:
            if out_images:
                batch_size = next(iter(out_images.values())).shape[0]
            else:
                state_shape = observation.state.shape
                batch_size = state_shape[0] if len(state_shape) > 0 else 1
            per_sample_view_params = [{} for _ in range(batch_size)]
            per_sample_key_components = [[] for _ in range(batch_size)]

        keys = [
            "|".join(components) if components else "no_geom_aug"
            for components in per_sample_key_components
        ]

        aug_metadata = {
            "params": per_sample_view_params,
            "keys": keys,
        }
        return processed_obs, aug_metadata

    return processed_obs
